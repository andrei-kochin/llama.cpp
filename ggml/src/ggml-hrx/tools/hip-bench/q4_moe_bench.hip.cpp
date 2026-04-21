#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../../kernels/mul_mat_id_q4_k_mul.hip.cpp"
#include "../../kernels/mul_mat_id_q4_k_swiglu.hip.cpp"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

static constexpr uint16_t half_one_bits = 0x3c00;
static constexpr uint16_t half_zero_bits = 0x0000;

struct options {
    std::string kernel = "mul_rows2_x16_wg32";
    int iters = 1000;
    int warmup = 50;
    bool check = false;
    bool trace_markers = false;
};

static int parse_int_arg(const char * arg, const char * prefix) {
    const size_t n = std::strlen(prefix);
    if (std::strncmp(arg, prefix, n) != 0) {
        return -1;
    }
    return std::atoi(arg + n);
}

static options parse_options(int argc, char ** argv) {
    options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--kernel=", 9) == 0) {
            opts.kernel = argv[i] + 9;
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            opts.iters = std::max(1, parse_int_arg(argv[i], "--iters="));
        } else if (std::strncmp(argv[i], "--warmup=", 9) == 0) {
            opts.warmup = std::max(0, parse_int_arg(argv[i], "--warmup="));
        } else if (std::strcmp(argv[i], "--check") == 0) {
            opts.check = true;
        } else if (std::strcmp(argv[i], "--trace-markers") == 0) {
            opts.trace_markers = true;
        } else {
            std::fprintf(stderr,
                "usage: %s [--kernel=<name>] [--iters=N] [--warmup=N] [--check]\n"
                "kernels: mul_wg64, mul_packed_wg64, mul_rows2_x16_wg32, "
                "mul_rows2_x16_wg16, mul_packed_2row_wg64, "
                "swiglu_wg64, swiglu_packed_wg64, swiglu_packed_wg32, swiglu_rows2_x16_wg32\n",
                argv[0]);
            std::exit(2);
        }
    }
    return opts;
}

template <typename T>
struct device_buffer {
    T * ptr = nullptr;
    size_t count = 0;

    explicit device_buffer(size_t count) : count(count) {
        HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
    }
    ~device_buffer() {
        if (ptr) {
            (void) hipFree(ptr);
        }
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
};

static void fill_q4_blocks_mul(
        std::vector<hrx_block_q4_K_id_mul> & blocks,
        int n_experts,
        int rows,
        int blocks_per_row) {
    for (int expert = 0; expert < n_experts; ++expert) {
        for (int row = 0; row < rows; ++row) {
            for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
                auto & block = blocks[(static_cast<size_t>(expert) * rows + row) * blocks_per_row + block_idx];
                block.d = half_one_bits;
                block.dmin = half_zero_bits;
                std::memset(block.scales, 1, sizeof(block.scales));
                for (int i = 0; i < 128; ++i) {
                    const uint8_t lo = static_cast<uint8_t>((expert * 3 + row * 5 + block_idx * 7 + i) & 15);
                    const uint8_t hi = static_cast<uint8_t>((expert * 11 + row * 13 + block_idx * 17 + i * 3) & 15);
                    block.qs[i] = static_cast<uint8_t>(lo | (hi << 4));
                }
            }
        }
    }
}

static void fill_q4_blocks_swiglu(
        std::vector<hrx_block_q4_K_id_swiglu> & blocks,
        int n_experts,
        int rows,
        int blocks_per_row,
        int seed) {
    for (int expert = 0; expert < n_experts; ++expert) {
        for (int row = 0; row < rows; ++row) {
            for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
                auto & block = blocks[(static_cast<size_t>(expert) * rows + row) * blocks_per_row + block_idx];
                block.d = half_one_bits;
                block.dmin = half_zero_bits;
                std::memset(block.scales, 1, sizeof(block.scales));
                for (int i = 0; i < 128; ++i) {
                    const uint8_t lo = static_cast<uint8_t>((seed + expert * 3 + row * 5 + block_idx * 7 + i) & 15);
                    const uint8_t hi = static_cast<uint8_t>((seed + expert * 11 + row * 13 + block_idx * 17 + i * 3) & 15);
                    block.qs[i] = static_cast<uint8_t>(lo | (hi << 4));
                }
            }
        }
    }
}

static void fill_src1(std::vector<float> & src1, int k, int n_ids, int n_tokens) {
    for (int token = 0; token < n_tokens; ++token) {
        for (int id = 0; id < n_ids; ++id) {
            for (int i = 0; i < k; ++i) {
                const int raw = (i * 17 + id * 31 + token * 43) % 257;
                src1[(static_cast<size_t>(token) * n_ids + id) * k + i] =
                    (static_cast<float>(raw) - 128.0f) * 0.00390625f;
            }
        }
    }
}

static uint8_t decode_q4(const uint8_t * qs, int element) {
    const int group = element >> 5;
    const int offset = element & 31;
    const uint8_t packed = qs[(group >> 1) * 32 + offset];
    return (group & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 15);
}

static void fill_ids(std::vector<int> & ids, int n_ids, int n_tokens, int n_experts) {
    for (int token = 0; token < n_tokens; ++token) {
        for (int id = 0; id < n_ids; ++id) {
            ids[token * n_ids + id] = (id * 37 + token * 19) % n_experts;
        }
    }
}

static float dot_q4_row_mul(
        const std::vector<hrx_block_q4_K_id_mul> & src0,
        const std::vector<float> & src1,
        int expert,
        int row,
        int id,
        int token,
        int rows,
        int k,
        int n_ids) {
    const int blocks_per_row = k / 256;
    const auto * row_blocks = src0.data() + (static_cast<size_t>(expert) * rows + row) * blocks_per_row;
    const float * y = src1.data() + (static_cast<size_t>(token) * n_ids + id) * k;
    float sum = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const auto & block = row_blocks[block_idx];
        for (int e = 0; e < 256; ++e) {
            sum += static_cast<float>(decode_q4(block.qs, e)) * y[block_idx * 256 + e];
        }
    }
    return sum;
}

static float dot_q4_row_swiglu(
        const std::vector<hrx_block_q4_K_id_swiglu> & src0,
        const std::vector<float> & src1,
        int expert,
        int row,
        int id,
        int token,
        int rows,
        int k,
        int n_ids) {
    const int blocks_per_row = k / 256;
    const auto * row_blocks = src0.data() + (static_cast<size_t>(expert) * rows + row) * blocks_per_row;
    const float * y = src1.data() + (static_cast<size_t>(token) * n_ids + id) * k;
    float sum = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const auto & block = row_blocks[block_idx];
        for (int e = 0; e < 256; ++e) {
            sum += static_cast<float>(decode_q4(block.qs, e)) * y[block_idx * 256 + e];
        }
    }
    return sum;
}

static void check_close(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t bad_idx = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        const double denom = std::max(1.0, std::abs(static_cast<double>(expected[i])));
        const double rel = diff / denom;
        if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            bad_idx = i;
        }
    }
    std::printf("%s check: max_abs=%g max_rel=%g idx=%zu actual=%g expected=%g\n",
        label, max_abs, max_rel, bad_idx, actual[bad_idx], expected[bad_idx]);
    if (max_abs > 2.0e-3 && max_rel > 2.0e-4) {
        std::fprintf(stderr, "%s correctness failed\n", label);
        std::exit(1);
    }
}

static void launch_mul(
        const std::string & kernel,
        const hrx_block_q4_K_id_mul * src0,
        const float * src1,
        const int * ids,
        const float * scale,
        float * dst,
        const hrx_mul_mat_id_q4_k_mul_constants & c) {
    const dim3 block(
        kernel == "mul_rows2_x16_wg32" ? 32 :
        kernel == "mul_rows2_x16_wg16" ? 16 :
        64);
    const dim3 grid(
        kernel == "mul_rows2_x16_wg32" || kernel == "mul_rows2_x16_wg16" || kernel == "mul_packed_2row_wg64" ?
            static_cast<unsigned int>((c.rows + 1) / 2) :
            static_cast<unsigned int>(c.rows),
        static_cast<unsigned int>(c.n_ids * c.n_tokens));
    if (kernel == "mul_wg64") {
        hrx_mul_mat_id_q4_k_mul_wg64_f32<<<grid, block>>>(src0, src1, ids, scale, dst, c);
    } else if (kernel == "mul_packed_wg64") {
        hrx_mul_mat_id_q4_k_mul_packed_wg64_f32<<<grid, block>>>(src0, src1, ids, scale, dst, c);
    } else if (kernel == "mul_rows2_x16_wg32") {
        hrx_mul_mat_id_q4_k_mul_rows2_x16_wg32_f32<<<grid, block>>>(src0, src1, ids, scale, dst, c);
    } else if (kernel == "mul_rows2_x16_wg16") {
        hrx_mul_mat_id_q4_k_mul_rows2_x16_wg16_f32<<<grid, block>>>(src0, src1, ids, scale, dst, c);
    } else if (kernel == "mul_packed_2row_wg64") {
        hrx_mul_mat_id_q4_k_mul_packed_2row_wg64_f32<<<grid, block>>>(src0, src1, ids, scale, dst, c);
    } else {
        std::fprintf(stderr, "unknown mul kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

static void launch_swiglu(
        const std::string & kernel,
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        const hrx_mul_mat_id_q4_k_swiglu_constants & c) {
    const dim3 block(kernel == "swiglu_packed_wg32" || kernel == "swiglu_rows2_x16_wg32" ? 32 : 64);
    const dim3 grid(
        kernel == "swiglu_rows2_x16_wg32" ?
            static_cast<unsigned int>((c.rows + 1) / 2) :
            static_cast<unsigned int>(c.rows),
        static_cast<unsigned int>(c.n_ids * c.n_tokens));
    if (kernel == "swiglu_wg64") {
        hrx_mul_mat_id_q4_k_swiglu_wg64_f32<<<grid, block>>>(gate, up, src1, ids, dst, c);
    } else if (kernel == "swiglu_packed_wg64") {
        hrx_mul_mat_id_q4_k_swiglu_packed_wg64_f32<<<grid, block>>>(gate, up, src1, ids, dst, c);
    } else if (kernel == "swiglu_packed_wg32") {
        hrx_mul_mat_id_q4_k_swiglu_packed_wg32_f32<<<grid, block>>>(gate, up, src1, ids, dst, c);
    } else if (kernel == "swiglu_rows2_x16_wg32") {
        hrx_mul_mat_id_q4_k_swiglu_rows2_x16_wg32_f32<<<grid, block>>>(gate, up, src1, ids, dst, c);
    } else {
        std::fprintf(stderr, "unknown swiglu kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

static void run_mul(const options & opts) {
    constexpr int k = 512;
    constexpr int rows = 2048;
    constexpr int n_ids = 8;
    constexpr int n_tokens = 1;
    constexpr int n_experts = 256;
    constexpr int blocks_per_row = k / 256;
    static_assert(sizeof(hrx_block_q4_K_id_mul) == 144, "unexpected q4_K block size");

    std::vector<hrx_block_q4_K_id_mul> h_src0(static_cast<size_t>(n_experts) * rows * blocks_per_row);
    std::vector<float> h_src1(static_cast<size_t>(n_tokens) * n_ids * k);
    std::vector<int> h_ids(n_tokens * n_ids);
    std::vector<float> h_scale(n_ids);
    std::vector<float> h_dst(static_cast<size_t>(n_tokens) * n_ids * rows, 0.0f);
    std::vector<float> h_expected(h_dst.size(), 0.0f);
    fill_q4_blocks_mul(h_src0, n_experts, rows, blocks_per_row);
    fill_src1(h_src1, k, n_ids, n_tokens);
    fill_ids(h_ids, n_ids, n_tokens, n_experts);
    for (int i = 0; i < n_ids; ++i) {
        h_scale[i] = 0.25f + 0.03125f * static_cast<float>(i);
    }

    device_buffer<hrx_block_q4_K_id_mul> d_src0(h_src0.size());
    device_buffer<float> d_src1(h_src1.size());
    device_buffer<int> d_ids(h_ids.size());
    device_buffer<float> d_scale(h_scale.size());
    device_buffer<float> d_dst(h_dst.size());
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), h_src0.size() * sizeof(h_src0[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), h_src1.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ids.ptr, h_ids.data(), h_ids.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scale.ptr, h_scale.data(), h_scale.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dst.ptr, 0, h_dst.size() * sizeof(float)));

    hrx_mul_mat_id_q4_k_mul_constants c = {
        k, rows, n_ids, n_tokens, n_experts,
        blocks_per_row * static_cast<long long>(sizeof(hrx_block_q4_K_id_mul)),
        rows * blocks_per_row * static_cast<long long>(sizeof(hrx_block_q4_K_id_mul)),
        k * static_cast<long long>(sizeof(float)),
        n_ids * k * static_cast<long long>(sizeof(float)),
        static_cast<long long>(sizeof(int)),
        n_ids * static_cast<long long>(sizeof(int)),
        rows * static_cast<long long>(sizeof(float)),
        rows * n_ids * static_cast<long long>(sizeof(float)),
        static_cast<long long>(sizeof(float)),
    };

    if (opts.check) {
        launch_mul(opts.kernel, d_src0.ptr, d_src1.ptr, d_ids.ptr, d_scale.ptr, d_dst.ptr, c);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, h_dst.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (int token = 0; token < n_tokens; ++token) {
            for (int id = 0; id < n_ids; ++id) {
                const int expert = h_ids[token * n_ids + id];
                for (int row = 0; row < rows; ++row) {
                    h_expected[(static_cast<size_t>(token) * n_ids + id) * rows + row] =
                        dot_q4_row_mul(h_src0, h_src1, expert, row, id, token, rows, k, n_ids) * h_scale[id];
                }
            }
        }
        check_close(h_dst, h_expected, opts.kernel.c_str());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_mul(opts.kernel, d_src0.ptr, d_src1.ptr, d_ids.ptr, d_scale.ptr, d_dst.ptr, c);
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < opts.iters; ++i) {
        launch_mul(opts.kernel, d_src0.ptr, d_src1.ptr, d_ids.ptr, d_scale.ptr, d_dst.ptr, c);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    std::printf("kernel=%s shape=mul_q4_k_m2048_n8_k512 iters=%d avg_us=%.3f\n",
        opts.kernel.c_str(), opts.iters, ms * 1000.0f / opts.iters);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

static void run_swiglu(const options & opts) {
    constexpr int k = 2048;
    constexpr int rows = 512;
    constexpr int n_ids = 8;
    constexpr int n_tokens = 1;
    constexpr int n_experts = 256;
    constexpr int blocks_per_row = k / 256;
    static_assert(sizeof(hrx_block_q4_K_id_swiglu) == 144, "unexpected q4_K block size");

    std::vector<hrx_block_q4_K_id_swiglu> h_gate(static_cast<size_t>(n_experts) * rows * blocks_per_row);
    std::vector<hrx_block_q4_K_id_swiglu> h_up(h_gate.size());
    std::vector<float> h_src1(static_cast<size_t>(n_tokens) * n_ids * k);
    std::vector<int> h_ids(n_tokens * n_ids);
    std::vector<float> h_dst(static_cast<size_t>(n_tokens) * n_ids * rows, 0.0f);
    std::vector<float> h_expected(h_dst.size(), 0.0f);
    fill_q4_blocks_swiglu(h_gate, n_experts, rows, blocks_per_row, 0);
    fill_q4_blocks_swiglu(h_up, n_experts, rows, blocks_per_row, 7);
    fill_src1(h_src1, k, n_ids, n_tokens);
    fill_ids(h_ids, n_ids, n_tokens, n_experts);

    device_buffer<hrx_block_q4_K_id_swiglu> d_gate(h_gate.size());
    device_buffer<hrx_block_q4_K_id_swiglu> d_up(h_up.size());
    device_buffer<float> d_src1(h_src1.size());
    device_buffer<int> d_ids(h_ids.size());
    device_buffer<float> d_dst(h_dst.size());
    HIP_CHECK(hipMemcpy(d_gate.ptr, h_gate.data(), h_gate.size() * sizeof(h_gate[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_up.ptr, h_up.data(), h_up.size() * sizeof(h_up[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), h_src1.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ids.ptr, h_ids.data(), h_ids.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dst.ptr, 0, h_dst.size() * sizeof(float)));

    hrx_mul_mat_id_q4_k_swiglu_constants c = {
        k, rows, n_ids, n_tokens, n_experts,
        blocks_per_row * static_cast<long long>(sizeof(hrx_block_q4_K_id_swiglu)),
        rows * blocks_per_row * static_cast<long long>(sizeof(hrx_block_q4_K_id_swiglu)),
        blocks_per_row * static_cast<long long>(sizeof(hrx_block_q4_K_id_swiglu)),
        rows * blocks_per_row * static_cast<long long>(sizeof(hrx_block_q4_K_id_swiglu)),
        k * static_cast<long long>(sizeof(float)),
        n_ids * k * static_cast<long long>(sizeof(float)),
        static_cast<long long>(sizeof(int)),
        n_ids * static_cast<long long>(sizeof(int)),
        rows * static_cast<long long>(sizeof(float)),
        rows * n_ids * static_cast<long long>(sizeof(float)),
    };

    if (opts.check) {
        launch_swiglu(opts.kernel, d_gate.ptr, d_up.ptr, d_src1.ptr, d_ids.ptr, d_dst.ptr, c);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, h_dst.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (int token = 0; token < n_tokens; ++token) {
            for (int id = 0; id < n_ids; ++id) {
                const int expert = h_ids[token * n_ids + id];
                for (int row = 0; row < rows; ++row) {
                    const float gate = dot_q4_row_swiglu(h_gate, h_src1, expert, row, id, token, rows, k, n_ids);
                    const float up = dot_q4_row_swiglu(h_up, h_src1, expert, row, id, token, rows, k, n_ids);
                    const float silu_gate = gate / (1.0f + std::exp(-gate));
                    h_expected[(static_cast<size_t>(token) * n_ids + id) * rows + row] = up * silu_gate;
                }
            }
        }
        check_close(h_dst, h_expected, opts.kernel.c_str());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_swiglu(opts.kernel, d_gate.ptr, d_up.ptr, d_src1.ptr, d_ids.ptr, d_dst.ptr, c);
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < opts.iters; ++i) {
        launch_swiglu(opts.kernel, d_gate.ptr, d_up.ptr, d_src1.ptr, d_ids.ptr, d_dst.ptr, c);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    std::printf("kernel=%s shape=swiglu_q4_k_m512_n8_k2048 iters=%d avg_us=%.3f\n",
        opts.kernel.c_str(), opts.iters, ms * 1000.0f / opts.iters);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

int main(int argc, char ** argv) {
    options opts = parse_options(argc, argv);
    HIP_CHECK(hipSetDevice(0));
    if (opts.kernel.rfind("mul_", 0) == 0) {
        run_mul(opts);
    } else if (opts.kernel.rfind("swiglu_", 0) == 0) {
        run_swiglu(opts);
    } else {
        std::fprintf(stderr, "unknown kernel family: %s\n", opts.kernel.c_str());
        return 2;
    }
    return 0;
}
