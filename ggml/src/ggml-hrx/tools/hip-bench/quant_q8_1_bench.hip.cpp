#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../kernels/quantize_q8_1.hip.cpp"
#include "../../kernels/mul_mat_vec_q5_k_q8_1.hip.cpp"
#include "../../kernels/mul_mat_vec_q6_k_q8_1.hip.cpp"

extern "C" __global__ void probe_mul_mat_vec_q5_k_q8_1_wg32_f32(
        const hrx_block_q5_K_q8_1_lhs * src0,
        const hrx_block_q8_1_rhs_q5 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const hrx_block_q5_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    const hrx_block_q8_1_rhs_q5 * src1_col = src1 + col * q8_blocks_per_col;

    const int group = static_cast<int>(tid >> 2);
    const int iqs0 = static_cast<int>((tid & 3u) << 1);
    float sum = 0.0f;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const hrx_block_q5_K_q8_1_lhs * block = row_blocks + block_idx;
        const hrx_block_q8_1_rhs_q5 * rhs = src1_col + block_idx * 8 + group;

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_q5_q8_1(group, block->scales, &sc, &m);
        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
        const float d8 = __half2float(__ushort_as_half(rhs->d));

        const int r0 = *reinterpret_cast<const int *>(rhs->qs + iqs0 * 4);
        const int r1 = *reinterpret_cast<const int *>(rhs->qs + (iqs0 + 1) * 4);
        const int qsum =
            hrx_sudot4_q5_q8_1(hrx_q5_k_pack4(block, group, iqs0), r0) +
            hrx_sudot4_q5_q8_1(hrx_q5_k_pack4(block, group, iqs0 + 1), r1);
        sum += d * d8 * static_cast<float>(qsum);
        if (iqs0 == 0) {
            sum -= min * __half2float(__ushort_as_half(rhs->s));
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void probe_mul_mat_vec_q6_k_q8_1_wg32_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_rhs_q6 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const hrx_block_q6_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    const hrx_block_q8_1_rhs_q6 * src1_col = src1 + col * q8_blocks_per_col;

    const int group = static_cast<int>(tid >> 2);
    const int iqs0 = static_cast<int>((tid & 3u) << 1);
    float sum = 0.0f;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const hrx_block_q6_K_q8_1_lhs * block = row_blocks + block_idx;
        const hrx_block_q8_1_rhs_q6 * rhs = src1_col + block_idx * 8 + group;
        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(hrx_q6_k_scale(block, group, iqs0 * 4));
        const float d8 = __half2float(__ushort_as_half(rhs->d));

        const int r0 = *reinterpret_cast<const int *>(rhs->qs + iqs0 * 4);
        const int r1 = *reinterpret_cast<const int *>(rhs->qs + (iqs0 + 1) * 4);
        const int qsum =
            hrx_sdot4_q6_q8_1_qpack(hrx_q6_k_pack4(block, group, iqs0), r0) +
            hrx_sdot4_q6_q8_1_qpack(hrx_q6_k_pack4(block, group, iqs0 + 1), r1);
        sum += d * d8 * static_cast<float>(qsum);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

static constexpr unsigned short half_small_bits = 0x3800;
static constexpr unsigned short half_tiny_bits = 0x3000;

struct options {
    std::string kernel = "q6_q8_wg32";
    int iters = 1000;
    int warmup = 50;
    int repeats = 5;
    int k = 2048;
    int rows = 2048;
    int cols = 1;
    bool include_quant = false;
    bool check = false;
};

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
        } else if (std::strncmp(argv[i], "--repeats=", 10) == 0) {
            opts.repeats = std::max(1, parse_int_arg(argv[i], "--repeats="));
        } else if (std::strncmp(argv[i], "--k=", 4) == 0) {
            opts.k = std::max(256, parse_int_arg(argv[i], "--k="));
        } else if (std::strncmp(argv[i], "--rows=", 7) == 0) {
            opts.rows = std::max(1, parse_int_arg(argv[i], "--rows="));
        } else if (std::strncmp(argv[i], "--cols=", 7) == 0) {
            opts.cols = std::max(1, parse_int_arg(argv[i], "--cols="));
        } else if (std::strcmp(argv[i], "--include-quant") == 0) {
            opts.include_quant = true;
        } else if (std::strcmp(argv[i], "--check") == 0) {
            opts.check = true;
        } else {
            std::fprintf(stderr,
                "usage: %s [--kernel=q5_q8_wg256|q5_q8_wg32|q6_q8_wg256|q6_q8_wg32] "
                "[--iters=N] [--warmup=N] [--repeats=N] [--k=N] [--rows=N] [--cols=N] "
                "[--include-quant] [--check]\n",
                argv[0]);
            std::exit(2);
        }
    }
    opts.k = (opts.k / 256) * 256;
    return opts;
}

static float make_value(int index, int seed) {
    const int raw = (index * 17 + seed * 31) % 257;
    return (static_cast<float>(raw) - 128.0f) * 0.00390625f;
}

static void fill_q5(std::vector<hrx_block_q5_K_q8_1_lhs> & blocks, int rows, int blocks_per_row) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            hrx_block_q5_K_q8_1_lhs & block = blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
            block.d = half_small_bits;
            block.dmin = half_tiny_bits;
            for (int i = 0; i < 12; ++i) {
                block.scales[i] = static_cast<uint8_t>(1 + ((row * 3 + block_idx * 5 + i) & 15));
            }
            for (int i = 0; i < 32; ++i) {
                block.qh[i] = static_cast<uint8_t>((row * 11 + block_idx * 13 + i * 7) & 0xff);
            }
            for (int i = 0; i < 128; ++i) {
                const uint8_t lo = static_cast<uint8_t>((row * 5 + block_idx * 7 + i) & 15);
                const uint8_t hi = static_cast<uint8_t>((row * 13 + block_idx * 17 + i * 3) & 15);
                block.qs[i] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }
    }
}

static void fill_q6(std::vector<hrx_block_q6_K_q8_1_lhs> & blocks, int rows, int blocks_per_row) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            hrx_block_q6_K_q8_1_lhs & block = blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
            block.d = half_small_bits;
            for (int i = 0; i < 128; ++i) {
                block.ql[i] = static_cast<uint8_t>((row * 5 + block_idx * 7 + i * 3) & 0xff);
            }
            for (int i = 0; i < 64; ++i) {
                block.qh[i] = static_cast<uint8_t>((row * 11 + block_idx * 13 + i * 5) & 0xff);
            }
            for (int i = 0; i < 16; ++i) {
                block.scales[i] = static_cast<int8_t>((i & 1) ? 3 : -2);
            }
        }
    }
}

static void fill_rhs(std::vector<float> & src1, int k, int cols) {
    for (int col = 0; col < cols; ++col) {
        for (int i = 0; i < k; ++i) {
            src1[static_cast<size_t>(col) * k + i] = make_value(i + col * k, 23);
        }
    }
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
    if (max_abs > 5.0e-3 && max_rel > 5.0e-4) {
        std::fprintf(stderr, "%s correctness failed\n", label);
        std::exit(1);
    }
}

template <typename Q8Block>
static void launch_quantize(const float * src1, Q8Block * q8, int k, int cols) {
    hrx_quantize_q8_1_constants c = {
        /* .ne00 = */ k,
        /* .s01  = */ k,
        /* .s02  = */ static_cast<long long>(k) * cols,
        /* .s03  = */ static_cast<long long>(k) * cols,
        /* .ne0  = */ k,
        /* .ne1  = */ cols,
        /* .ne2  = */ 1,
    };
    hrx_quantize_q8_1_f32<<<dim3(k / 32, cols, 1), dim3(32, 1, 1)>>>(src1, reinterpret_cast<hrx_block_q8_1 *>(q8), c);
    HIP_CHECK(hipGetLastError());
}

static void launch_q5(
        const std::string & kernel,
        const hrx_block_q5_K_q8_1_lhs * src0,
        const hrx_block_q8_1_rhs_q5 * src1,
        float * dst,
        int k,
        int rows,
        int cols) {
    if (kernel == "q5_q8_wg256") {
        hrx_mul_mat_vec_q5_k_q8_1_f32<<<dim3(rows, cols, 1), dim3(256, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else if (kernel == "q5_q8_wg32") {
        probe_mul_mat_vec_q5_k_q8_1_wg32_f32<<<dim3(rows, cols, 1), dim3(32, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else {
        std::fprintf(stderr, "unknown Q5 kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

static void launch_q6(
        const std::string & kernel,
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_rhs_q6 * src1,
        float * dst,
        int k,
        int rows,
        int cols) {
    if (kernel == "q6_q8_wg256") {
        hrx_mul_mat_vec_q6_k_q8_1_f32<<<dim3(rows, cols, 1), dim3(256, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_q8_wg32") {
        probe_mul_mat_vec_q6_k_q8_1_wg32_f32<<<dim3(rows, cols, 1), dim3(32, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else {
        std::fprintf(stderr, "unknown Q6 kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

template <typename LhsBlock, typename Q8Block, typename FillFn, typename LaunchFn>
static void run_q8(
        const options & opts,
        const char * family,
        FillFn fill_blocks,
        LaunchFn launch) {
    const int blocks_per_row = opts.k / 256;
    const size_t block_count = static_cast<size_t>(opts.rows) * blocks_per_row;
    const size_t src1_count = static_cast<size_t>(opts.cols) * opts.k;
    const size_t q8_count = static_cast<size_t>(opts.cols) * (opts.k / 32);
    const size_t dst_count = static_cast<size_t>(opts.cols) * opts.rows;

    std::vector<LhsBlock> h_src0(block_count);
    std::vector<float> h_src1(src1_count);
    std::vector<float> h_ref(dst_count, 0.0f);
    std::vector<float> h_dst(dst_count, 0.0f);
    fill_blocks(h_src0, opts.rows, blocks_per_row);
    fill_rhs(h_src1, opts.k, opts.cols);

    device_buffer<LhsBlock> d_src0(block_count);
    device_buffer<float> d_src1(src1_count);
    device_buffer<Q8Block> d_q8(q8_count);
    device_buffer<float> d_ref(dst_count);
    device_buffer<float> d_dst(dst_count);
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), block_count * sizeof(LhsBlock), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), src1_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_ref.ptr, 0, dst_count * sizeof(float)));
    HIP_CHECK(hipMemset(d_dst.ptr, 0, dst_count * sizeof(float)));
    launch_quantize(d_src1.ptr, d_q8.ptr, opts.k, opts.cols);
    HIP_CHECK(hipDeviceSynchronize());

    if (opts.check) {
        launch("ref", d_src0.ptr, d_q8.ptr, d_ref.ptr, opts.k, opts.rows, opts.cols);
        launch(opts.kernel, d_src0.ptr, d_q8.ptr, d_dst.ptr, opts.k, opts.rows, opts.cols);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_ref.data(), d_ref.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
        check_close(h_dst, h_ref, opts.kernel.c_str());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        if (opts.include_quant) {
            launch_quantize(d_src1.ptr, d_q8.ptr, opts.k, opts.cols);
        }
        launch(opts.kernel, d_src0.ptr, d_q8.ptr, d_dst.ptr, opts.k, opts.rows, opts.cols);
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(opts.repeats);
    for (int repeat = 0; repeat < opts.repeats; ++repeat) {
        hipEvent_t start;
        hipEvent_t stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < opts.iters; ++i) {
            if (opts.include_quant) {
                launch_quantize(d_src1.ptr, d_q8.ptr, opts.k, opts.cols);
            }
            launch(opts.kernel, d_src0.ptr, d_q8.ptr, d_dst.ptr, opts.k, opts.rows, opts.cols);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        samples.push_back(ms * 1000.0f / static_cast<float>(opts.iters));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    double mean = 0.0;
    for (float sample : samples) {
        mean += sample;
    }
    mean /= samples.size();
    double var = 0.0;
    for (float sample : samples) {
        const double diff = sample - mean;
        var += diff * diff;
    }
    var /= samples.size();
    std::printf("kernel=%s shape=%s k=%d rows=%d cols=%d include_quant=%d mean_us=%.3f stdev_us=%.3f samples=",
        opts.kernel.c_str(), family, opts.k, opts.rows, opts.cols, opts.include_quant ? 1 : 0,
        mean, std::sqrt(var));
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.3f", i == 0 ? "[" : ",", samples[i]);
    }
    std::printf("]\n");
}

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    HIP_CHECK(hipSetDevice(0));
    if (opts.kernel.rfind("q5_", 0) == 0) {
        run_q8<hrx_block_q5_K_q8_1_lhs, hrx_block_q8_1_rhs_q5>(
            opts, "q5_k_q8_1", fill_q5,
            [](const std::string & kernel, const hrx_block_q5_K_q8_1_lhs * src0,
               const hrx_block_q8_1_rhs_q5 * src1, float * dst, int k, int rows, int cols) {
                launch_q5(kernel == "ref" ? "q5_q8_wg256" : kernel, src0, src1, dst, k, rows, cols);
            });
    } else if (opts.kernel.rfind("q6_", 0) == 0) {
        run_q8<hrx_block_q6_K_q8_1_lhs, hrx_block_q8_1_rhs_q6>(
            opts, "q6_k_q8_1", fill_q6,
            [](const std::string & kernel, const hrx_block_q6_K_q8_1_lhs * src0,
               const hrx_block_q8_1_rhs_q6 * src1, float * dst, int k, int rows, int cols) {
                launch_q6(kernel == "ref" ? "q6_q8_wg256" : kernel, src0, src1, dst, k, rows, cols);
            });
    } else {
        std::fprintf(stderr, "unknown kernel family: %s\n", opts.kernel.c_str());
        return 2;
    }
    return 0;
}
