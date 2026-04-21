#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../kernels/mul_mat_vec_bf16_swiglu.hip.cpp"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

struct options {
    std::string kernel = "rows4_k2048";
    int iters = 1000;
    int warmup = 50;
    int repeats = 5;
    int k = 2048;
    int rows = 512;
    bool check = false;
};

static int parse_int_arg(const char * arg, const char * prefix) {
    const size_t n = std::strlen(prefix);
    return std::strncmp(arg, prefix, n) == 0 ? std::atoi(arg + n) : -1;
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
            opts.k = std::max(16, parse_int_arg(argv[i], "--k="));
        } else if (std::strncmp(argv[i], "--rows=", 7) == 0) {
            opts.rows = std::max(1, parse_int_arg(argv[i], "--rows="));
        } else if (std::strcmp(argv[i], "--check") == 0) {
            opts.check = true;
        } else {
            std::fprintf(stderr,
                "usage: %s [--kernel=<name>] [--iters=N] [--warmup=N] [--repeats=N] [--k=N] [--rows=N] [--check]\n"
                "kernels: rows4_k2048, rows8_k2048, rows2, cols1, generic_wg64, generic_wg128, generic_wg256, wmma16\n",
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

static uint16_t f32_to_bf16_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits = { value };
    const uint32_t lsb = (bits.u >> 16) & 1u;
    bits.u += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits.u >> 16);
}

static float bf16_bits_to_f32(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } bits = { static_cast<uint32_t>(value) << 16 };
    return bits.f;
}

static float make_value(int index, int seed) {
    const int raw = (index * 17 + seed * 31) % 257;
    return (static_cast<float>(raw) - 128.0f) * 0.00390625f;
}

static void launch_kernel(
        const std::string & kernel,
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        int k,
        int rows) {
    if (kernel == "rows4_k2048") {
        hrx_mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_lds_wg256_f32<<<dim3((rows + 3) / 4, 1, 1), dim3(256, 1, 1)>>>(
            gate, up, src1, dst, k, rows, 1);
    } else if (kernel == "rows8_k2048") {
        hrx_mul_mat_vec_bf16_swiglu_rows8_k2048_cols1_lds_wg256_f32<<<dim3((rows + 7) / 8, 1, 1), dim3(256, 1, 1)>>>(
            gate, up, src1, dst, k, rows, 1);
    } else if (kernel == "rows2") {
        hrx_mul_mat_vec_bf16_swiglu_rows2_cols1_f32<<<dim3((rows + 1) / 2, 1, 1), dim3(256, 1, 1)>>>(
            gate, up, src1, dst, k, rows, 1);
    } else if (kernel == "cols1") {
        hrx_mul_mat_vec_bf16_swiglu_cols1_f32<<<dim3(rows, 1, 1), dim3(256, 1, 1)>>>(
            gate, up, src1, dst, k, rows, 1);
    } else if (kernel == "generic_wg64") {
        hrx_mul_mat_vec_bf16_swiglu_wg64_f32<<<dim3(rows, 1, 1), dim3(64, 1, 1)>>>(
            gate, up, src1, dst, k, rows, 1);
    } else if (kernel == "generic_wg128") {
        hrx_mul_mat_vec_bf16_swiglu_wg128_f32<<<dim3(rows, 1, 1), dim3(128, 1, 1)>>>(
            gate, up, src1, dst, k, rows, 1);
    } else if (kernel == "generic_wg256") {
        hrx_mul_mat_vec_bf16_swiglu_f32<<<dim3(rows, 1, 1), dim3(256, 1, 1)>>>(
            gate, up, src1, dst, k, rows, 1);
    } else if (kernel == "wmma16") {
        hrx_mul_mat_vec_bf16_swiglu_wmma16x16_f32<<<dim3((rows + 15) / 16, 1, 1), dim3(32, 1, 1)>>>(
            gate, up, src1, dst, k, rows, 1);
    } else {
        std::fprintf(stderr, "unknown kernel: %s\n", kernel.c_str());
        std::exit(2);
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
    if (max_abs > 7.0e-3 && max_rel > 1.0e-3) {
        std::fprintf(stderr, "%s correctness failed\n", label);
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    const int k = opts.k;
    const int rows = opts.rows;

    std::vector<uint16_t> h_gate(static_cast<size_t>(rows) * k);
    std::vector<uint16_t> h_up(static_cast<size_t>(rows) * k);
    std::vector<float> h_src1(k);
    std::vector<float> h_ref(rows);
    std::vector<float> h_dst(rows);
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < k; ++i) {
            h_gate[static_cast<size_t>(row) * k + i] = f32_to_bf16_bits(make_value(row * k + i, 3));
            h_up[static_cast<size_t>(row) * k + i] = f32_to_bf16_bits(make_value(row * k + i, 11));
        }
    }
    for (int i = 0; i < k; ++i) {
        h_src1[i] = make_value(i, 23);
    }
    for (int row = 0; row < rows; ++row) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            gate_sum += bf16_bits_to_f32(h_gate[static_cast<size_t>(row) * k + i]) * h_src1[i];
            up_sum += bf16_bits_to_f32(h_up[static_cast<size_t>(row) * k + i]) * h_src1[i];
        }
        const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
        h_ref[row] = up_sum * silu_gate;
    }

    device_buffer<uint16_t> d_gate(h_gate.size());
    device_buffer<uint16_t> d_up(h_up.size());
    device_buffer<float> d_src1(h_src1.size());
    device_buffer<float> d_dst(h_dst.size());
    HIP_CHECK(hipMemcpy(d_gate.ptr, h_gate.data(), h_gate.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_up.ptr, h_up.data(), h_up.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), h_src1.size() * sizeof(float), hipMemcpyHostToDevice));

    launch_kernel(opts.kernel, d_gate.ptr, d_up.ptr, d_src1.ptr, d_dst.ptr, k, rows);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    if (opts.check) {
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, h_dst.size() * sizeof(float), hipMemcpyDeviceToHost));
        check_close(h_dst, h_ref, opts.kernel.c_str());
    }

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(opts.repeats));
    for (int r = 0; r < opts.repeats; ++r) {
        for (int i = 0; i < opts.warmup; ++i) {
            launch_kernel(opts.kernel, d_gate.ptr, d_up.ptr, d_src1.ptr, d_dst.ptr, k, rows);
        }
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t start;
        hipEvent_t stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < opts.iters; ++i) {
            launch_kernel(opts.kernel, d_gate.ptr, d_up.ptr, d_src1.ptr, d_dst.ptr, k, rows);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        samples.push_back(ms * 1000.0f / static_cast<float>(opts.iters));
    }
    double sum = 0.0;
    for (float v : samples) {
        sum += v;
    }
    const double mean = sum / samples.size();
    double var = 0.0;
    for (float v : samples) {
        const double d = v - mean;
        var += d * d;
    }
    const double stdev = std::sqrt(var / samples.size());
    std::printf("kernel=%s shape=bf16_swiglu k=%d rows=%d mean_us=%.3f stdev_us=%.3f samples=[",
        opts.kernel.c_str(), k, rows, mean, stdev);
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.3f", i ? "," : "", samples[i]);
    }
    std::printf("]\n");
    return 0;
}
