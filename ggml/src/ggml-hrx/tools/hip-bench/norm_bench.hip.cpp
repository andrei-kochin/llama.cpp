#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../kernels/rms_norm_mul_f32.hip.cpp"
#include "../../kernels/add_rms_norm_mul_f32_broadcast.hip.cpp"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

struct options {
    std::string kernel = "rms_wg512";
    int iters = 1000;
    int warmup = 50;
    int repeats = 5;
    int ncols = 2048;
    int nrows = 1;
    bool broadcast_add = false;
    bool check = false;
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
        } else if (std::strncmp(argv[i], "--ncols=", 8) == 0) {
            opts.ncols = std::max(1, parse_int_arg(argv[i], "--ncols="));
        } else if (std::strncmp(argv[i], "--nrows=", 8) == 0) {
            opts.nrows = std::max(1, parse_int_arg(argv[i], "--nrows="));
        } else if (std::strcmp(argv[i], "--broadcast-add") == 0) {
            opts.broadcast_add = true;
        } else if (std::strcmp(argv[i], "--check") == 0) {
            opts.check = true;
        } else {
            std::fprintf(stderr,
                "usage: %s [--kernel=<name>] [--iters=N] [--warmup=N] [--repeats=N] [--ncols=N] [--nrows=N] [--broadcast-add] [--check]\n"
                "kernels: rms_wg512, rms_wg256, rms_wg128, rms_wg64, "
                "add_rms_wg512, add_rms_wg256, add_rms_wg128, add_rms_wg64\n",
                argv[0]);
            std::exit(2);
        }
    }
    return opts;
}

extern "C" __global__ void probe_rms_norm_mul_wg256_f32(
        const float * src, const float * weight, float * dst,
        hrx_rms_norm_mul_constants c) {
    hrx_rms_norm_mul_impl<256>(src, weight, dst, c);
}

extern "C" __global__ void probe_rms_norm_mul_wg64_f32(
        const float * src, const float * weight, float * dst,
        hrx_rms_norm_mul_constants c) {
    hrx_rms_norm_mul_impl<64>(src, weight, dst, c);
}

template <int WG_SIZE>
static __device__ __forceinline__ float add_rms_reduce(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 31;
    const unsigned int wave = tid >> 5;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __builtin_amdgcn_s_barrier();

    sum = lane < waves ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
        if (lane == 0) {
            shared[0] = sum;
        }
    }
    __builtin_amdgcn_s_barrier();
    return shared[0];
}

template <int WG_SIZE>
static __device__ __forceinline__ void add_rms_norm_mul_impl(
        const float * src0, const float * src1, float * add_dst, const float * weight, float * dst,
        hrx_add_rms_norm_mul_f32_broadcast_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float sumsh[(WG_SIZE + 31) / 32];

    const long long i3 = row / (c.ne1 * c.ne2);
    const long long i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    const long long i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;
    const long long wi1 = c.weight_ne1 == 1 ? 0 : i1;
    const long long wi2 = c.weight_ne2 == 1 ? 0 : i2;
    const long long wi3 = c.weight_ne3 == 1 ? 0 : i3;

    const char * src0_row = reinterpret_cast<const char *>(src0) +
        i1 * c.src0_nb1 + i2 * c.src0_nb2 + i3 * c.src0_nb3;
    const char * src1_row = reinterpret_cast<const char *>(src1) +
        (c.src1_nb1 == 0 ? 0 : i1 * c.src1_nb1) +
        (c.src1_nb2 == 0 ? 0 : i2 * c.src1_nb2) +
        (c.src1_nb3 == 0 ? 0 : i3 * c.src1_nb3);
    const char * weight_row = reinterpret_cast<const char *>(weight) +
        wi1 * c.weight_nb1 + wi2 * c.weight_nb2 + wi3 * c.weight_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) +
        i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;
    char * add_dst_row = reinterpret_cast<char *>(add_dst) +
        i1 * c.add_dst_nb1 + i2 * c.add_dst_nb2 + i3 * c.add_dst_nb3;

    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += WG_SIZE) {
        const long long src1_col = c.src1_ne0 == 1 ? 0 : col;
        const float value =
            *reinterpret_cast<const float *>(src0_row + col * sizeof(float)) +
            *reinterpret_cast<const float *>(src1_row + src1_col * sizeof(float));
        sum += value * value;
    }

    const float scale = 1.0f / __builtin_sqrtf(add_rms_reduce<WG_SIZE>(sum, sumsh) / (float) c.ncols + c.eps);
    for (long long col = tid; col < c.ncols; col += WG_SIZE) {
        const long long src1_col = c.src1_ne0 == 1 ? 0 : col;
        const long long wcol = c.weight_ne0 == 1 ? 0 : col;
        const float value =
            *reinterpret_cast<const float *>(src0_row + col * sizeof(float)) +
            *reinterpret_cast<const float *>(src1_row + src1_col * sizeof(float));
        *reinterpret_cast<float *>(add_dst_row + col * sizeof(float)) = value;
        const float weight_value = *reinterpret_cast<const float *>(weight_row + wcol * sizeof(float));
        *reinterpret_cast<float *>(dst_row + col * sizeof(float)) = value * scale * weight_value;
    }
}

#define DEFINE_ADD_RMS(WG) \
extern "C" __global__ void probe_add_rms_norm_mul_wg##WG##_f32( \
        const float * src0, const float * src1, float * add_dst, const float * weight, float * dst, \
        hrx_add_rms_norm_mul_f32_broadcast_constants c) { \
    add_rms_norm_mul_impl<WG>(src0, src1, add_dst, weight, dst, c); \
}

DEFINE_ADD_RMS(256)
DEFINE_ADD_RMS(128)
DEFINE_ADD_RMS(64)

#undef DEFINE_ADD_RMS

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

static float make_value(int index, int seed) {
    const int raw = (index * 17 + seed * 31) % 257;
    return (static_cast<float>(raw) - 128.0f) * 0.00390625f;
}

static void fill_values(std::vector<float> & values, int seed) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = make_value(static_cast<int>(i), seed);
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

static void launch_rms(
        const std::string & kernel,
        const float * src,
        const float * weight,
        float * dst,
        const hrx_rms_norm_mul_constants & c) {
    const dim3 grid(c.nrows, 1, 1);
    if (kernel == "rms_wg512") {
        hrx_rms_norm_mul_f32<<<grid, dim3(512, 1, 1)>>>(src, weight, dst, c);
    } else if (kernel == "rms_wg256") {
        probe_rms_norm_mul_wg256_f32<<<grid, dim3(256, 1, 1)>>>(src, weight, dst, c);
    } else if (kernel == "rms_wg128") {
        hrx_rms_norm_mul_wg128_f32<<<grid, dim3(128, 1, 1)>>>(src, weight, dst, c);
    } else if (kernel == "rms_wg64") {
        probe_rms_norm_mul_wg64_f32<<<grid, dim3(64, 1, 1)>>>(src, weight, dst, c);
    } else {
        std::fprintf(stderr, "unknown RMS kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

static void launch_add_rms(
        const std::string & kernel,
        const float * src0,
        const float * src1,
        float * add_dst,
        const float * weight,
        float * dst,
        const hrx_add_rms_norm_mul_f32_broadcast_constants & c) {
    const dim3 grid(c.nrows, 1, 1);
    if (kernel == "add_rms_wg512") {
        hrx_add_rms_norm_mul_f32_broadcast<<<grid, dim3(512, 1, 1)>>>(src0, src1, add_dst, weight, dst, c);
    } else if (kernel == "add_rms_wg256") {
        probe_add_rms_norm_mul_wg256_f32<<<grid, dim3(256, 1, 1)>>>(src0, src1, add_dst, weight, dst, c);
    } else if (kernel == "add_rms_wg128") {
        probe_add_rms_norm_mul_wg128_f32<<<grid, dim3(128, 1, 1)>>>(src0, src1, add_dst, weight, dst, c);
    } else if (kernel == "add_rms_wg64") {
        probe_add_rms_norm_mul_wg64_f32<<<grid, dim3(64, 1, 1)>>>(src0, src1, add_dst, weight, dst, c);
    } else {
        std::fprintf(stderr, "unknown ADD_RMS kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    HIP_CHECK(hipSetDevice(0));

    const size_t row_values = static_cast<size_t>(opts.ncols) * opts.nrows;
    const size_t add_values = opts.broadcast_add ? 1 : row_values;
    std::vector<float> h_src0(row_values);
    std::vector<float> h_src1(add_values);
    std::vector<float> h_weight(opts.ncols);
    std::vector<float> h_ref(row_values, 0.0f);
    std::vector<float> h_dst(row_values, 0.0f);
    std::vector<float> h_add_ref(row_values, 0.0f);
    std::vector<float> h_add_dst(row_values, 0.0f);
    fill_values(h_src0, 11);
    fill_values(h_src1, 13);
    fill_values(h_weight, 17);

    device_buffer<float> d_src0(row_values);
    device_buffer<float> d_src1(add_values);
    device_buffer<float> d_weight(opts.ncols);
    device_buffer<float> d_ref(row_values);
    device_buffer<float> d_dst(row_values);
    device_buffer<float> d_add_ref(row_values);
    device_buffer<float> d_add_dst(row_values);
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), row_values * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), add_values * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight.ptr, h_weight.data(), opts.ncols * sizeof(float), hipMemcpyHostToDevice));

    hrx_rms_norm_mul_constants rms_c = {
        opts.ncols, opts.nrows, opts.nrows, 1,
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.ncols, 1, 1, 1,
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        1.0e-6f, 0,
    };
    hrx_add_rms_norm_mul_f32_broadcast_constants add_c = {
        opts.ncols, opts.nrows, opts.nrows, 1,
        opts.broadcast_add ? 1LL : static_cast<long long>(opts.ncols),
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.broadcast_add ? 0LL : opts.ncols * static_cast<long long>(sizeof(float)),
        opts.broadcast_add ? 0LL : opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.broadcast_add ? 0LL : opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.ncols, 1, 1, 1,
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.ncols * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        opts.ncols * opts.nrows * static_cast<long long>(sizeof(float)),
        1.0e-6f, 0,
    };

    const bool is_add = opts.kernel.rfind("add_rms_", 0) == 0;
    if (opts.check) {
        if (is_add) {
            launch_add_rms("add_rms_wg512", d_src0.ptr, d_src1.ptr, d_add_ref.ptr, d_weight.ptr, d_ref.ptr, add_c);
            launch_add_rms(opts.kernel, d_src0.ptr, d_src1.ptr, d_add_dst.ptr, d_weight.ptr, d_dst.ptr, add_c);
        } else {
            launch_rms("rms_wg512", d_src0.ptr, d_weight.ptr, d_ref.ptr, rms_c);
            launch_rms(opts.kernel, d_src0.ptr, d_weight.ptr, d_dst.ptr, rms_c);
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_ref.data(), d_ref.ptr, row_values * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, row_values * sizeof(float), hipMemcpyDeviceToHost));
        check_close(h_dst, h_ref, opts.kernel.c_str());
        if (is_add) {
            HIP_CHECK(hipMemcpy(h_add_ref.data(), d_add_ref.ptr, row_values * sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_add_dst.data(), d_add_dst.ptr, row_values * sizeof(float), hipMemcpyDeviceToHost));
            check_close(h_add_dst, h_add_ref, "add_dst");
        }
    }

    for (int i = 0; i < opts.warmup; ++i) {
        if (is_add) {
            launch_add_rms(opts.kernel, d_src0.ptr, d_src1.ptr, d_add_dst.ptr, d_weight.ptr, d_dst.ptr, add_c);
        } else {
            launch_rms(opts.kernel, d_src0.ptr, d_weight.ptr, d_dst.ptr, rms_c);
        }
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
            if (is_add) {
                launch_add_rms(opts.kernel, d_src0.ptr, d_src1.ptr, d_add_dst.ptr, d_weight.ptr, d_dst.ptr, add_c);
            } else {
                launch_rms(opts.kernel, d_src0.ptr, d_weight.ptr, d_dst.ptr, rms_c);
            }
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
    std::printf("kernel=%s ncols=%d nrows=%d broadcast_add=%d mean_us=%.3f stdev_us=%.3f samples=",
        opts.kernel.c_str(), opts.ncols, opts.nrows, opts.broadcast_add ? 1 : 0, mean, std::sqrt(var));
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.3f", i == 0 ? "[" : ",", samples[i]);
    }
    std::printf("]\n");
    return 0;
}
