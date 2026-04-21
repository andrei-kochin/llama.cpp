#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../kernels/mul_mat_vec_q8_0.hip.cpp"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

static constexpr unsigned short half_small_bits = 0x3800;

struct options {
    std::string kernel = "q8_add_wg256";
    int iters = 1000;
    int warmup = 50;
    int repeats = 5;
    int k = 4096;
    int rows = 2048;
    int cols = 1;
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
        } else if (std::strncmp(argv[i], "--k=", 4) == 0) {
            opts.k = std::max(32, parse_int_arg(argv[i], "--k="));
        } else if (std::strncmp(argv[i], "--rows=", 7) == 0) {
            opts.rows = std::max(1, parse_int_arg(argv[i], "--rows="));
        } else if (std::strncmp(argv[i], "--cols=", 7) == 0) {
            opts.cols = std::max(1, parse_int_arg(argv[i], "--cols="));
        } else if (std::strcmp(argv[i], "--check") == 0) {
            opts.check = true;
        } else {
            std::fprintf(stderr,
                "usage: %s [--kernel=<name>] [--iters=N] [--warmup=N] [--repeats=N] [--k=N] [--rows=N] [--cols=N] [--check]\n"
                "kernels: q8_add_wg256, q8_add_wg128, q8_add_wg64, q8_add_wg32, "
                "q8_add_rows2_wg128, q8_add_rows2_wg64, q8_add_rows2_wg32, "
                "q8_add_rows4_wg128, q8_add_rows4_wg64, q8_add_rows4_wg32\n",
                argv[0]);
            std::exit(2);
        }
    }
    opts.k = (opts.k / 32) * 32;
    return opts;
}

template <int WG, int N>
static __device__ __forceinline__ void reduce_rows(float (&sum)[N], float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = WG / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            sum[i] += __shfl_down(sum[i], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            shared[i * waves + wave] = sum[i];
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < N; ++i) {
        sum[i] = lane < waves ? shared[i * waves + lane] : 0.0f;
    }
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                sum[i] += __shfl_down(sum[i], offset);
            }
        }
    }
}

template <int WG, int ROWS_PER_WORKGROUP>
static __device__ __forceinline__ void q8_0_add_rows_impl(
        const hrx_block_q8_0 * src0,
        const float * src1,
        const float * bias,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * ROWS_PER_WORKGROUP;
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[ROWS_PER_WORKGROUP * (WG / 32)];

    const long long blocks_per_row = k / 32;
    const float * src1_col = src1 + col * k;
    float sum[ROWS_PER_WORKGROUP] = {};

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += WG / 8) {
        const long long src_base = block_idx * 32 + in_block_base;
        const float4 rhs = *reinterpret_cast<const float4 *>(src1_col + src_base);
#pragma unroll
        for (int r = 0; r < ROWS_PER_WORKGROUP; ++r) {
            if (row0 + r < rows) {
                const hrx_block_q8_0 * block = src0 + (row0 + r) * blocks_per_row + block_idx;
                const float d = __half2float(__ushort_as_half(block->d));
                sum[r] += d * static_cast<float>(block->qs[in_block_base + 0]) * rhs.x;
                sum[r] += d * static_cast<float>(block->qs[in_block_base + 1]) * rhs.y;
                sum[r] += d * static_cast<float>(block->qs[in_block_base + 2]) * rhs.z;
                sum[r] += d * static_cast<float>(block->qs[in_block_base + 3]) * rhs.w;
            }
        }
    }

    reduce_rows<WG>(sum, sumsh);
    if (tid == 0) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_WORKGROUP; ++r) {
            if (row0 + r < rows) {
                const long long out_idx = col * rows + row0 + r;
                dst[out_idx] = sum[r] + bias[out_idx];
            }
        }
    }
}

#define DEFINE_Q8_ADD_ROWS(WG, ROWS) \
extern "C" __global__ void probe_q8_0_add_rows##ROWS##_wg##WG##_f32( \
        const hrx_block_q8_0 * src0, const float * src1, const float * bias, float * dst, \
        long long k, long long rows, long long cols) { \
    q8_0_add_rows_impl<WG, ROWS>(src0, src1, bias, dst, k, rows, cols); \
}

DEFINE_Q8_ADD_ROWS(128, 1)
DEFINE_Q8_ADD_ROWS(64, 1)
DEFINE_Q8_ADD_ROWS(32, 1)
DEFINE_Q8_ADD_ROWS(128, 2)
DEFINE_Q8_ADD_ROWS(64, 2)
DEFINE_Q8_ADD_ROWS(32, 2)
DEFINE_Q8_ADD_ROWS(128, 4)
DEFINE_Q8_ADD_ROWS(64, 4)
DEFINE_Q8_ADD_ROWS(32, 4)

#undef DEFINE_Q8_ADD_ROWS

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

static void fill_q8(std::vector<hrx_block_q8_0> & blocks, int rows, int blocks_per_row) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            hrx_block_q8_0 & block = blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
            block.d = half_small_bits;
            for (int i = 0; i < 32; ++i) {
                block.qs[i] = static_cast<int8_t>(((row * 11 + block_idx * 13 + i * 7) & 31) - 16);
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

static void fill_bias(std::vector<float> & bias, int rows, int cols) {
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            bias[static_cast<size_t>(col) * rows + row] = make_value(row + col * rows, 47);
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

static void launch_q8_add(
        const std::string & kernel,
        const hrx_block_q8_0 * src0,
        const float * src1,
        const float * bias,
        float * dst,
        int k,
        int rows,
        int cols) {
    if (kernel == "q8_add_wg256") {
        hrx_mul_mat_vec_q8_0_add_f32<<<dim3(rows, cols, 1), dim3(256, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else if (kernel == "q8_add_wg128") {
        probe_q8_0_add_rows1_wg128_f32<<<dim3(rows, cols, 1), dim3(128, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else if (kernel == "q8_add_wg64") {
        probe_q8_0_add_rows1_wg64_f32<<<dim3(rows, cols, 1), dim3(64, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else if (kernel == "q8_add_wg32") {
        probe_q8_0_add_rows1_wg32_f32<<<dim3(rows, cols, 1), dim3(32, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else if (kernel == "q8_add_rows2_wg128") {
        probe_q8_0_add_rows2_wg128_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(128, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else if (kernel == "q8_add_rows2_wg64") {
        probe_q8_0_add_rows2_wg64_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(64, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else if (kernel == "q8_add_rows2_wg32") {
        probe_q8_0_add_rows2_wg32_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(32, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else if (kernel == "q8_add_rows4_wg128") {
        probe_q8_0_add_rows4_wg128_f32<<<dim3((rows + 3) / 4, cols, 1), dim3(128, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else if (kernel == "q8_add_rows4_wg64") {
        probe_q8_0_add_rows4_wg64_f32<<<dim3((rows + 3) / 4, cols, 1), dim3(64, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else if (kernel == "q8_add_rows4_wg32") {
        probe_q8_0_add_rows4_wg32_f32<<<dim3((rows + 3) / 4, cols, 1), dim3(32, 1, 1)>>>(src0, src1, bias, dst, k, rows, cols);
    } else {
        std::fprintf(stderr, "unknown Q8_0 ADD kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    HIP_CHECK(hipSetDevice(0));

    const int blocks_per_row = opts.k / 32;
    const size_t block_count = static_cast<size_t>(opts.rows) * blocks_per_row;
    const size_t src1_count = static_cast<size_t>(opts.cols) * opts.k;
    const size_t dst_count = static_cast<size_t>(opts.cols) * opts.rows;

    std::vector<hrx_block_q8_0> h_src0(block_count);
    std::vector<float> h_src1(src1_count);
    std::vector<float> h_bias(dst_count);
    std::vector<float> h_ref(dst_count, 0.0f);
    std::vector<float> h_dst(dst_count, 0.0f);
    fill_q8(h_src0, opts.rows, blocks_per_row);
    fill_rhs(h_src1, opts.k, opts.cols);
    fill_bias(h_bias, opts.rows, opts.cols);

    device_buffer<hrx_block_q8_0> d_src0(block_count);
    device_buffer<float> d_src1(src1_count);
    device_buffer<float> d_bias(dst_count);
    device_buffer<float> d_ref(dst_count);
    device_buffer<float> d_dst(dst_count);
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), block_count * sizeof(hrx_block_q8_0), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), src1_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias.ptr, h_bias.data(), dst_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_ref.ptr, 0, dst_count * sizeof(float)));
    HIP_CHECK(hipMemset(d_dst.ptr, 0, dst_count * sizeof(float)));

    if (opts.check) {
        launch_q8_add("q8_add_wg256", d_src0.ptr, d_src1.ptr, d_bias.ptr, d_ref.ptr, opts.k, opts.rows, opts.cols);
        launch_q8_add(opts.kernel, d_src0.ptr, d_src1.ptr, d_bias.ptr, d_dst.ptr, opts.k, opts.rows, opts.cols);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_ref.data(), d_ref.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
        check_close(h_dst, h_ref, opts.kernel.c_str());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_q8_add(opts.kernel, d_src0.ptr, d_src1.ptr, d_bias.ptr, d_dst.ptr, opts.k, opts.rows, opts.cols);
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
            launch_q8_add(opts.kernel, d_src0.ptr, d_src1.ptr, d_bias.ptr, d_dst.ptr, opts.k, opts.rows, opts.cols);
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
    std::printf("kernel=%s shape=q8_0_add k=%d rows=%d cols=%d mean_us=%.3f stdev_us=%.3f samples=",
        opts.kernel.c_str(), opts.k, opts.rows, opts.cols, mean, std::sqrt(var));
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.3f", i == 0 ? "[" : ",", samples[i]);
    }
    std::printf("]\n");
    return 0;
}
