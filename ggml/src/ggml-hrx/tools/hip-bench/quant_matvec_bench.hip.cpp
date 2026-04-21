#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../kernels/mul_mat_vec_q5_k.hip.cpp"
#define hrx_reduce_wg hrx_q6_reduce_wg
#define hrx_reduce_wg_array hrx_q6_reduce_wg_array
#include "../../kernels/mul_mat_vec_q6_k.hip.cpp"
#undef hrx_reduce_wg_array
#undef hrx_reduce_wg

extern "C" __global__ void probe_mul_mat_vec_q5_k_rows2_cols1_wg32_f32(
        const hrx_block_q5_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q5_k_rows2_cols_f32_impl<1, 32>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void probe_mul_mat_vec_q5_k_wg32_f32(
        const hrx_block_q5_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q5_k_f32_impl<32>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void probe_mul_mat_vec_q5_k_rows4_cols1_wg32_f32(
        const hrx_block_q5_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[4];

    const long long blocks_per_row = k / 256;
    const float * src1_col = src1 + col * k;
    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    float sum[4] = {};

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const float * src_block = src1_col + block_idx * 256;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (row0 + r < rows) {
                const hrx_block_q5_K * row_blocks = src0 + (row0 + r) * blocks_per_row;
                sum[r] += hrx_q5_k_dot16(row_blocks + block_idx, src_block, itid);
            }
        }
    }

    hrx_reduce_wg_array<32, 4>(sum, sumsh);

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (row0 + r < rows) {
                dst[col * rows + row0 + r] = sum[r];
            }
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_q5_k_rows2_dot16_wg32_f32(
        const hrx_block_q5_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[2];

    const long long blocks_per_row = k / 256;
    const hrx_block_q5_K * row0_blocks = src0 + row0 * blocks_per_row;
    const hrx_block_q5_K * row1_blocks = row0_blocks + blocks_per_row;
    const float * src1_col = src1 + col * k;
    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    float sum[2] = {};

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const float * src_block = src1_col + block_idx * 256;
        sum[0] += hrx_q5_k_dot16(row0_blocks + block_idx, src_block, itid);
        if (row0 + 1 < rows) {
            sum[1] += hrx_q5_k_dot16(row1_blocks + block_idx, src_block, itid);
        }
    }

    hrx_reduce_wg_array<32, 2>(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row0] = sum[0];
        if (row0 + 1 < rows) {
            dst[col * rows + row0 + 1] = sum[1];
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_q5_k_rows2_cols1_wg64_f32(
        const hrx_block_q5_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q5_k_rows2_cols_f32_impl<1, 64>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void probe_mul_mat_vec_q5_k_rows2_cols1_wg128_f32(
        const hrx_block_q5_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q5_k_rows2_cols_f32_impl<1, 128>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void probe_mul_mat_vec_q6_k_rows2_cols1_generic_wg32_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q6_k_rows2_cols_wg32_f32_impl<1>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void probe_mul_mat_vec_q6_k_rows2_cols1_even_wg32_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= rows || col >= cols) {
        return;
    }

    const long long blocks_per_row = k / 256;
    const hrx_block_q6_K * row0_blocks = src0 + row0 * blocks_per_row;
    const hrx_block_q6_K * row1_blocks = row0_blocks + blocks_per_row;
    const float * src1_col = src1 + col * k;

    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const float * src_block = src1_col + block_idx * 256;
        sum0 += hrx_q6_k_dot16(row0_blocks + block_idx, src_block, itid);
        sum1 += hrx_q6_k_dot16(row1_blocks + block_idx, src_block, itid);
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }

    if (tid == 0) {
        dst[col * rows + row0] = sum0;
        dst[col * rows + row0 + 1] = sum1;
    }
}

template <int BLOCKS_PER_ROW>
static __device__ __forceinline__ void probe_q6_rows2_cols1_kblocks_impl(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= rows || col >= cols) {
        return;
    }

    const hrx_block_q6_K * row0_blocks = src0 + row0 * BLOCKS_PER_ROW;
    const hrx_block_q6_K * row1_blocks = row0_blocks + BLOCKS_PER_ROW;
    const float * src1_col = src1 + col * (BLOCKS_PER_ROW * 256);

    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

#pragma unroll
    for (int block_idx = block_slot; block_idx < BLOCKS_PER_ROW; block_idx += 2) {
        const float * src_block = src1_col + block_idx * 256;
        sum0 += hrx_q6_k_dot16(row0_blocks + block_idx, src_block, itid);
        sum1 += hrx_q6_k_dot16(row1_blocks + block_idx, src_block, itid);
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }

    if (tid == 0) {
        dst[col * rows + row0] = sum0;
        dst[col * rows + row0 + 1] = sum1;
    }
}

extern "C" __global__ void probe_mul_mat_vec_q6_k_rows2_cols1_k2048_wg32_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    (void) k;
    probe_q6_rows2_cols1_kblocks_impl<8>(src0, src1, dst, rows, cols);
}

extern "C" __global__ void probe_mul_mat_vec_q6_k_rows2_cols1_k4096_wg32_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    (void) k;
    probe_q6_rows2_cols1_kblocks_impl<16>(src0, src1, dst, rows, cols);
}

static __device__ __forceinline__ float probe_q6_k_dot16_scale_shuffle(
        const hrx_block_q6_K * block,
        const float * src,
        int itid,
        int lane_base,
        int loaded_scale) {
    const int v_im = itid >> 3;
    const int v_in = itid & 7;
    const int l0 = 4 * v_in;
    const int is = v_in >> 2;

    const int ql_offset = 64 * v_im + l0;
    const int qh_offset = 32 * v_im + l0;
    const int s_offset = 8 * v_im + is;
    const int y_offset = 128 * v_im + l0;

    const uint32_t ql0 = hrx_load_u32(block->ql + ql_offset);
    const uint32_t ql32 = hrx_load_u32(block->ql + ql_offset + 32);
    const uint32_t qh = hrx_load_u32(block->qh + qh_offset);

    const uint32_t q0 = (ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4);
    const uint32_t q1 = (ql32 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2);
    const uint32_t q2 = ((ql0 >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
    const uint32_t q3 = ((ql32 >> 4) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2);

    const float4 by0 = *reinterpret_cast<const float4 *>(src + y_offset);
    const float4 by32 = *reinterpret_cast<const float4 *>(src + y_offset + 32);
    const float4 by64 = *reinterpret_cast<const float4 *>(src + y_offset + 64);
    const float4 by96 = *reinterpret_cast<const float4 *>(src + y_offset + 96);

    const float sx =
        (static_cast<float>((q0 >> 0) & 0xFFu) - 32.0f) * by0.x +
        (static_cast<float>((q0 >> 8) & 0xFFu) - 32.0f) * by0.y +
        (static_cast<float>((q0 >> 16) & 0xFFu) - 32.0f) * by0.z +
        (static_cast<float>((q0 >> 24) & 0xFFu) - 32.0f) * by0.w;
    const float sy =
        (static_cast<float>((q1 >> 0) & 0xFFu) - 32.0f) * by32.x +
        (static_cast<float>((q1 >> 8) & 0xFFu) - 32.0f) * by32.y +
        (static_cast<float>((q1 >> 16) & 0xFFu) - 32.0f) * by32.z +
        (static_cast<float>((q1 >> 24) & 0xFFu) - 32.0f) * by32.w;
    const float sz =
        (static_cast<float>((q2 >> 0) & 0xFFu) - 32.0f) * by64.x +
        (static_cast<float>((q2 >> 8) & 0xFFu) - 32.0f) * by64.y +
        (static_cast<float>((q2 >> 16) & 0xFFu) - 32.0f) * by64.z +
        (static_cast<float>((q2 >> 24) & 0xFFu) - 32.0f) * by64.w;
    const float sw =
        (static_cast<float>((q3 >> 0) & 0xFFu) - 32.0f) * by96.x +
        (static_cast<float>((q3 >> 8) & 0xFFu) - 32.0f) * by96.y +
        (static_cast<float>((q3 >> 16) & 0xFFu) - 32.0f) * by96.z +
        (static_cast<float>((q3 >> 24) & 0xFFu) - 32.0f) * by96.w;

    const float d = __half2float(__ushort_as_half(block->d));
    const float sc0 = static_cast<float>(__shfl(loaded_scale, lane_base + s_offset));
    const float sc1 = static_cast<float>(__shfl(loaded_scale, lane_base + s_offset + 2));
    const float sc2 = static_cast<float>(__shfl(loaded_scale, lane_base + s_offset + 4));
    const float sc3 = static_cast<float>(__shfl(loaded_scale, lane_base + s_offset + 6));
    return d * (sx * sc0 + sy * sc1 + sz * sc2 + sw * sc3);
}

extern "C" __global__ void probe_mul_mat_vec_q6_k_rows2_cols1_scale_shuffle_wg32_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= rows || col >= cols) {
        return;
    }

    const long long blocks_per_row = k / 256;
    const hrx_block_q6_K * row0_blocks = src0 + row0 * blocks_per_row;
    const hrx_block_q6_K * row1_blocks = row0_blocks + blocks_per_row;
    const float * src1_col = src1 + col * k;

    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    const int lane_base = block_slot * 16;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const hrx_block_q6_K * block0 = row0_blocks + block_idx;
        const hrx_block_q6_K * block1 = row1_blocks + block_idx;
        const float * src_block = src1_col + block_idx * 256;
        const int scale0 = static_cast<int>(block0->scales[itid]);
        const int scale1 = static_cast<int>(block1->scales[itid]);
        sum0 += probe_q6_k_dot16_scale_shuffle(block0, src_block, itid, lane_base, scale0);
        sum1 += probe_q6_k_dot16_scale_shuffle(block1, src_block, itid, lane_base, scale1);
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }

    if (tid == 0) {
        dst[col * rows + row0] = sum0;
        dst[col * rows + row0 + 1] = sum1;
    }
}

extern "C" __global__ void probe_mul_mat_vec_q6_k_rows4_cols1_wg32_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[4];

    const long long blocks_per_row = k / 256;
    const float * src1_col = src1 + col * k;
    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    float sum[4] = {};

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const float * src_block = src1_col + block_idx * 256;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (row0 + r < rows) {
                const hrx_block_q6_K * row_blocks = src0 + (row0 + r) * blocks_per_row;
                sum[r] += hrx_q6_k_dot16(row_blocks + block_idx, src_block, itid);
            }
        }
    }

    hrx_q6_reduce_wg_array<32, 4>(sum, sumsh);

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (row0 + r < rows) {
                dst[col * rows + row0 + r] = sum[r];
            }
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_q6_k_rows2_cols1_wg64_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q6_k_rows2_cols_f32_impl<1, 64>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void probe_mul_mat_vec_q6_k_rows2_cols1_wg128_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q6_k_rows2_cols_f32_impl<1, 128>(src0, src1, dst, k, rows, cols);
}

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

static constexpr unsigned short half_one_bits = 0x3c00;
static constexpr unsigned short half_small_bits = 0x3800;

struct options {
    std::string kernel = "q6_rows2_cols1_wg32";
    int iters = 1000;
    int warmup = 50;
    int repeats = 5;
    int k = 2048;
    int rows = 2048;
    int cols = 1;
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
        } else if (std::strcmp(argv[i], "--check") == 0) {
            opts.check = true;
        } else {
            std::fprintf(stderr,
                "usage: %s [--kernel=<name>] [--iters=N] [--warmup=N] [--repeats=N] [--k=N] [--rows=N] [--cols=N] [--check]\n"
                "  Q5: q5_wg256, q5_wg128, q5_wg64, q5_wg32, q5_rows2_cols1_wg32, q5_rows2_dot16_wg32, q5_rows2_cols1_wg64, q5_rows2_cols1_wg128, q5_rows2_cols{2..8}_wg128, q5_rows2_cols{2..8}_wg64\n"
                "  Q6: q6_wg256, q6_wg128, q6_wg64, q6_rows2_cols1_wg32, q6_rows2_cols1_generic_wg32, q6_rows2_cols1_even_wg32, q6_rows2_cols1_scale_shuffle_wg32, q6_rows2_cols1_wg64, q6_rows2_cols1_wg128, q6_rows2_cols{2..8}_wg32, q6_rows2_cols{2..8}_wg64, q6_rows2_cols{2..8}_wg128\n",
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

static void fill_q5(std::vector<hrx_block_q5_K> & blocks, int rows, int blocks_per_row) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            hrx_block_q5_K & block = blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
            block.d = half_small_bits;
            block.dmin = 0;
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

static void fill_q6(std::vector<hrx_block_q6_K> & blocks, int rows, int blocks_per_row) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            hrx_block_q6_K & block = blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
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

static int parse_rows2_cols(const std::string & kernel) {
    const char * needle = "_rows2_cols";
    const size_t pos = kernel.find(needle);
    if (pos == std::string::npos) {
        return 0;
    }
    return std::atoi(kernel.c_str() + pos + std::strlen(needle));
}

static void launch_q5(
        const std::string & kernel,
        const hrx_block_q5_K * src0,
        const float * src1,
        float * dst,
        int k,
        int rows,
        int cols) {
    if (kernel == "q5_wg256") {
        hrx_mul_mat_vec_q5_k_f32<<<dim3(rows, cols, 1), dim3(256, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else if (kernel == "q5_wg128") {
        hrx_mul_mat_vec_q5_k_wg128_f32<<<dim3(rows, cols, 1), dim3(128, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else if (kernel == "q5_wg64") {
        hrx_mul_mat_vec_q5_k_wg64_f32<<<dim3(rows, cols, 1), dim3(64, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else if (kernel == "q5_wg32") {
        probe_mul_mat_vec_q5_k_wg32_f32<<<dim3(rows, cols, 1), dim3(32, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else if (kernel == "q5_rows2_cols1_wg32") {
        probe_mul_mat_vec_q5_k_rows2_cols1_wg32_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q5_rows2_dot16_wg32") {
        probe_mul_mat_vec_q5_k_rows2_dot16_wg32_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q5_rows4_cols1_wg32") {
        probe_mul_mat_vec_q5_k_rows4_cols1_wg32_f32<<<dim3((rows + 3) / 4, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q5_rows2_cols1_wg64") {
        probe_mul_mat_vec_q5_k_rows2_cols1_wg64_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(64, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q5_rows2_cols1_wg128") {
        probe_mul_mat_vec_q5_k_rows2_cols1_wg128_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(128, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else {
        const int group_cols = parse_rows2_cols(kernel);
        if (group_cols < 2 || group_cols > 8) {
            std::fprintf(stderr, "unknown Q5 kernel: %s\n", kernel.c_str());
            std::exit(2);
        }
#define HRX_LAUNCH_Q5_ROWS2(COLS, WG) \
        do { \
            hrx_mul_mat_vec_q5_k_rows2_cols##COLS##_wg##WG##_f32<<< \
                dim3((rows + 1) / 2, (cols + (COLS) - 1) / (COLS), 1), dim3(WG, 1, 1)>>>(src0, src1, dst, k, rows, cols); \
        } while (0)
        if (kernel.find("_wg64") != std::string::npos) {
            switch (group_cols) {
                case 2: HRX_LAUNCH_Q5_ROWS2(2, 64); break;
                case 3: HRX_LAUNCH_Q5_ROWS2(3, 64); break;
                case 4: HRX_LAUNCH_Q5_ROWS2(4, 64); break;
                case 5: HRX_LAUNCH_Q5_ROWS2(5, 64); break;
                case 6: HRX_LAUNCH_Q5_ROWS2(6, 64); break;
                case 7: HRX_LAUNCH_Q5_ROWS2(7, 64); break;
                case 8: HRX_LAUNCH_Q5_ROWS2(8, 64); break;
            }
        } else if (kernel.find("_wg128") != std::string::npos) {
            switch (group_cols) {
                case 2: HRX_LAUNCH_Q5_ROWS2(2, 128); break;
                case 3: HRX_LAUNCH_Q5_ROWS2(3, 128); break;
                case 4: HRX_LAUNCH_Q5_ROWS2(4, 128); break;
                case 5: HRX_LAUNCH_Q5_ROWS2(5, 128); break;
                case 6: HRX_LAUNCH_Q5_ROWS2(6, 128); break;
                case 7: HRX_LAUNCH_Q5_ROWS2(7, 128); break;
                case 8: HRX_LAUNCH_Q5_ROWS2(8, 128); break;
            }
        } else {
            std::fprintf(stderr, "unknown Q5 rows2 kernel: %s\n", kernel.c_str());
            std::exit(2);
        }
#undef HRX_LAUNCH_Q5_ROWS2
    }
    HIP_CHECK(hipGetLastError());
}

static void launch_q6(
        const std::string & kernel,
        const hrx_block_q6_K * src0,
        const float * src1,
        float * dst,
        int k,
        int rows,
        int cols) {
    if (kernel == "q6_wg256") {
        hrx_mul_mat_vec_q6_k_f32<<<dim3(rows, cols, 1), dim3(256, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_wg128") {
        hrx_mul_mat_vec_q6_k_wg128_f32<<<dim3(rows, cols, 1), dim3(128, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_wg64") {
        hrx_mul_mat_vec_q6_k_wg64_f32<<<dim3(rows, cols, 1), dim3(64, 1, 1)>>>(src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_rows2_cols1_wg32") {
        hrx_mul_mat_vec_q6_k_rows2_cols1_wg32_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_rows2_cols1_generic_wg32") {
        probe_mul_mat_vec_q6_k_rows2_cols1_generic_wg32_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_rows2_cols1_even_wg32") {
        probe_mul_mat_vec_q6_k_rows2_cols1_even_wg32_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_rows2_cols1_k2048_wg32") {
        probe_mul_mat_vec_q6_k_rows2_cols1_k2048_wg32_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_rows2_cols1_k4096_wg32") {
        probe_mul_mat_vec_q6_k_rows2_cols1_k4096_wg32_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_rows2_cols1_scale_shuffle_wg32") {
        probe_mul_mat_vec_q6_k_rows2_cols1_scale_shuffle_wg32_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_rows4_cols1_wg32") {
        probe_mul_mat_vec_q6_k_rows4_cols1_wg32_f32<<<dim3((rows + 3) / 4, cols, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_rows2_cols1_wg64") {
        probe_mul_mat_vec_q6_k_rows2_cols1_wg64_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(64, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "q6_rows2_cols1_wg128") {
        probe_mul_mat_vec_q6_k_rows2_cols1_wg128_f32<<<dim3((rows + 1) / 2, cols, 1), dim3(128, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else {
        const int group_cols = parse_rows2_cols(kernel);
        if (group_cols < 2 || group_cols > 8) {
            std::fprintf(stderr, "unknown Q6 kernel: %s\n", kernel.c_str());
            std::exit(2);
        }
#define HRX_LAUNCH_Q6_ROWS2(COLS, WG) \
        do { \
            hrx_mul_mat_vec_q6_k_rows2_cols##COLS##_wg##WG##_f32<<< \
                dim3((rows + 1) / 2, (cols + (COLS) - 1) / (COLS), 1), dim3(WG, 1, 1)>>>(src0, src1, dst, k, rows, cols); \
        } while (0)
        if (kernel.find("_wg32") != std::string::npos) {
            switch (group_cols) {
                case 2: HRX_LAUNCH_Q6_ROWS2(2, 32); break;
                case 3: HRX_LAUNCH_Q6_ROWS2(3, 32); break;
                case 4: HRX_LAUNCH_Q6_ROWS2(4, 32); break;
                case 5: HRX_LAUNCH_Q6_ROWS2(5, 32); break;
                case 6: HRX_LAUNCH_Q6_ROWS2(6, 32); break;
                case 7: HRX_LAUNCH_Q6_ROWS2(7, 32); break;
                case 8: HRX_LAUNCH_Q6_ROWS2(8, 32); break;
            }
        } else if (kernel.find("_wg64") != std::string::npos) {
            switch (group_cols) {
                case 2: HRX_LAUNCH_Q6_ROWS2(2, 64); break;
                case 3: HRX_LAUNCH_Q6_ROWS2(3, 64); break;
                case 4: HRX_LAUNCH_Q6_ROWS2(4, 64); break;
                case 5: HRX_LAUNCH_Q6_ROWS2(5, 64); break;
                case 6: HRX_LAUNCH_Q6_ROWS2(6, 64); break;
                case 7: HRX_LAUNCH_Q6_ROWS2(7, 64); break;
                case 8: HRX_LAUNCH_Q6_ROWS2(8, 64); break;
            }
        } else if (kernel.find("_wg128") != std::string::npos) {
            switch (group_cols) {
                case 2: HRX_LAUNCH_Q6_ROWS2(2, 128); break;
                case 3: HRX_LAUNCH_Q6_ROWS2(3, 128); break;
                case 4: HRX_LAUNCH_Q6_ROWS2(4, 128); break;
                case 5: HRX_LAUNCH_Q6_ROWS2(5, 128); break;
                case 6: HRX_LAUNCH_Q6_ROWS2(6, 128); break;
                case 7: HRX_LAUNCH_Q6_ROWS2(7, 128); break;
                case 8: HRX_LAUNCH_Q6_ROWS2(8, 128); break;
            }
        } else {
            std::fprintf(stderr, "unknown Q6 rows2 kernel: %s\n", kernel.c_str());
            std::exit(2);
        }
#undef HRX_LAUNCH_Q6_ROWS2
    }
    HIP_CHECK(hipGetLastError());
}

template <typename Block, typename LaunchFn>
static void run_quant(
        const options & opts,
        const char * family,
        const std::string & reference_kernel,
        void (*fill_blocks)(std::vector<Block> &, int, int),
        LaunchFn launch) {
    const int blocks_per_row = opts.k / 256;
    const size_t block_count = static_cast<size_t>(opts.rows) * blocks_per_row;
    const size_t src1_count = static_cast<size_t>(opts.cols) * opts.k;
    const size_t dst_count = static_cast<size_t>(opts.cols) * opts.rows;

    std::vector<Block> h_src0(block_count);
    std::vector<float> h_src1(src1_count);
    std::vector<float> h_ref(dst_count, 0.0f);
    std::vector<float> h_dst(dst_count, 0.0f);
    fill_blocks(h_src0, opts.rows, blocks_per_row);
    fill_rhs(h_src1, opts.k, opts.cols);

    device_buffer<Block> d_src0(block_count);
    device_buffer<float> d_src1(src1_count);
    device_buffer<float> d_ref(dst_count);
    device_buffer<float> d_dst(dst_count);
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), block_count * sizeof(Block), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), src1_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_ref.ptr, 0, dst_count * sizeof(float)));
    HIP_CHECK(hipMemset(d_dst.ptr, 0, dst_count * sizeof(float)));

    if (opts.check) {
        launch(reference_kernel, d_src0.ptr, d_src1.ptr, d_ref.ptr, opts.k, opts.rows, opts.cols);
        launch(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, opts.k, opts.rows, opts.cols);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_ref.data(), d_ref.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
        check_close(h_dst, h_ref, opts.kernel.c_str());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, opts.k, opts.rows, opts.cols);
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
            launch(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, opts.k, opts.rows, opts.cols);
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
    std::printf("kernel=%s shape=%s k=%d rows=%d cols=%d mean_us=%.3f stdev_us=%.3f samples=",
        opts.kernel.c_str(), family, opts.k, opts.rows, opts.cols, mean, std::sqrt(var));
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.3f", i == 0 ? "[" : ",", samples[i]);
    }
    std::printf("]\n");
}

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    HIP_CHECK(hipSetDevice(0));
    if (opts.kernel.rfind("q5_", 0) == 0) {
        run_quant<hrx_block_q5_K>(
            opts, "q5_k", "q5_wg256", fill_q5,
            [](const std::string & kernel, const hrx_block_q5_K * src0, const float * src1,
               float * dst, int k, int rows, int cols) {
                launch_q5(kernel, src0, src1, dst, k, rows, cols);
            });
    } else if (opts.kernel.rfind("q6_", 0) == 0) {
        run_quant<hrx_block_q6_K>(
            opts, "q6_k", "q6_wg256", fill_q6,
            [](const std::string & kernel, const hrx_block_q6_K * src0, const float * src1,
               float * dst, int k, int rows, int cols) {
                launch_q6(kernel, src0, src1, dst, k, rows, cols);
            });
    } else {
        std::fprintf(stderr, "unknown kernel family: %s\n", opts.kernel.c_str());
        return 2;
    }
    return 0;
}
