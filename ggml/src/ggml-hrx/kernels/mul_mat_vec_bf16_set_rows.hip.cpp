#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_mul_mat_vec_bf16_set_rows_constants {
    long long k;
    long long rows;
    long long set_rows_ne1;
    long long idx_nb0;
    long long dst_nb1;
};

static __device__ __forceinline__ float hrx_bf16_to_f32(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } bits = { static_cast<uint32_t>(value) << 16 };
    return bits.f;
}

static __device__ __forceinline__ long long hrx_load_i64(const long long * base, long long byte_offset) {
    return *reinterpret_cast<const long long *>(reinterpret_cast<const char *>(base) + byte_offset);
}

static __device__ __forceinline__ float hrx_reduce_256(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < (256 / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_set_rows_f16(
        const uint16_t * src0, const float * src1, const long long * idxs, __half * dst,
        hrx_mul_mat_vec_bf16_set_rows_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    __shared__ float sumsh[256];

    const uint16_t * src0_row = src0 + row * c.k;
    float sum = 0.0f;
    for (long long i = tid; i < c.k; i += 256) {
        sum += hrx_bf16_to_f32(src0_row[i]) * src1[i];
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        const long long dst_row = hrx_load_i64(idxs, row * c.idx_nb0);
        if (dst_row >= 0 && dst_row < c.set_rows_ne1) {
            *reinterpret_cast<__half *>(
                reinterpret_cast<char *>(dst) + dst_row * c.dst_nb1) = __float2half(sum);
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_rows4_k2048_cols1_set_rows_f16(
        const uint16_t * src0, const float * src1, const long long * idxs, __half * dst,
        hrx_mul_mat_vec_bf16_set_rows_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    (void) c.k;

    const long long dst_row = hrx_load_i64(idxs, 0);
    if (dst_row < 0 || dst_row >= c.set_rows_ne1) {
        return;
    }

    __shared__ float rhs[2048];
    __shared__ float partial[8];
    for (unsigned int i = tid; i < 2048; i += 256) {
        rhs[i] = src1[i];
    }
    __syncthreads();

    const unsigned int row_lane = tid >> 6;
    const unsigned int lane = tid & 63;
    const long long row = row0 + static_cast<long long>(row_lane);
    float sum = 0.0f;
    if (row < c.rows) {
        const uint16_t * src0_row = src0 + row * 2048;
#pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const unsigned int i = lane * 2 + static_cast<unsigned int>(iter) * 128;
            const uint32_t a = *reinterpret_cast<const uint32_t *>(src0_row + i);
            const float2 b = *reinterpret_cast<const float2 *>(rhs + i);
            sum += hrx_bf16_to_f32(static_cast<uint16_t>(a & 0xffffu)) * b.x;
            sum += hrx_bf16_to_f32(static_cast<uint16_t>(a >> 16)) * b.y;
        }
    }

    const unsigned int sublane = lane & 31;
    const unsigned int subwave = lane >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (sublane == 0) {
        partial[row_lane * 2 + subwave] = sum;
    }
    __syncthreads();

    if (lane == 0 && row < c.rows) {
        *reinterpret_cast<__half *>(
            reinterpret_cast<char *>(dst) + dst_row * c.dst_nb1 + row * static_cast<long long>(sizeof(__half))) =
                __float2half(partial[row_lane * 2] + partial[row_lane * 2 + 1]);
    }
}
