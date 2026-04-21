#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../kernels/mul_mat_vec_bf16.hip.cpp"
#include "../../kernels/mul_mat_vec_f16_batched.hip.cpp"
#define hrx_reduce_256 hrx_f32_batched_reduce_256
#define hrx_reduce8_256 hrx_f32_batched_reduce8_256
#define hrx_reduce16_256 hrx_f32_batched_reduce16_256
#include "../../kernels/mul_mat_vec_f32_batched.hip.cpp"
#undef hrx_reduce16_256
#undef hrx_reduce8_256
#undef hrx_reduce_256

extern "C" __global__ void probe_mul_mat_vec_f16_batched_rows2_cols1_wg32_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= c.rows) {
        return;
    }

    const long long i12 = outer % c.dst_ne2;
    const long long i13 = outer / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row0 = reinterpret_cast<const char *>(src0) +
        row0 * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src0_row1 = reinterpret_cast<const char *>(src0) +
        row1 * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col = reinterpret_cast<const char *>(src1) + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    const bool have_row1 = row1 < c.rows;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (long long i = static_cast<long long>(tid) * 2; i < c.k; i += 64) {
        const long long byte_i = i * static_cast<long long>(sizeof(float));
        const float2 b = *reinterpret_cast<const float2 *>(src1_col + byte_i);
        const __half2 a0 = *reinterpret_cast<const __half2 *>(src0_row0 + i * static_cast<long long>(sizeof(__half)));
        const float2 a0f = __half22float2(a0);
        sum0 += a0f.x * b.x + a0f.y * b.y;
        if (have_row1) {
            const __half2 a1 = *reinterpret_cast<const __half2 *>(src0_row1 + i * static_cast<long long>(sizeof(__half)));
            const float2 a1f = __half22float2(a1);
            sum1 += a1f.x * b.x + a1f.y * b.y;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
    }

    if (tid == 0) {
        char * dst_row0 = reinterpret_cast<char *>(dst) +
            row0 * sizeof(float) + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        *reinterpret_cast<float *>(dst_row0) = sum0;
        if (have_row1) {
            char * dst_row1 = reinterpret_cast<char *>(dst) +
                row1 * sizeof(float) + i12 * c.dst_nb2 + i13 * c.dst_nb3;
            *reinterpret_cast<float *>(dst_row1) = sum1;
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_f16_batched_rows4_cols1_wg32_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= c.rows) {
        return;
    }

    const long long i12 = outer % c.dst_ne2;
    const long long i13 = outer / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_base = reinterpret_cast<const char *>(src0) +
        src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col = reinterpret_cast<const char *>(src1) + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (long long i = static_cast<long long>(tid) * 2; i < c.k; i += 64) {
        const long long byte_i_f32 = i * static_cast<long long>(sizeof(float));
        const long long byte_i_f16 = i * static_cast<long long>(sizeof(__half));
        const float2 b = *reinterpret_cast<const float2 *>(src1_col + byte_i_f32);
        const __half2 a0 = *reinterpret_cast<const __half2 *>(src0_base + (row0 + 0) * c.src0_nb1 + byte_i_f16);
        const float2 a0f = __half22float2(a0);
        sum0 += a0f.x * b.x + a0f.y * b.y;
        if (row0 + 1 < c.rows) {
            const __half2 a1 = *reinterpret_cast<const __half2 *>(src0_base + (row0 + 1) * c.src0_nb1 + byte_i_f16);
            const float2 a1f = __half22float2(a1);
            sum1 += a1f.x * b.x + a1f.y * b.y;
        }
        if (row0 + 2 < c.rows) {
            const __half2 a2 = *reinterpret_cast<const __half2 *>(src0_base + (row0 + 2) * c.src0_nb1 + byte_i_f16);
            const float2 a2f = __half22float2(a2);
            sum2 += a2f.x * b.x + a2f.y * b.y;
        }
        if (row0 + 3 < c.rows) {
            const __half2 a3 = *reinterpret_cast<const __half2 *>(src0_base + (row0 + 3) * c.src0_nb1 + byte_i_f16);
            const float2 a3f = __half22float2(a3);
            sum3 += a3f.x * b.x + a3f.y * b.y;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
        sum2 += __shfl_down(sum2, offset, 32);
        sum3 += __shfl_down(sum3, offset, 32);
    }

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        *reinterpret_cast<float *>(dst_base + (row0 + 0) * sizeof(float)) = sum0;
        if (row0 + 1 < c.rows) {
            *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = sum1;
        }
        if (row0 + 2 < c.rows) {
            *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = sum2;
        }
        if (row0 + 3 < c.rows) {
            *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = sum3;
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_f16_batched_rows2_cols1_x4_wg32_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= c.rows) {
        return;
    }

    const long long i12 = outer % c.dst_ne2;
    const long long i13 = outer / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row0 = reinterpret_cast<const char *>(src0) +
        row0 * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src0_row1 = reinterpret_cast<const char *>(src0) +
        row1 * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col = reinterpret_cast<const char *>(src1) + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    const bool have_row1 = row1 < c.rows;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (long long i = static_cast<long long>(tid) * 4; i < c.k; i += 128) {
        const long long byte_i_f32 = i * static_cast<long long>(sizeof(float));
        const long long byte_i_f16 = i * static_cast<long long>(sizeof(__half));
        const float4 b = *reinterpret_cast<const float4 *>(src1_col + byte_i_f32);
        const __half2 a00 = *reinterpret_cast<const __half2 *>(src0_row0 + byte_i_f16);
        const __half2 a01 = *reinterpret_cast<const __half2 *>(src0_row0 + byte_i_f16 + 2 * static_cast<long long>(sizeof(__half)));
        const float2 a00f = __half22float2(a00);
        const float2 a01f = __half22float2(a01);
        sum0 += a00f.x * b.x + a00f.y * b.y + a01f.x * b.z + a01f.y * b.w;
        if (have_row1) {
            const __half2 a10 = *reinterpret_cast<const __half2 *>(src0_row1 + byte_i_f16);
            const __half2 a11 = *reinterpret_cast<const __half2 *>(src0_row1 + byte_i_f16 + 2 * static_cast<long long>(sizeof(__half)));
            const float2 a10f = __half22float2(a10);
            const float2 a11f = __half22float2(a11);
            sum1 += a10f.x * b.x + a10f.y * b.y + a11f.x * b.z + a11f.y * b.w;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
    }

    if (tid == 0) {
        char * dst_row0 = reinterpret_cast<char *>(dst) +
            row0 * sizeof(float) + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        *reinterpret_cast<float *>(dst_row0) = sum0;
        if (have_row1) {
            char * dst_row1 = reinterpret_cast<char *>(dst) +
                row1 * sizeof(float) + i12 * c.dst_nb2 + i13 * c.dst_nb3;
            *reinterpret_cast<float *>(dst_row1) = sum1;
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_f16_batched_rows2_cols1_x8_wg32_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= c.rows) {
        return;
    }

    const long long i12 = outer % c.dst_ne2;
    const long long i13 = outer / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row0 = reinterpret_cast<const char *>(src0) +
        row0 * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src0_row1 = reinterpret_cast<const char *>(src0) +
        row1 * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col = reinterpret_cast<const char *>(src1) + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    const bool have_row1 = row1 < c.rows;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (long long i = static_cast<long long>(tid) * 8; i < c.k; i += 256) {
        const long long byte_i_f32 = i * static_cast<long long>(sizeof(float));
        const long long byte_i_f16 = i * static_cast<long long>(sizeof(__half));
        const float4 b0 = *reinterpret_cast<const float4 *>(src1_col + byte_i_f32);
        const float4 b1 = *reinterpret_cast<const float4 *>(src1_col + byte_i_f32 + 4 * static_cast<long long>(sizeof(float)));
        const __half2 a00 = *reinterpret_cast<const __half2 *>(src0_row0 + byte_i_f16);
        const __half2 a01 = *reinterpret_cast<const __half2 *>(src0_row0 + byte_i_f16 + 2 * static_cast<long long>(sizeof(__half)));
        const __half2 a02 = *reinterpret_cast<const __half2 *>(src0_row0 + byte_i_f16 + 4 * static_cast<long long>(sizeof(__half)));
        const __half2 a03 = *reinterpret_cast<const __half2 *>(src0_row0 + byte_i_f16 + 6 * static_cast<long long>(sizeof(__half)));
        const float2 a00f = __half22float2(a00);
        const float2 a01f = __half22float2(a01);
        const float2 a02f = __half22float2(a02);
        const float2 a03f = __half22float2(a03);
        sum0 += a00f.x * b0.x + a00f.y * b0.y + a01f.x * b0.z + a01f.y * b0.w;
        sum0 += a02f.x * b1.x + a02f.y * b1.y + a03f.x * b1.z + a03f.y * b1.w;
        if (have_row1) {
            const __half2 a10 = *reinterpret_cast<const __half2 *>(src0_row1 + byte_i_f16);
            const __half2 a11 = *reinterpret_cast<const __half2 *>(src0_row1 + byte_i_f16 + 2 * static_cast<long long>(sizeof(__half)));
            const __half2 a12 = *reinterpret_cast<const __half2 *>(src0_row1 + byte_i_f16 + 4 * static_cast<long long>(sizeof(__half)));
            const __half2 a13 = *reinterpret_cast<const __half2 *>(src0_row1 + byte_i_f16 + 6 * static_cast<long long>(sizeof(__half)));
            const float2 a10f = __half22float2(a10);
            const float2 a11f = __half22float2(a11);
            const float2 a12f = __half22float2(a12);
            const float2 a13f = __half22float2(a13);
            sum1 += a10f.x * b0.x + a10f.y * b0.y + a11f.x * b0.z + a11f.y * b0.w;
            sum1 += a12f.x * b1.x + a12f.y * b1.y + a13f.x * b1.z + a13f.y * b1.w;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
    }

    if (tid == 0) {
        char * dst_row0 = reinterpret_cast<char *>(dst) +
            row0 * sizeof(float) + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        *reinterpret_cast<float *>(dst_row0) = sum0;
        if (have_row1) {
            char * dst_row1 = reinterpret_cast<char *>(dst) +
                row1 * sizeof(float) + i12 * c.dst_nb2 + i13 * c.dst_nb3;
            *reinterpret_cast<float *>(dst_row1) = sum1;
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_bf16_rows4_cols1_wg32_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows) {
        return;
    }
    (void) cols;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (long long i = static_cast<long long>(tid) * 2; i < k; i += 64) {
        const float2 b = *reinterpret_cast<const float2 *>(src1 + i);
        const uint32_t a0 = *reinterpret_cast<const uint32_t *>(src0 + (row0 + 0) * k + i);
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0 & 0xffffu)) * b.x;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0 >> 16)) * b.y;
        if (row0 + 1 < rows) {
            const uint32_t a1 = *reinterpret_cast<const uint32_t *>(src0 + (row0 + 1) * k + i);
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1 & 0xffffu)) * b.x;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1 >> 16)) * b.y;
        }
        if (row0 + 2 < rows) {
            const uint32_t a2 = *reinterpret_cast<const uint32_t *>(src0 + (row0 + 2) * k + i);
            sum2 += hrx_bf16_to_f32(static_cast<uint16_t>(a2 & 0xffffu)) * b.x;
            sum2 += hrx_bf16_to_f32(static_cast<uint16_t>(a2 >> 16)) * b.y;
        }
        if (row0 + 3 < rows) {
            const uint32_t a3 = *reinterpret_cast<const uint32_t *>(src0 + (row0 + 3) * k + i);
            sum3 += hrx_bf16_to_f32(static_cast<uint16_t>(a3 & 0xffffu)) * b.x;
            sum3 += hrx_bf16_to_f32(static_cast<uint16_t>(a3 >> 16)) * b.y;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
        sum2 += __shfl_down(sum2, offset, 32);
        sum3 += __shfl_down(sum3, offset, 32);
    }

    if (tid == 0) {
        dst[row0 + 0] = sum0;
        if (row0 + 1 < rows) {
            dst[row0 + 1] = sum1;
        }
        if (row0 + 2 < rows) {
            dst[row0 + 2] = sum2;
        }
        if (row0 + 3 < rows) {
            dst[row0 + 3] = sum3;
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_bf16_rows2_cols1_x4_wg32_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows) {
        return;
    }
    (void) cols;

    const bool have_row1 = row1 < rows;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (long long i = static_cast<long long>(tid) * 4; i < k; i += 128) {
        const float4 b = *reinterpret_cast<const float4 *>(src1 + i);
        const uint2 a0 = *reinterpret_cast<const uint2 *>(src0 + row0 * k + i);
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.x & 0xffffu)) * b.x;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.x >> 16)) * b.y;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.y & 0xffffu)) * b.z;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.y >> 16)) * b.w;
        if (have_row1) {
            const uint2 a1 = *reinterpret_cast<const uint2 *>(src0 + row1 * k + i);
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.x & 0xffffu)) * b.x;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.x >> 16)) * b.y;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.y & 0xffffu)) * b.z;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.y >> 16)) * b.w;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
    }

    if (tid == 0) {
        dst[row0] = sum0;
        if (have_row1) {
            dst[row1] = sum1;
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_bf16_rows2_cols1_x8_wg32_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows) {
        return;
    }
    (void) cols;

    const bool have_row1 = row1 < rows;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (long long i = static_cast<long long>(tid) * 8; i < k; i += 256) {
        const float4 b0 = *reinterpret_cast<const float4 *>(src1 + i);
        const float4 b1 = *reinterpret_cast<const float4 *>(src1 + i + 4);
        const uint4 a0 = *reinterpret_cast<const uint4 *>(src0 + row0 * k + i);
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.x & 0xffffu)) * b0.x;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.x >> 16)) * b0.y;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.y & 0xffffu)) * b0.z;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.y >> 16)) * b0.w;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.z & 0xffffu)) * b1.x;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.z >> 16)) * b1.y;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.w & 0xffffu)) * b1.z;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0.w >> 16)) * b1.w;
        if (have_row1) {
            const uint4 a1 = *reinterpret_cast<const uint4 *>(src0 + row1 * k + i);
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.x & 0xffffu)) * b0.x;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.x >> 16)) * b0.y;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.y & 0xffffu)) * b0.z;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.y >> 16)) * b0.w;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.z & 0xffffu)) * b1.x;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.z >> 16)) * b1.y;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.w & 0xffffu)) * b1.z;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1.w >> 16)) * b1.w;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
    }

    if (tid == 0) {
        dst[row0] = sum0;
        if (have_row1) {
            dst[row1] = sum1;
        }
    }
}

extern "C" __global__ void probe_mul_mat_vec_f32_batched_rows4_k2048_cols1_lds_wg256_f32(
        const float * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f32_batched_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long i13 = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= c.rows || i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_base = reinterpret_cast<const char *>(src0) + src0_i03 * c.src0_nb3;
    const char * src1_col = reinterpret_cast<const char *>(src1) + i13 * c.src1_nb3;

    __shared__ float rhs[2048];
    __shared__ float partial[8];
    for (unsigned int i = tid; i < 2048; i += 256) {
        rhs[i] = *reinterpret_cast<const float *>(src1_col + i * static_cast<long long>(sizeof(float)));
    }
    __syncthreads();

    const unsigned int row_lane = tid >> 6;
    const unsigned int lane = tid & 63;
    const long long row = row0 + static_cast<long long>(row_lane);
    float sum = 0.0f;
    if (row < c.rows) {
        const char * src0_row = src0_base + row * c.src0_nb1;
#pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const unsigned int i = lane * 2 + static_cast<unsigned int>(iter) * 128;
            const float2 a = *reinterpret_cast<const float2 *>(src0_row + i * static_cast<long long>(sizeof(float)));
            const float2 b = *reinterpret_cast<const float2 *>(rhs + i);
            sum += a.x * b.x + a.y * b.y;
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
        *reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + row * sizeof(float) + i13 * c.dst_nb3) =
            partial[row_lane * 2] + partial[row_lane * 2 + 1];
    }
}

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

struct options {
    std::string kernel = "bf16_rows4_k512";
    int iters = 1000;
    int warmup = 50;
    int repeats = 5;
    int k = 0;
    int rows = 0;
    int cols = 0;
    int batch = 0;
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
            opts.k = std::max(1, parse_int_arg(argv[i], "--k="));
        } else if (std::strncmp(argv[i], "--rows=", 7) == 0) {
            opts.rows = std::max(1, parse_int_arg(argv[i], "--rows="));
        } else if (std::strncmp(argv[i], "--cols=", 7) == 0) {
            opts.cols = std::max(1, parse_int_arg(argv[i], "--cols="));
        } else if (std::strncmp(argv[i], "--batch=", 8) == 0) {
            opts.batch = std::max(1, parse_int_arg(argv[i], "--batch="));
        } else if (std::strcmp(argv[i], "--check") == 0) {
            opts.check = true;
        } else {
            std::fprintf(stderr,
                "usage: %s [--kernel=<name>] [--iters=N] [--warmup=N] [--repeats=N] [--check]\\n"
                "  BF16: bf16_rows4_k512, bf16_rows4_k2048, bf16_rows4_wg32, bf16_rows2_x8_wg32, bf16_rows2_x4_wg32, bf16_rows2_wg32, bf16_rows2_wg256, "
                "bf16_cols1, bf16_generic_wg64, bf16_generic_wg128, bf16_generic_wg256\\n"
                "  F16: f16_cols1, f16_rows2_x8_wg32, f16_rows2_x4_wg32, f16_rows2_wg32, f16_rows4_wg32, f16_generic\\n"
                "  F32: f32_cols1_ne2_1, f32_cols1_ne2_1_k2048_wg32, f32_rows4_k2048_lds, f32_cols8, f32_cols16, f32_rows2_cols8, f32_generic\\n",
                argv[0]);
            std::exit(2);
        }
    }
    return opts;
}

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
    if (max_abs > 3.0e-3 && max_rel > 3.0e-4) {
        std::fprintf(stderr, "%s correctness failed\n", label);
        std::exit(1);
    }
}

static void launch_bf16(
        const std::string & kernel,
        const uint16_t * src0,
        const float * src1,
        float * dst,
        int k,
        int rows,
        int cols) {
    if (kernel == "bf16_rows4_k512") {
        hrx_mul_mat_vec_bf16_rows4_k512_cols1_lds_wg256_f32<<<dim3((rows + 3) / 4, 1, 1), dim3(256, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_rows4_k2048") {
        hrx_mul_mat_vec_bf16_rows4_k2048_cols1_lds_wg256_f32<<<dim3((rows + 3) / 4, 1, 1), dim3(256, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_rows2_wg32") {
        hrx_mul_mat_vec_bf16_rows2_cols1_wg32_f32<<<dim3((rows + 1) / 2, 1, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_rows2_x4_wg32") {
        probe_mul_mat_vec_bf16_rows2_cols1_x4_wg32_f32<<<dim3((rows + 1) / 2, 1, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_rows2_x8_wg32") {
        probe_mul_mat_vec_bf16_rows2_cols1_x8_wg32_f32<<<dim3((rows + 1) / 2, 1, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_rows4_wg32") {
        probe_mul_mat_vec_bf16_rows4_cols1_wg32_f32<<<dim3((rows + 3) / 4, 1, 1), dim3(32, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_rows2_wg256") {
        hrx_mul_mat_vec_bf16_rows2_cols1_f32<<<dim3((rows + 1) / 2, 1, 1), dim3(256, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_cols1") {
        hrx_mul_mat_vec_bf16_cols1_f32<<<dim3(rows, 1, 1), dim3(256, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_generic_wg64") {
        hrx_mul_mat_vec_bf16_wg64_f32<<<dim3(rows, cols, 1), dim3(64, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_generic_wg128") {
        hrx_mul_mat_vec_bf16_wg128_f32<<<dim3(rows, cols, 1), dim3(128, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else if (kernel == "bf16_generic_wg256") {
        hrx_mul_mat_vec_bf16_f32<<<dim3(rows, cols, 1), dim3(256, 1, 1)>>>(
            src0, src1, dst, k, rows, cols);
    } else {
        std::fprintf(stderr, "unknown BF16 kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

static void launch_f16(
        const std::string & kernel,
        const __half * src0,
        const float * src1,
        float * dst,
        const hrx_mul_mat_vec_f16_batched_constants & c) {
    const dim3 grid(c.rows, c.dst_ne2 * c.dst_ne3, 1);
    const dim3 block(256, 1, 1);
    if (kernel == "f16_cols1") {
        hrx_mul_mat_vec_f16_batched_cols1_f32<<<grid, block>>>(src0, src1, dst, c);
    } else if (kernel == "f16_rows2_wg32") {
        probe_mul_mat_vec_f16_batched_rows2_cols1_wg32_f32<<<
            dim3((c.rows + 1) / 2, c.dst_ne2 * c.dst_ne3, 1), dim3(32, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f16_rows2_x4_wg32") {
        probe_mul_mat_vec_f16_batched_rows2_cols1_x4_wg32_f32<<<
            dim3((c.rows + 1) / 2, c.dst_ne2 * c.dst_ne3, 1), dim3(32, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f16_rows2_x8_wg32") {
        probe_mul_mat_vec_f16_batched_rows2_cols1_x8_wg32_f32<<<
            dim3((c.rows + 1) / 2, c.dst_ne2 * c.dst_ne3, 1), dim3(32, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f16_rows4_wg32") {
        probe_mul_mat_vec_f16_batched_rows4_cols1_wg32_f32<<<
            dim3((c.rows + 3) / 4, c.dst_ne2 * c.dst_ne3, 1), dim3(32, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f16_generic") {
        hrx_mul_mat_vec_f16_batched_f32<<<grid, block>>>(src0, src1, dst, c);
    } else {
        std::fprintf(stderr, "unknown F16 kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

static void launch_f32(
        const std::string & kernel,
        const float * src0,
        const float * src1,
        float * dst,
        const hrx_mul_mat_vec_f32_batched_constants & c) {
    const uint32_t cols_per_group =
        kernel == "f32_cols16" ? 16 :
        kernel == "f32_cols8" || kernel == "f32_rows2_cols8" ? 8 : 1;
    const uint32_t rows_per_group = kernel == "f32_rows2_cols8" ? 2 : 1;
    const dim3 grid(
        static_cast<unsigned int>((c.rows + rows_per_group - 1) / rows_per_group),
        static_cast<unsigned int>(((c.cols + cols_per_group - 1) / cols_per_group) * c.dst_ne2 * c.dst_ne3),
        1);
    if (kernel == "f32_cols1_ne2_1") {
        hrx_mul_mat_vec_f32_batched_cols1_ne2_1_f32<<<
            dim3(c.rows, c.dst_ne3, 1), dim3(256, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f32_cols1_ne2_1_k2048_wg32") {
        hrx_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32<<<
            dim3(c.rows, c.dst_ne3, 1), dim3(32, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f32_rows4_k2048_lds") {
        probe_mul_mat_vec_f32_batched_rows4_k2048_cols1_lds_wg256_f32<<<
            dim3((c.rows + 3) / 4, c.dst_ne3, 1), dim3(256, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f32_cols8") {
        hrx_mul_mat_vec_f32_batched_cols8_f32<<<grid, dim3(256, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f32_cols16") {
        hrx_mul_mat_vec_f32_batched_cols16_f32<<<grid, dim3(256, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f32_rows2_cols8") {
        hrx_mul_mat_vec_f32_batched_rows2_cols8_f32<<<grid, dim3(256, 1, 1)>>>(src0, src1, dst, c);
    } else if (kernel == "f32_generic") {
        hrx_mul_mat_vec_f32_batched_f32<<<grid, dim3(256, 1, 1)>>>(src0, src1, dst, c);
    } else {
        std::fprintf(stderr, "unknown F32 kernel: %s\n", kernel.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

static void run_bf16(const options & opts) {
    const int k = opts.k ? opts.k : (opts.kernel == "bf16_rows4_k2048" ? 2048 : 512);
    const int rows = opts.rows ? opts.rows : 2048;
    const int cols = opts.cols ? opts.cols : 1;
    const size_t src0_count = static_cast<size_t>(rows) * k;
    const size_t src1_count = static_cast<size_t>(cols) * k;
    const size_t dst_count = static_cast<size_t>(cols) * rows;

    std::vector<uint16_t> h_src0(src0_count);
    std::vector<float> h_src1(src1_count);
    std::vector<float> h_dst(dst_count, 0.0f);
    std::vector<float> h_ref(dst_count, 0.0f);

    for (size_t i = 0; i < h_src0.size(); ++i) {
        h_src0[i] = f32_to_bf16_bits(make_value(static_cast<int>(i), 3));
    }
    for (size_t i = 0; i < h_src1.size(); ++i) {
        h_src1[i] = make_value(static_cast<int>(i), 11);
    }

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += bf16_bits_to_f32(h_src0[static_cast<size_t>(row) * k + i]) *
                    h_src1[static_cast<size_t>(col) * k + i];
            }
            h_ref[static_cast<size_t>(col) * rows + row] = sum;
        }
    }

    device_buffer<uint16_t> d_src0(src0_count);
    device_buffer<float> d_src1(src1_count);
    device_buffer<float> d_dst(dst_count);
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), src0_count * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), src1_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dst.ptr, 0, dst_count * sizeof(float)));

    if (opts.check) {
        launch_bf16(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, k, rows, cols);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
        check_close(h_dst, h_ref, opts.kernel.c_str());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_bf16(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, k, rows, cols);
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
            launch_bf16(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, k, rows, cols);
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
    std::printf("kernel=%s shape=bf16 k=%d rows=%d cols=%d mean_us=%.3f stdev_us=%.3f samples=",
        opts.kernel.c_str(), k, rows, cols, mean, std::sqrt(var));
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.3f", i == 0 ? "[" : ",", samples[i]);
    }
    std::printf("]\n");
}

static void run_f16(const options & opts) {
    const int k = opts.k ? opts.k : 256;
    const int rows = opts.rows ? opts.rows : 256;
    const int cols = opts.cols ? opts.cols : 1;
    const int batch = opts.batch ? opts.batch : 16;
    const size_t src0_count = static_cast<size_t>(batch) * rows * k;
    const size_t src1_count = static_cast<size_t>(batch) * cols * k;
    const size_t dst_count = static_cast<size_t>(batch) * cols * rows;

    std::vector<__half> h_src0(src0_count);
    std::vector<float> h_src1(src1_count);
    std::vector<float> h_dst(dst_count, 0.0f);
    std::vector<float> h_ref(dst_count, 0.0f);

    for (size_t i = 0; i < h_src0.size(); ++i) {
        h_src0[i] = __float2half(make_value(static_cast<int>(i), 5));
    }
    for (size_t i = 0; i < h_src1.size(); ++i) {
        h_src1[i] = make_value(static_cast<int>(i), 13);
    }

    for (int b = 0; b < batch; ++b) {
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                float sum = 0.0f;
                for (int i = 0; i < k; ++i) {
                    sum += __half2float(h_src0[(static_cast<size_t>(b) * rows + row) * k + i]) *
                        h_src1[(static_cast<size_t>(b) * cols + col) * k + i];
                }
                h_ref[(static_cast<size_t>(b) * cols + col) * rows + row] = sum;
            }
        }
    }

    device_buffer<__half> d_src0(src0_count);
    device_buffer<float> d_src1(src1_count);
    device_buffer<float> d_dst(dst_count);
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), src0_count * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), src1_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dst.ptr, 0, dst_count * sizeof(float)));

    const hrx_mul_mat_vec_f16_batched_constants c = {
        /* .k         = */ k,
        /* .rows      = */ rows,
        /* .cols      = */ cols,
        /* .dst_ne2   = */ batch,
        /* .dst_ne3   = */ 1,
        /* .src0_ne2  = */ batch,
        /* .src0_ne3  = */ 1,
        /* .src0_nb1  = */ static_cast<long long>(k * sizeof(__half)),
        /* .src0_nb2  = */ static_cast<long long>(rows * k * sizeof(__half)),
        /* .src0_nb3  = */ static_cast<long long>(batch * rows * k * sizeof(__half)),
        /* .src1_nb1  = */ static_cast<long long>(k * sizeof(float)),
        /* .src1_nb2  = */ static_cast<long long>(cols * k * sizeof(float)),
        /* .src1_nb3  = */ static_cast<long long>(batch * cols * k * sizeof(float)),
        /* .dst_nb1   = */ static_cast<long long>(rows * sizeof(float)),
        /* .dst_nb2   = */ static_cast<long long>(cols * rows * sizeof(float)),
        /* .dst_nb3   = */ static_cast<long long>(batch * cols * rows * sizeof(float)),
    };

    if (opts.check) {
        launch_f16(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, c);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
        check_close(h_dst, h_ref, opts.kernel.c_str());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_f16(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, c);
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
            launch_f16(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, c);
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
    std::printf("kernel=%s shape=f16 k=%d rows=%d cols=%d batch=%d mean_us=%.3f stdev_us=%.3f samples=",
        opts.kernel.c_str(), k, rows, cols, batch, mean, std::sqrt(var));
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.3f", i == 0 ? "[" : ",", samples[i]);
    }
    std::printf("]\n");
}

static void run_f32(const options & opts) {
    const int k = opts.k ? opts.k : 2048;
    const int rows = opts.rows ? opts.rows : 16;
    const int cols = opts.cols ? opts.cols : 1;
    const int batch = opts.batch ? opts.batch : 80;
    const size_t src0_count = static_cast<size_t>(batch) * rows * k;
    const size_t src1_count = static_cast<size_t>(batch) * cols * k;
    const size_t dst_count = static_cast<size_t>(batch) * cols * rows;

    std::vector<float> h_src0(src0_count);
    std::vector<float> h_src1(src1_count);
    std::vector<float> h_dst(dst_count, 0.0f);
    std::vector<float> h_ref(dst_count, 0.0f);

    for (size_t i = 0; i < h_src0.size(); ++i) {
        h_src0[i] = make_value(static_cast<int>(i), 7);
    }
    for (size_t i = 0; i < h_src1.size(); ++i) {
        h_src1[i] = make_value(static_cast<int>(i), 17);
    }

    for (int b = 0; b < batch; ++b) {
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                float sum = 0.0f;
                for (int i = 0; i < k; ++i) {
                    sum += h_src0[(static_cast<size_t>(b) * rows + row) * k + i] *
                        h_src1[(static_cast<size_t>(b) * cols + col) * k + i];
                }
                h_ref[(static_cast<size_t>(b) * cols + col) * rows + row] = sum;
            }
        }
    }

    device_buffer<float> d_src0(src0_count);
    device_buffer<float> d_src1(src1_count);
    device_buffer<float> d_dst(dst_count);
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), src0_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), src1_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dst.ptr, 0, dst_count * sizeof(float)));

    const hrx_mul_mat_vec_f32_batched_constants c = {
        /* .k         = */ k,
        /* .rows      = */ rows,
        /* .cols      = */ cols,
        /* .dst_ne2   = */ 1,
        /* .dst_ne3   = */ batch,
        /* .src0_ne2  = */ 1,
        /* .src0_ne3  = */ batch,
        /* .src0_nb1  = */ static_cast<long long>(k * sizeof(float)),
        /* .src0_nb2  = */ static_cast<long long>(rows * k * sizeof(float)),
        /* .src0_nb3  = */ static_cast<long long>(rows * k * sizeof(float)),
        /* .src1_nb1  = */ static_cast<long long>(k * sizeof(float)),
        /* .src1_nb2  = */ static_cast<long long>(cols * k * sizeof(float)),
        /* .src1_nb3  = */ static_cast<long long>(cols * k * sizeof(float)),
        /* .dst_nb1   = */ static_cast<long long>(rows * sizeof(float)),
        /* .dst_nb2   = */ static_cast<long long>(cols * rows * sizeof(float)),
        /* .dst_nb3   = */ static_cast<long long>(cols * rows * sizeof(float)),
    };

    if (opts.check) {
        launch_f32(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, c);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
        check_close(h_dst, h_ref, opts.kernel.c_str());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_f32(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, c);
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
            launch_f32(opts.kernel, d_src0.ptr, d_src1.ptr, d_dst.ptr, c);
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
    std::printf("kernel=%s shape=f32 k=%d rows=%d cols=%d batch=%d mean_us=%.3f stdev_us=%.3f samples=",
        opts.kernel.c_str(), k, rows, cols, batch, mean, std::sqrt(var));
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.3f", i == 0 ? "[" : ",", samples[i]);
    }
    std::printf("]\n");
}

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    HIP_CHECK(hipSetDevice(0));
    if (opts.kernel.rfind("bf16_", 0) == 0) {
        run_bf16(opts);
    } else if (opts.kernel.rfind("f16_", 0) == 0) {
        run_f16(opts);
    } else if (opts.kernel.rfind("f32_", 0) == 0) {
        run_f32(opts);
    } else {
        std::fprintf(stderr, "unknown kernel family: %s\n", opts.kernel.c_str());
        return 2;
    }
    return 0;
}
