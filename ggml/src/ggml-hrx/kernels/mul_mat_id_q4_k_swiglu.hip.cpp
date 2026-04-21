#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q4_K_id_swiglu {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct hrx_mul_mat_id_q4_k_swiglu_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long gate_nb1;
    long long gate_nb2;
    long long up_nb1;
    long long up_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long ids_nb0;
    long long ids_nb1;
    long long dst_nb1;
    long long dst_nb2;
};

struct hrx_mul_mat_id_q4_k_swiglu_grouped_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long route_capacity;
    long long gate_nb1;
    long long gate_nb2;
    long long up_nb1;
    long long up_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long dst_nb1;
    long long dst_nb2;
};

static __device__ __forceinline__ void hrx_get_scale_min_k4_id_swiglu(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    const uint32_t scales0 = *reinterpret_cast<const uint32_t *>(q);
    const uint32_t mins0 = *reinterpret_cast<const uint32_t *>(q + 4);
    if (j < 4) {
        *d = static_cast<uint8_t>((scales0 >> (8 * j)) & 63u);
        *m = static_cast<uint8_t>((mins0 >> (8 * j)) & 63u);
    } else {
        const int k = j - 4;
        const uint32_t scale8 = *reinterpret_cast<const uint32_t *>(q + 8);
        const uint32_t packed = (scale8 >> (8 * k)) & 0xFFu;
        *d = static_cast<uint8_t>((packed & 0x0Fu) | (((scales0 >> (8 * k + 6)) & 3u) << 4));
        *m = static_cast<uint8_t>((packed >> 4) | (((mins0 >> (8 * k + 6)) & 3u) << 4));
    }
}

static __device__ __forceinline__ void hrx_q4_k_swiglu_scale_min4(
        const uint8_t * scales,
        int v_im,
        float d,
        float dmin,
        float & d0,
        float & d1,
        float & d2,
        float & d3,
        float & min0,
        float & min1,
        float & min2,
        float & min3) {
    const uint16_t * scales16 = reinterpret_cast<const uint16_t *>(scales);
    const uint32_t scale0 = scales16[v_im];
    const uint32_t scale4 = scales16[v_im + 2];
    const uint32_t scale8_raw = scales16[v_im + 4];
    const uint32_t scale_0_4_l = (scale4 << 16) | scale0;
    const uint32_t scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2;
    const uint32_t scale_0_4_l6 = scale_0_4_l & 0x3F3F3F3Fu;
    const uint32_t scale8 = (((scale8_raw << 12) | scale8_raw) & 0x0F0F0F0Fu) | scale_0_4_h;

    d0 = d * static_cast<float>((scale_0_4_l6 >> 0) & 0xFFu);
    d1 = d * static_cast<float>((scale_0_4_l6 >> 8) & 0xFFu);
    d2 = d * static_cast<float>((scale8 >> 0) & 0xFFu);
    d3 = d * static_cast<float>((scale8 >> 8) & 0xFFu);
    min0 = dmin * static_cast<float>((scale_0_4_l6 >> 16) & 0xFFu);
    min1 = dmin * static_cast<float>((scale_0_4_l6 >> 24) & 0xFFu);
    min2 = dmin * static_cast<float>((scale8 >> 16) & 0xFFu);
    min3 = dmin * static_cast<float>((scale8 >> 24) & 0xFFu);
}

template <int I>
static __device__ __forceinline__ float hrx_q4_k_swiglu_q_from_pack(unsigned long long pack, bool high) {
    const unsigned int byte = static_cast<unsigned int>((pack >> (8 * I)) & 0xFFu);
    return high ? static_cast<float>(byte >> 4) : static_cast<float>(byte & 0x0Fu);
}

static __device__ __forceinline__ float hrx_q4_k_swiglu_sum_float4(float4 v) {
    return (v.x + v.y) + (v.z + v.w);
}

static __device__ __forceinline__ float hrx_q4_k_swiglu_dot4_from_pack(
        unsigned long long pack,
        bool high,
        float4 y) {
    return hrx_q4_k_swiglu_q_from_pack<0>(pack, high) * y.x +
           hrx_q4_k_swiglu_q_from_pack<1>(pack, high) * y.y +
           hrx_q4_k_swiglu_q_from_pack<2>(pack, high) * y.z +
           hrx_q4_k_swiglu_q_from_pack<3>(pack, high) * y.w;
}

template <int WG_SIZE>
static __device__ __forceinline__ float hrx_reduce_wg_swiglu(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (WG_SIZE <= warpSize) {
        return sum;
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < ((WG_SIZE + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_reduce_wg4_swiglu(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
    }
    if (WG_SIZE <= warpSize) {
        return;
    }
    if (lane == 0) {
        shared[wave + 0 * waves] = sum0;
        shared[wave + 1 * waves] = sum1;
        shared[wave + 2 * waves] = sum2;
        shared[wave + 3 * waves] = sum3;
    }
    __syncthreads();

    sum0 = lane < waves ? shared[lane + 0 * waves] : 0.0f;
    sum1 = lane < waves ? shared[lane + 1 * waves] : 0.0f;
    sum2 = lane < waves ? shared[lane + 2 * waves] : 0.0f;
    sum3 = lane < waves ? shared[lane + 3 * waves] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum0 += __shfl_down(sum0, offset);
            sum1 += __shfl_down(sum1, offset);
            sum2 += __shfl_down(sum2, offset);
            sum3 += __shfl_down(sum3, offset);
        }
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_reduce_wg8_swiglu(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float & sum4,
        float & sum5,
        float & sum6,
        float & sum7,
        float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
        sum4 += __shfl_down(sum4, offset);
        sum5 += __shfl_down(sum5, offset);
        sum6 += __shfl_down(sum6, offset);
        sum7 += __shfl_down(sum7, offset);
    }
    if (WG_SIZE <= warpSize) {
        return;
    }
    if (lane == 0) {
        shared[wave + 0 * waves] = sum0;
        shared[wave + 1 * waves] = sum1;
        shared[wave + 2 * waves] = sum2;
        shared[wave + 3 * waves] = sum3;
        shared[wave + 4 * waves] = sum4;
        shared[wave + 5 * waves] = sum5;
        shared[wave + 6 * waves] = sum6;
        shared[wave + 7 * waves] = sum7;
    }
    __syncthreads();

    sum0 = lane < waves ? shared[lane + 0 * waves] : 0.0f;
    sum1 = lane < waves ? shared[lane + 1 * waves] : 0.0f;
    sum2 = lane < waves ? shared[lane + 2 * waves] : 0.0f;
    sum3 = lane < waves ? shared[lane + 3 * waves] : 0.0f;
    sum4 = lane < waves ? shared[lane + 4 * waves] : 0.0f;
    sum5 = lane < waves ? shared[lane + 5 * waves] : 0.0f;
    sum6 = lane < waves ? shared[lane + 6 * waves] : 0.0f;
    sum7 = lane < waves ? shared[lane + 7 * waves] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum0 += __shfl_down(sum0, offset);
            sum1 += __shfl_down(sum1, offset);
            sum2 += __shfl_down(sum2, offset);
            sum3 += __shfl_down(sum3, offset);
            sum4 += __shfl_down(sum4, offset);
            sum5 += __shfl_down(sum5, offset);
            sum6 += __shfl_down(sum6, offset);
            sum7 += __shfl_down(sum7, offset);
        }
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_id_q4_k_swiglu_f32_impl(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float gate_sumsh[WG_SIZE / 32];
    __shared__ float up_sumsh[WG_SIZE / 32];
    const char * gate_row_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2 + row * c.gate_nb1;
    const char * up_row_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2 + row * c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const hrx_block_q4_K_id_swiglu * gate_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        uint8_t gate_sc = 0;
        uint8_t gate_m = 0;
        uint8_t up_sc = 0;
        uint8_t up_m = 0;
        hrx_get_scale_min_k4_id_swiglu(group, gate_block->scales, &gate_sc, &gate_m);
        hrx_get_scale_min_k4_id_swiglu(group, up_block->scales, &up_sc, &up_m);

        const float gate_d = __half2float(__ushort_as_half(gate_block->d)) * static_cast<float>(gate_sc);
        const float gate_min = __half2float(__ushort_as_half(gate_block->dmin)) * static_cast<float>(gate_m);
        const float up_d = __half2float(__ushort_as_half(up_block->d)) * static_cast<float>(up_sc);
        const float up_min = __half2float(__ushort_as_half(up_block->dmin)) * static_cast<float>(up_m);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        const uint32_t gate_packed_qs = *reinterpret_cast<const uint32_t *>(gate_block->qs + qs_base);
        const uint32_t up_packed_qs = *reinterpret_cast<const uint32_t *>(up_block->qs + qs_base);
        const float4 b4 = *reinterpret_cast<const float4 *>(src1_col + src_base * sizeof(float));
        const float ysum = hrx_q4_k_swiglu_sum_float4(b4);
        const bool high = (group & 1) != 0;
        gate_sum += gate_d * hrx_q4_k_swiglu_dot4_from_pack(gate_packed_qs, high, b4) - gate_min * ysum;
        up_sum += up_d * hrx_q4_k_swiglu_dot4_from_pack(up_packed_qs, high, b4) - up_min * ysum;
    }

    gate_sum = hrx_reduce_wg_swiglu<WG_SIZE>(gate_sum, gate_sumsh);
    up_sum = hrx_reduce_wg_swiglu<WG_SIZE>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            up_sum * silu_gate;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    hrx_mul_mat_id_q4_k_swiglu_f32_impl<256>(gate, up, src1, ids, dst, c);
}

static __global__ void hrx_mul_mat_id_q4_k_swiglu_wg128_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    hrx_mul_mat_id_q4_k_swiglu_f32_impl<128>(gate, up, src1, ids, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    hrx_mul_mat_id_q4_k_swiglu_f32_impl<64>(gate, up, src1, ids, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_row2_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[4 * (64 / 32)];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum0 = 0.0f;
    float up_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum1 = 0.0f;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        uint8_t gate_sc0 = 0;
        uint8_t gate_m0 = 0;
        uint8_t gate_sc1 = 0;
        uint8_t gate_m1 = 0;
        uint8_t up_sc0 = 0;
        uint8_t up_m0 = 0;
        uint8_t up_sc1 = 0;
        uint8_t up_m1 = 0;
        hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
        hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
        hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
        hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);

        const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
        const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
        const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
        const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
        const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
        const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
        const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
        const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        const uint32_t gate_packed_qs0 = *reinterpret_cast<const uint32_t *>(gate_block0->qs + qs_base);
        const uint32_t gate_packed_qs1 = *reinterpret_cast<const uint32_t *>(gate_block1->qs + qs_base);
        const uint32_t up_packed_qs0 = *reinterpret_cast<const uint32_t *>(up_block0->qs + qs_base);
        const uint32_t up_packed_qs1 = *reinterpret_cast<const uint32_t *>(up_block1->qs + qs_base);
        const float4 b4 = *reinterpret_cast<const float4 *>(src1_col + src_base * sizeof(float));
        const float ysum = hrx_q4_k_swiglu_sum_float4(b4);
        const bool high = (group & 1) != 0;
        gate_sum0 += gate_d0 * hrx_q4_k_swiglu_dot4_from_pack(gate_packed_qs0, high, b4) - gate_min0 * ysum;
        gate_sum1 += gate_d1 * hrx_q4_k_swiglu_dot4_from_pack(gate_packed_qs1, high, b4) - gate_min1 * ysum;
        up_sum0 += up_d0 * hrx_q4_k_swiglu_dot4_from_pack(up_packed_qs0, high, b4) - up_min0 * ysum;
        up_sum1 += up_d1 * hrx_q4_k_swiglu_dot4_from_pack(up_packed_qs1, high, b4) - up_min1 * ysum;
    }

    hrx_reduce_wg4_swiglu<64>(gate_sum0, up_sum0, gate_sum1, up_sum1, sumsh);

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + id_pos * c.dst_nb1 + token * c.dst_nb2;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0 * silu_gate0;
        *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1 * silu_gate1;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_row4_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 3 >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[8 * (64 / 32)];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * gate_row2_base = gate_row1_base + c.gate_nb1;
    const char * gate_row3_base = gate_row2_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;
    const char * up_row2_base = up_row1_base + c.up_nb1;
    const char * up_row3_base = up_row2_base + c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum0 = 0.0f;
    float up_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum1 = 0.0f;
    float gate_sum2 = 0.0f;
    float up_sum2 = 0.0f;
    float gate_sum3 = 0.0f;
    float up_sum3 = 0.0f;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * gate_block2 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row2_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * gate_block3 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row3_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block2 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row2_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block3 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row3_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        uint8_t gate_sc0 = 0;
        uint8_t gate_m0 = 0;
        uint8_t gate_sc1 = 0;
        uint8_t gate_m1 = 0;
        uint8_t gate_sc2 = 0;
        uint8_t gate_m2 = 0;
        uint8_t gate_sc3 = 0;
        uint8_t gate_m3 = 0;
        uint8_t up_sc0 = 0;
        uint8_t up_m0 = 0;
        uint8_t up_sc1 = 0;
        uint8_t up_m1 = 0;
        uint8_t up_sc2 = 0;
        uint8_t up_m2 = 0;
        uint8_t up_sc3 = 0;
        uint8_t up_m3 = 0;
        hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
        hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
        hrx_get_scale_min_k4_id_swiglu(group, gate_block2->scales, &gate_sc2, &gate_m2);
        hrx_get_scale_min_k4_id_swiglu(group, gate_block3->scales, &gate_sc3, &gate_m3);
        hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
        hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);
        hrx_get_scale_min_k4_id_swiglu(group, up_block2->scales, &up_sc2, &up_m2);
        hrx_get_scale_min_k4_id_swiglu(group, up_block3->scales, &up_sc3, &up_m3);

        const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
        const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
        const float gate_d2 = __half2float(__ushort_as_half(gate_block2->d)) * static_cast<float>(gate_sc2);
        const float gate_d3 = __half2float(__ushort_as_half(gate_block3->d)) * static_cast<float>(gate_sc3);
        const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
        const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
        const float up_d2 = __half2float(__ushort_as_half(up_block2->d)) * static_cast<float>(up_sc2);
        const float up_d3 = __half2float(__ushort_as_half(up_block3->d)) * static_cast<float>(up_sc3);
        const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
        const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
        const float gate_min2 = __half2float(__ushort_as_half(gate_block2->dmin)) * static_cast<float>(gate_m2);
        const float gate_min3 = __half2float(__ushort_as_half(gate_block3->dmin)) * static_cast<float>(gate_m3);
        const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
        const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
        const float up_min2 = __half2float(__ushort_as_half(up_block2->dmin)) * static_cast<float>(up_m2);
        const float up_min3 = __half2float(__ushort_as_half(up_block3->dmin)) * static_cast<float>(up_m3);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        const float4 b4 = *reinterpret_cast<const float4 *>(src1_col + src_base * sizeof(float));
        const float ysum = hrx_q4_k_swiglu_sum_float4(b4);
        const bool high = (group & 1) != 0;
        const uint32_t gate_packed_qs0 = *reinterpret_cast<const uint32_t *>(gate_block0->qs + qs_base);
        const uint32_t gate_packed_qs1 = *reinterpret_cast<const uint32_t *>(gate_block1->qs + qs_base);
        const uint32_t gate_packed_qs2 = *reinterpret_cast<const uint32_t *>(gate_block2->qs + qs_base);
        const uint32_t gate_packed_qs3 = *reinterpret_cast<const uint32_t *>(gate_block3->qs + qs_base);
        const uint32_t up_packed_qs0 = *reinterpret_cast<const uint32_t *>(up_block0->qs + qs_base);
        const uint32_t up_packed_qs1 = *reinterpret_cast<const uint32_t *>(up_block1->qs + qs_base);
        const uint32_t up_packed_qs2 = *reinterpret_cast<const uint32_t *>(up_block2->qs + qs_base);
        const uint32_t up_packed_qs3 = *reinterpret_cast<const uint32_t *>(up_block3->qs + qs_base);
#define HRX_Q4K_SWIGLU_ROW4_ACC(N) \
        do { \
            gate_sum##N += gate_d##N * hrx_q4_k_swiglu_dot4_from_pack(gate_packed_qs##N, high, b4) - gate_min##N * ysum; \
            up_sum##N += up_d##N * hrx_q4_k_swiglu_dot4_from_pack(up_packed_qs##N, high, b4) - up_min##N * ysum; \
        } while (0)
        HRX_Q4K_SWIGLU_ROW4_ACC(0);
        HRX_Q4K_SWIGLU_ROW4_ACC(1);
        HRX_Q4K_SWIGLU_ROW4_ACC(2);
        HRX_Q4K_SWIGLU_ROW4_ACC(3);
#undef HRX_Q4K_SWIGLU_ROW4_ACC
    }

    hrx_reduce_wg8_swiglu<64>(
        gate_sum0, up_sum0,
        gate_sum1, up_sum1,
        gate_sum2, up_sum2,
        gate_sum3, up_sum3,
        sumsh);

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + id_pos * c.dst_nb1 + token * c.dst_nb2;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        const float silu_gate2 = gate_sum2 / (1.0f + __expf(-gate_sum2));
        const float silu_gate3 = gate_sum3 / (1.0f + __expf(-gate_sum3));
        *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0 * silu_gate0;
        *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1 * silu_gate1;
        *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = up_sum2 * silu_gate2;
        *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = up_sum3 * silu_gate3;
    }
}

static __global__ void hrx_mul_mat_id_q4_k_swiglu_grouped_row4_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_grouped_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long expert = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 3 >= c.rows || expert >= c.n_experts) {
        return;
    }

    const uint32_t count = counts[expert];
    if (count == 0) {
        return;
    }

    __shared__ float gate_sumsh0[64 / 32];
    __shared__ float up_sumsh0[64 / 32];
    __shared__ float gate_sumsh1[64 / 32];
    __shared__ float up_sumsh1[64 / 32];
    __shared__ float gate_sumsh2[64 / 32];
    __shared__ float up_sumsh2[64 / 32];
    __shared__ float gate_sumsh3[64 / 32];
    __shared__ float up_sumsh3[64 / 32];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * gate_row2_base = gate_row1_base + c.gate_nb1;
    const char * gate_row3_base = gate_row2_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;
    const char * up_row2_base = up_row1_base + c.up_nb1;
    const char * up_row3_base = up_row2_base + c.up_nb1;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;
    const uint32_t * expert_routes = routes + expert * c.route_capacity;

    for (uint32_t route_idx = 0; route_idx < count; route_idx += 2) {
        const uint32_t route0 = expert_routes[route_idx];
        const uint32_t route1 = route_idx + 1 < count ? expert_routes[route_idx + 1] : route0;
        const bool has_route1 = route_idx + 1 < count;
        const long long id_pos0 = route0 % c.n_ids;
        const long long token0 = route0 / c.n_ids;
        const long long id_pos1 = route1 % c.n_ids;
        const long long token1 = route1 / c.n_ids;

        const char * src1_col0 = reinterpret_cast<const char *>(src1) + id_pos0 * c.src1_nb1 + token0 * c.src1_nb2;
        const char * src1_col1 = reinterpret_cast<const char *>(src1) + id_pos1 * c.src1_nb1 + token1 * c.src1_nb2;
        float gate_sum0a = 0.0f;
        float up_sum0a = 0.0f;
        float gate_sum1a = 0.0f;
        float up_sum1a = 0.0f;
        float gate_sum2a = 0.0f;
        float up_sum2a = 0.0f;
        float gate_sum3a = 0.0f;
        float up_sum3a = 0.0f;
        float gate_sum0b = 0.0f;
        float up_sum0b = 0.0f;
        float gate_sum1b = 0.0f;
        float up_sum1b = 0.0f;
        float gate_sum2b = 0.0f;
        float up_sum2b = 0.0f;
        float gate_sum3b = 0.0f;
        float up_sum3b = 0.0f;

        for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block2 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row2_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block3 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row3_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block2 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row2_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block3 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row3_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

            uint8_t gate_sc0 = 0;
            uint8_t gate_m0 = 0;
            uint8_t gate_sc1 = 0;
            uint8_t gate_m1 = 0;
            uint8_t gate_sc2 = 0;
            uint8_t gate_m2 = 0;
            uint8_t gate_sc3 = 0;
            uint8_t gate_m3 = 0;
            uint8_t up_sc0 = 0;
            uint8_t up_m0 = 0;
            uint8_t up_sc1 = 0;
            uint8_t up_m1 = 0;
            uint8_t up_sc2 = 0;
            uint8_t up_m2 = 0;
            uint8_t up_sc3 = 0;
            uint8_t up_m3 = 0;
            hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block2->scales, &gate_sc2, &gate_m2);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block3->scales, &gate_sc3, &gate_m3);
            hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
            hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);
            hrx_get_scale_min_k4_id_swiglu(group, up_block2->scales, &up_sc2, &up_m2);
            hrx_get_scale_min_k4_id_swiglu(group, up_block3->scales, &up_sc3, &up_m3);

            const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
            const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
            const float gate_d2 = __half2float(__ushort_as_half(gate_block2->d)) * static_cast<float>(gate_sc2);
            const float gate_d3 = __half2float(__ushort_as_half(gate_block3->d)) * static_cast<float>(gate_sc3);
            const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
            const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
            const float up_d2 = __half2float(__ushort_as_half(up_block2->d)) * static_cast<float>(up_sc2);
            const float up_d3 = __half2float(__ushort_as_half(up_block3->d)) * static_cast<float>(up_sc3);
            const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
            const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
            const float gate_min2 = __half2float(__ushort_as_half(gate_block2->dmin)) * static_cast<float>(gate_m2);
            const float gate_min3 = __half2float(__ushort_as_half(gate_block3->dmin)) * static_cast<float>(gate_m3);
            const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
            const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
            const float up_min2 = __half2float(__ushort_as_half(up_block2->dmin)) * static_cast<float>(up_m2);
            const float up_min3 = __half2float(__ushort_as_half(up_block3->dmin)) * static_cast<float>(up_m3);
            const long long src_base = block_idx * 256 + group * 32 + lane;
            const int qs_base = (group >> 1) * 32 + lane;

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float b0 = *reinterpret_cast<const float *>(src1_col0 + (src_base + j) * sizeof(float));
                const float b1 = has_route1 ?
                    *reinterpret_cast<const float *>(src1_col1 + (src_base + j) * sizeof(float)) : 0.0f;
#define HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(N) \
                do { \
                    const uint8_t gate_packed = gate_block##N->qs[qs_base + j]; \
                    const uint8_t up_packed = up_block##N->qs[qs_base + j]; \
                    const float gate_q = (group & 1) ? static_cast<float>(gate_packed >> 4) : static_cast<float>(gate_packed & 0x0F); \
                    const float up_q = (group & 1) ? static_cast<float>(up_packed >> 4) : static_cast<float>(up_packed & 0x0F); \
                    const float gate_val = gate_d##N * gate_q - gate_min##N; \
                    const float up_val = up_d##N * up_q - up_min##N; \
                    gate_sum##N##a += gate_val * b0; \
                    up_sum##N##a += up_val * b0; \
                    gate_sum##N##b += gate_val * b1; \
                    up_sum##N##b += up_val * b1; \
                } while (0)
                HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(0);
                HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(1);
                HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(2);
                HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(3);
#undef HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC
            }
        }

        gate_sum0a = hrx_reduce_wg_swiglu<64>(gate_sum0a, gate_sumsh0);
        up_sum0a = hrx_reduce_wg_swiglu<64>(up_sum0a, up_sumsh0);
        gate_sum1a = hrx_reduce_wg_swiglu<64>(gate_sum1a, gate_sumsh1);
        up_sum1a = hrx_reduce_wg_swiglu<64>(up_sum1a, up_sumsh1);
        gate_sum2a = hrx_reduce_wg_swiglu<64>(gate_sum2a, gate_sumsh2);
        up_sum2a = hrx_reduce_wg_swiglu<64>(up_sum2a, up_sumsh2);
        gate_sum3a = hrx_reduce_wg_swiglu<64>(gate_sum3a, gate_sumsh3);
        up_sum3a = hrx_reduce_wg_swiglu<64>(up_sum3a, up_sumsh3);

        if (tid == 0) {
            char * dst_base = reinterpret_cast<char *>(dst) + id_pos0 * c.dst_nb1 + token0 * c.dst_nb2;
            const float silu_gate0 = gate_sum0a / (1.0f + __expf(-gate_sum0a));
            const float silu_gate1 = gate_sum1a / (1.0f + __expf(-gate_sum1a));
            const float silu_gate2 = gate_sum2a / (1.0f + __expf(-gate_sum2a));
            const float silu_gate3 = gate_sum3a / (1.0f + __expf(-gate_sum3a));
            *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0a * silu_gate0;
            *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1a * silu_gate1;
            *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = up_sum2a * silu_gate2;
            *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = up_sum3a * silu_gate3;
        }
        __syncthreads();

        if (has_route1) {
            gate_sum0b = hrx_reduce_wg_swiglu<64>(gate_sum0b, gate_sumsh0);
            up_sum0b = hrx_reduce_wg_swiglu<64>(up_sum0b, up_sumsh0);
            gate_sum1b = hrx_reduce_wg_swiglu<64>(gate_sum1b, gate_sumsh1);
            up_sum1b = hrx_reduce_wg_swiglu<64>(up_sum1b, up_sumsh1);
            gate_sum2b = hrx_reduce_wg_swiglu<64>(gate_sum2b, gate_sumsh2);
            up_sum2b = hrx_reduce_wg_swiglu<64>(up_sum2b, up_sumsh2);
            gate_sum3b = hrx_reduce_wg_swiglu<64>(gate_sum3b, gate_sumsh3);
            up_sum3b = hrx_reduce_wg_swiglu<64>(up_sum3b, up_sumsh3);

            if (tid == 0) {
                char * dst_base = reinterpret_cast<char *>(dst) + id_pos1 * c.dst_nb1 + token1 * c.dst_nb2;
                const float silu_gate0 = gate_sum0b / (1.0f + __expf(-gate_sum0b));
                const float silu_gate1 = gate_sum1b / (1.0f + __expf(-gate_sum1b));
                const float silu_gate2 = gate_sum2b / (1.0f + __expf(-gate_sum2b));
                const float silu_gate3 = gate_sum3b / (1.0f + __expf(-gate_sum3b));
                *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0b * silu_gate0;
                *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1b * silu_gate1;
                *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = up_sum2b * silu_gate2;
                *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = up_sum3b * silu_gate3;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_grouped_row2_route4_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_grouped_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long expert = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= c.rows || expert >= c.n_experts) {
        return;
    }

    const uint32_t count = counts[expert];
    if (count == 0) {
        return;
    }

    __shared__ float sumsh[8 * 4 * (64 / 32)];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;
    const uint32_t * expert_routes = routes + expert * c.route_capacity;

    for (uint32_t route_idx = 0; route_idx < count; route_idx += 4) {
        const uint32_t route_a = expert_routes[route_idx];
        const uint32_t route_b = route_idx + 1 < count ? expert_routes[route_idx + 1] : route_a;
        const uint32_t route_c = route_idx + 2 < count ? expert_routes[route_idx + 2] : route_a;
        const uint32_t route_d = route_idx + 3 < count ? expert_routes[route_idx + 3] : route_a;
        const bool has_b = route_idx + 1 < count;
        const bool has_c = route_idx + 2 < count;
        const bool has_d = route_idx + 3 < count;
        const long long id_a = route_a % c.n_ids;
        const long long tok_a = route_a / c.n_ids;
        const long long id_b = route_b % c.n_ids;
        const long long tok_b = route_b / c.n_ids;
        const long long id_c = route_c % c.n_ids;
        const long long tok_c = route_c / c.n_ids;
        const long long id_d = route_d % c.n_ids;
        const long long tok_d = route_d / c.n_ids;
        const char * src1_a = reinterpret_cast<const char *>(src1) + id_a * c.src1_nb1 + tok_a * c.src1_nb2;
        const char * src1_b = reinterpret_cast<const char *>(src1) + id_b * c.src1_nb1 + tok_b * c.src1_nb2;
        const char * src1_c = reinterpret_cast<const char *>(src1) + id_c * c.src1_nb1 + tok_c * c.src1_nb2;
        const char * src1_d = reinterpret_cast<const char *>(src1) + id_d * c.src1_nb1 + tok_d * c.src1_nb2;

        float gate_sum0a = 0.0f, up_sum0a = 0.0f, gate_sum1a = 0.0f, up_sum1a = 0.0f;
        float gate_sum0b = 0.0f, up_sum0b = 0.0f, gate_sum1b = 0.0f, up_sum1b = 0.0f;
        float gate_sum0c = 0.0f, up_sum0c = 0.0f, gate_sum1c = 0.0f, up_sum1c = 0.0f;
        float gate_sum0d = 0.0f, up_sum0d = 0.0f, gate_sum1d = 0.0f, up_sum1d = 0.0f;

        for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

            uint8_t gate_sc0 = 0, gate_m0 = 0, gate_sc1 = 0, gate_m1 = 0;
            uint8_t up_sc0 = 0, up_m0 = 0, up_sc1 = 0, up_m1 = 0;
            hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
            hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
            hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);

            const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
            const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
            const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
            const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
            const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
            const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
            const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
            const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
            const long long src_base = block_idx * 256 + group * 32 + lane;
            const int qs_base = (group >> 1) * 32 + lane;

            const uint32_t gate_packed_qs0 = *reinterpret_cast<const uint32_t *>(gate_block0->qs + qs_base);
            const uint32_t gate_packed_qs1 = *reinterpret_cast<const uint32_t *>(gate_block1->qs + qs_base);
            const uint32_t up_packed_qs0 = *reinterpret_cast<const uint32_t *>(up_block0->qs + qs_base);
            const uint32_t up_packed_qs1 = *reinterpret_cast<const uint32_t *>(up_block1->qs + qs_base);

#define HRX_Q4K_SWIGLU_ROW2_ROUTE4_LOAD4(S, HAS) \
            float4 b4_##S; \
            if (HAS) { \
                b4_##S = *reinterpret_cast<const float4 *>(src1_##S + src_base * sizeof(float)); \
            } else { \
                b4_##S = { 0.0f, 0.0f, 0.0f, 0.0f }; \
            }
            HRX_Q4K_SWIGLU_ROW2_ROUTE4_LOAD4(a, true);
            HRX_Q4K_SWIGLU_ROW2_ROUTE4_LOAD4(b, has_b);
            HRX_Q4K_SWIGLU_ROW2_ROUTE4_LOAD4(c, has_c);
            HRX_Q4K_SWIGLU_ROW2_ROUTE4_LOAD4(d, has_d);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE4_LOAD4

            const bool high = (group & 1) != 0;
#define HRX_Q4K_SWIGLU_ROW2_ROUTE4_YSUM(S) \
            const float ysum_##S = hrx_q4_k_swiglu_sum_float4(b4_##S)
            HRX_Q4K_SWIGLU_ROW2_ROUTE4_YSUM(a);
            HRX_Q4K_SWIGLU_ROW2_ROUTE4_YSUM(b);
            HRX_Q4K_SWIGLU_ROW2_ROUTE4_YSUM(c);
            HRX_Q4K_SWIGLU_ROW2_ROUTE4_YSUM(d);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE4_YSUM

#define HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(N, S) \
            do { \
                gate_sum##N##S += gate_d##N * hrx_q4_k_swiglu_dot4_from_pack(gate_packed_qs##N, high, b4_##S) - gate_min##N * ysum_##S; \
                up_sum##N##S += up_d##N * hrx_q4_k_swiglu_dot4_from_pack(up_packed_qs##N, high, b4_##S) - up_min##N * ysum_##S; \
            } while (0)
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(0, a);
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(0, b);
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(0, c);
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(0, d);
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(1, a);
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(1, b);
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(1, c);
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(1, d);
#undef HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC
        }

#define HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(S, ID, TOK) \
        do { \
            hrx_reduce_wg4_swiglu<64>(gate_sum0##S, up_sum0##S, gate_sum1##S, up_sum1##S, sumsh); \
            if (tid == 0) { \
                char * dst_base = reinterpret_cast<char *>(dst) + (ID) * c.dst_nb1 + (TOK) * c.dst_nb2; \
                const float silu_gate0 = gate_sum0##S / (1.0f + __expf(-gate_sum0##S)); \
                const float silu_gate1 = gate_sum1##S / (1.0f + __expf(-gate_sum1##S)); \
                *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0##S * silu_gate0; \
                *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1##S * silu_gate1; \
            } \
            __syncthreads(); \
        } while (0)
        HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(a, id_a, tok_a);
        if (has_b) {
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(b, id_b, tok_b);
        }
        if (has_c) {
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(c, id_c, tok_c);
        }
        if (has_d) {
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(d, id_d, tok_d);
        }
#undef HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_grouped_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long expert = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= c.rows || expert >= c.n_experts) {
        return;
    }

    const uint32_t count = counts[expert];
    if (count == 0) {
        return;
    }

    __shared__ float sumsh[8 * 4 * (64 / 32)];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;
    const uint32_t * expert_routes = routes + expert * c.route_capacity;

    for (uint32_t route_idx = 0; route_idx < count; route_idx += 8) {
        const uint32_t route_a = expert_routes[route_idx];
        const uint32_t route_b = route_idx + 1 < count ? expert_routes[route_idx + 1] : route_a;
        const uint32_t route_c = route_idx + 2 < count ? expert_routes[route_idx + 2] : route_a;
        const uint32_t route_d = route_idx + 3 < count ? expert_routes[route_idx + 3] : route_a;
        const uint32_t route_e = route_idx + 4 < count ? expert_routes[route_idx + 4] : route_a;
        const uint32_t route_f = route_idx + 5 < count ? expert_routes[route_idx + 5] : route_a;
        const uint32_t route_g = route_idx + 6 < count ? expert_routes[route_idx + 6] : route_a;
        const uint32_t route_h = route_idx + 7 < count ? expert_routes[route_idx + 7] : route_a;
        const bool has_b = route_idx + 1 < count;
        const bool has_c = route_idx + 2 < count;
        const bool has_d = route_idx + 3 < count;
        const bool has_e = route_idx + 4 < count;
        const bool has_f = route_idx + 5 < count;
        const bool has_g = route_idx + 6 < count;
        const bool has_h = route_idx + 7 < count;
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(S) \
        const long long id_##S = static_cast<long long>(route_##S & 7u); \
        const long long tok_##S = static_cast<long long>(route_##S >> 3); \
        const char * src1_##S = reinterpret_cast<const char *>(src1) + id_##S * c.src1_nb1 + tok_##S * c.src1_nb2
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(a);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(b);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(c);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(d);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(e);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(f);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(g);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(h);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL

        float gate_sum0a = 0.0f, up_sum0a = 0.0f, gate_sum1a = 0.0f, up_sum1a = 0.0f;
        float gate_sum0b = 0.0f, up_sum0b = 0.0f, gate_sum1b = 0.0f, up_sum1b = 0.0f;
        float gate_sum0c = 0.0f, up_sum0c = 0.0f, gate_sum1c = 0.0f, up_sum1c = 0.0f;
        float gate_sum0d = 0.0f, up_sum0d = 0.0f, gate_sum1d = 0.0f, up_sum1d = 0.0f;
        float gate_sum0e = 0.0f, up_sum0e = 0.0f, gate_sum1e = 0.0f, up_sum1e = 0.0f;
        float gate_sum0f = 0.0f, up_sum0f = 0.0f, gate_sum1f = 0.0f, up_sum1f = 0.0f;
        float gate_sum0g = 0.0f, up_sum0g = 0.0f, gate_sum1g = 0.0f, up_sum1g = 0.0f;
        float gate_sum0h = 0.0f, up_sum0h = 0.0f, gate_sum1h = 0.0f, up_sum1h = 0.0f;

        for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

            uint8_t gate_sc0 = 0, gate_m0 = 0, gate_sc1 = 0, gate_m1 = 0;
            uint8_t up_sc0 = 0, up_m0 = 0, up_sc1 = 0, up_m1 = 0;
            hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
            hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
            hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);

            const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
            const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
            const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
            const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
            const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
            const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
            const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
            const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
            const long long src_base = block_idx * 256 + group * 32 + lane;
            const int qs_base = (group >> 1) * 32 + lane;
            const uint32_t gate_packed_qs0 = *reinterpret_cast<const uint32_t *>(gate_block0->qs + qs_base);
            const uint32_t gate_packed_qs1 = *reinterpret_cast<const uint32_t *>(gate_block1->qs + qs_base);
            const uint32_t up_packed_qs0 = *reinterpret_cast<const uint32_t *>(up_block0->qs + qs_base);
            const uint32_t up_packed_qs1 = *reinterpret_cast<const uint32_t *>(up_block1->qs + qs_base);

#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(S, HAS) \
            float4 b4_##S; \
            if (HAS) { \
                b4_##S = *reinterpret_cast<const float4 *>(src1_##S + src_base * sizeof(float)); \
            } else { \
                b4_##S = { 0.0f, 0.0f, 0.0f, 0.0f }; \
            }
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(a, true);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(b, has_b);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(c, has_c);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(d, has_d);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(e, has_e);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(f, has_f);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(g, has_g);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(h, has_h);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4

            const bool high = (group & 1) != 0;
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM(S) \
            const float ysum_##S = hrx_q4_k_swiglu_sum_float4(b4_##S)
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM(a);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM(b);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM(c);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM(d);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM(e);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM(f);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM(g);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM(h);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_YSUM

#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(N, S) \
            do { \
                gate_sum##N##S += gate_d##N * hrx_q4_k_swiglu_dot4_from_pack(gate_packed_qs##N, high, b4_##S) - gate_min##N * ysum_##S; \
                up_sum##N##S += up_d##N * hrx_q4_k_swiglu_dot4_from_pack(up_packed_qs##N, high, b4_##S) - up_min##N * ysum_##S; \
            } while (0)
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(0, a);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(0, b);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(0, c);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(0, d);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(0, e);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(0, f);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(0, g);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(0, h);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(1, a);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(1, b);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(1, c);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(1, d);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(1, e);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(1, f);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(1, g);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(1, h);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC
        }

        const unsigned int lane = tid & (warpSize - 1);
        const unsigned int wave = tid / warpSize;
        constexpr int waves = 64 / 32;
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(S) \
        do { \
            gate_sum0##S += __shfl_down(gate_sum0##S, offset); \
            up_sum0##S += __shfl_down(up_sum0##S, offset); \
            gate_sum1##S += __shfl_down(gate_sum1##S, offset); \
            up_sum1##S += __shfl_down(up_sum1##S, offset); \
        } while (0)
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(a);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(b);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(c);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(d);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(e);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(f);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(g);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(h);
        }
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(S, R) \
        do { \
            sumsh[wave + ((R) * 4 + 0) * waves] = gate_sum0##S; \
            sumsh[wave + ((R) * 4 + 1) * waves] = up_sum0##S; \
            sumsh[wave + ((R) * 4 + 2) * waves] = gate_sum1##S; \
            sumsh[wave + ((R) * 4 + 3) * waves] = up_sum1##S; \
        } while (0)
        if (lane == 0) {
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(a, 0);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(b, 1);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(c, 2);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(d, 3);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(e, 4);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(f, 5);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(g, 6);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(h, 7);
        }
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE
        __syncthreads();

#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(S, R) \
        do { \
            gate_sum0##S = lane < waves ? sumsh[lane + ((R) * 4 + 0) * waves] : 0.0f; \
            up_sum0##S = lane < waves ? sumsh[lane + ((R) * 4 + 1) * waves] : 0.0f; \
            gate_sum1##S = lane < waves ? sumsh[lane + ((R) * 4 + 2) * waves] : 0.0f; \
            up_sum1##S = lane < waves ? sumsh[lane + ((R) * 4 + 3) * waves] : 0.0f; \
        } while (0)
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(a, 0);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(b, 1);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(c, 2);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(d, 3);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(e, 4);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(f, 5);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(g, 6);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(h, 7);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD
        if (wave == 0) {
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(S) \
            do { \
                gate_sum0##S += __shfl_down(gate_sum0##S, offset); \
                up_sum0##S += __shfl_down(up_sum0##S, offset); \
                gate_sum1##S += __shfl_down(gate_sum1##S, offset); \
                up_sum1##S += __shfl_down(up_sum1##S, offset); \
            } while (0)
            for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(a);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(b);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(c);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(d);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(e);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(f);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(g);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(h);
            }
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL
        }
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(S, ID, TOK) \
        do { \
            if (tid == 0) { \
                char * dst_base = reinterpret_cast<char *>(dst) + (ID) * c.dst_nb1 + (TOK) * c.dst_nb2; \
                const float silu_gate0 = gate_sum0##S / (1.0f + __expf(-gate_sum0##S)); \
                const float silu_gate1 = gate_sum1##S / (1.0f + __expf(-gate_sum1##S)); \
                *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0##S * silu_gate0; \
                *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1##S * silu_gate1; \
            } \
        } while (0)
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(a, id_a, tok_a);
        if (has_b) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(b, id_b, tok_b); }
        if (has_c) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(c, id_c, tok_c); }
        if (has_d) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(d, id_d, tok_d); }
        if (has_e) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(e, id_e, tok_e); }
        if (has_f) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(f, id_f, tok_f); }
        if (has_g) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(g, id_g, tok_g); }
        if (has_h) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(h, id_h, tok_h); }
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE
        __syncthreads();
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_id_q4_k_swiglu_packed_f32_impl(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float gate_sumsh[(WG_SIZE + 31) / 32];
    __shared__ float up_sumsh[(WG_SIZE + 31) / 32];

    const char * gate_row_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2 + row * c.gate_nb1;
    const char * up_row_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2 + row * c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    const long long blocks_per_row = c.k / 256;
    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    const int il = itid >> 2;
    const int ir = itid - 4 * il;
    const int v_im = il >> 1;
    const int v_in = il & 1;
    const int l0 = 4 * (2 * ir + v_in);
    const int q_offset = 32 * v_im + l0;
    const int y_offset = 64 * v_im + l0;
    constexpr int block_slots = WG_SIZE / 16;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_slots) {
        const hrx_block_q4_K_id_swiglu * gate_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        const float gate_d = __half2float(__ushort_as_half(gate_block->d));
        const float gate_dmin = __half2float(__ushort_as_half(gate_block->dmin));
        const float up_d = __half2float(__ushort_as_half(up_block->d));
        const float up_dmin = __half2float(__ushort_as_half(up_block->dmin));
        float gate_d0 = 0.0f;
        float gate_d1 = 0.0f;
        float gate_d2 = 0.0f;
        float gate_d3 = 0.0f;
        float gate_min0 = 0.0f;
        float gate_min1 = 0.0f;
        float gate_min2 = 0.0f;
        float gate_min3 = 0.0f;
        float up_d0 = 0.0f;
        float up_d1 = 0.0f;
        float up_d2 = 0.0f;
        float up_d3 = 0.0f;
        float up_min0 = 0.0f;
        float up_min1 = 0.0f;
        float up_min2 = 0.0f;
        float up_min3 = 0.0f;
        hrx_q4_k_swiglu_scale_min4(
            gate_block->scales, v_im, gate_d, gate_dmin,
            gate_d0, gate_d1, gate_d2, gate_d3, gate_min0, gate_min1, gate_min2, gate_min3);
        hrx_q4_k_swiglu_scale_min4(
            up_block->scales, v_im, up_d, up_dmin,
            up_d0, up_d1, up_d2, up_d3, up_min0, up_min1, up_min2, up_min3);

        const long long src_base = block_idx * 256 + y_offset;
        const unsigned long long gate_q01 = static_cast<unsigned long long>(
            *reinterpret_cast<const uint32_t *>(gate_block->qs + q_offset));
        const unsigned long long gate_q23 = static_cast<unsigned long long>(
            *reinterpret_cast<const uint32_t *>(gate_block->qs + q_offset + 64));
        const unsigned long long up_q01 = static_cast<unsigned long long>(
            *reinterpret_cast<const uint32_t *>(up_block->qs + q_offset));
        const unsigned long long up_q23 = static_cast<unsigned long long>(
            *reinterpret_cast<const uint32_t *>(up_block->qs + q_offset + 64));
        const float4 y0 = *reinterpret_cast<const float4 *>(src1_col + src_base * sizeof(float));
        const float4 y1 = *reinterpret_cast<const float4 *>(src1_col + (src_base + 32) * sizeof(float));
        const float4 y2 = *reinterpret_cast<const float4 *>(src1_col + (src_base + 128) * sizeof(float));
        const float4 y3 = *reinterpret_cast<const float4 *>(src1_col + (src_base + 160) * sizeof(float));

        const float ysum0 = hrx_q4_k_swiglu_sum_float4(y0);
        const float ysum1 = hrx_q4_k_swiglu_sum_float4(y1);
        const float ysum2 = hrx_q4_k_swiglu_sum_float4(y2);
        const float ysum3 = hrx_q4_k_swiglu_sum_float4(y3);

        gate_sum += gate_d0 * hrx_q4_k_swiglu_dot4_from_pack(gate_q01, false, y0) - gate_min0 * ysum0;
        gate_sum += gate_d1 * hrx_q4_k_swiglu_dot4_from_pack(gate_q01, true, y1) - gate_min1 * ysum1;
        gate_sum += gate_d2 * hrx_q4_k_swiglu_dot4_from_pack(gate_q23, false, y2) - gate_min2 * ysum2;
        gate_sum += gate_d3 * hrx_q4_k_swiglu_dot4_from_pack(gate_q23, true, y3) - gate_min3 * ysum3;

        up_sum += up_d0 * hrx_q4_k_swiglu_dot4_from_pack(up_q01, false, y0) - up_min0 * ysum0;
        up_sum += up_d1 * hrx_q4_k_swiglu_dot4_from_pack(up_q01, true, y1) - up_min1 * ysum1;
        up_sum += up_d2 * hrx_q4_k_swiglu_dot4_from_pack(up_q23, false, y2) - up_min2 * ysum2;
        up_sum += up_d3 * hrx_q4_k_swiglu_dot4_from_pack(up_q23, true, y3) - up_min3 * ysum3;
    }

    gate_sum = hrx_reduce_wg_swiglu<WG_SIZE>(gate_sum, gate_sumsh);
    up_sum = hrx_reduce_wg_swiglu<WG_SIZE>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            up_sum * silu_gate;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_packed_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    hrx_mul_mat_id_q4_k_swiglu_packed_f32_impl<64>(gate, up, src1, ids, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_packed_wg32_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    hrx_mul_mat_id_q4_k_swiglu_packed_f32_impl<32>(gate, up, src1, ids, dst, c);
}

static __global__ void hrx_mul_mat_id_q4_k_swiglu_rows2_x16_wg32_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const long long row = row0 + static_cast<long long>(tid >> 4);
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    if (row >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    const unsigned int lane = tid & 15;
    const char * gate_row_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2 + row * c.gate_nb1;
    const char * up_row_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2 + row * c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    #pragma unroll
    for (int iter = 0; iter < 16; ++iter) {
        const long long block_idx = iter >> 1;
        const int group = ((iter & 1) << 2) + static_cast<int>(lane >> 2);
        const int group_offset = static_cast<int>(lane & 3) << 3;
        const long long col = static_cast<long long>(iter) * 128 + static_cast<long long>(lane) * 8;
        const hrx_block_q4_K_id_swiglu * gate_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        uint8_t gate_sc = 0;
        uint8_t gate_m = 0;
        uint8_t up_sc = 0;
        uint8_t up_m = 0;
        hrx_get_scale_min_k4_id_swiglu(group, gate_block->scales, &gate_sc, &gate_m);
        hrx_get_scale_min_k4_id_swiglu(group, up_block->scales, &up_sc, &up_m);
        const float gate_d = __half2float(__ushort_as_half(gate_block->d)) * static_cast<float>(gate_sc);
        const float gate_min = __half2float(__ushort_as_half(gate_block->dmin)) * static_cast<float>(gate_m);
        const float up_d = __half2float(__ushort_as_half(up_block->d)) * static_cast<float>(up_sc);
        const float up_min = __half2float(__ushort_as_half(up_block->dmin)) * static_cast<float>(up_m);
        const int qs_base = (group >> 1) * 32 + group_offset;

        const bool high = (group & 1) != 0;
        const unsigned long long gate_qpack = *reinterpret_cast<const unsigned long long *>(gate_block->qs + qs_base);
        const unsigned long long up_qpack = *reinterpret_cast<const unsigned long long *>(up_block->qs + qs_base);
        const float4 b0 = *reinterpret_cast<const float4 *>(src1_col + col * sizeof(float));
        const float4 b1 = *reinterpret_cast<const float4 *>(src1_col + (col + 4) * sizeof(float));

        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<0>(gate_qpack, high) - gate_min) * b0.x;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<1>(gate_qpack, high) - gate_min) * b0.y;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<2>(gate_qpack, high) - gate_min) * b0.z;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<3>(gate_qpack, high) - gate_min) * b0.w;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<4>(gate_qpack, high) - gate_min) * b1.x;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<5>(gate_qpack, high) - gate_min) * b1.y;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<6>(gate_qpack, high) - gate_min) * b1.z;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<7>(gate_qpack, high) - gate_min) * b1.w;

        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<0>(up_qpack, high) - up_min) * b0.x;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<1>(up_qpack, high) - up_min) * b0.y;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<2>(up_qpack, high) - up_min) * b0.z;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<3>(up_qpack, high) - up_min) * b0.w;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<4>(up_qpack, high) - up_min) * b1.x;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<5>(up_qpack, high) - up_min) * b1.y;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<6>(up_qpack, high) - up_min) * b1.z;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<7>(up_qpack, high) - up_min) * b1.w;
    }

    for (int offset = 8; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down(gate_sum, offset, 16);
        up_sum += __shfl_down(up_sum, offset, 16);
    }

    if (lane == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            up_sum * silu_gate;
    }
}
