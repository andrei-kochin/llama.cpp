#include <hip/hip_runtime.h>

#include "topk_moe_f32_common.hip.inc"

static __device__ __forceinline__ int hrx_topk_top8_shfl_xor_i32(int value, int mask) {
    return __shfl_xor(value, mask, 32);
}

static __device__ __forceinline__ float hrx_topk_top8_shfl_xor_f32(float value, int mask) {
    return __shfl_xor(value, mask, 32);
}

template<int n_experts>
static __device__ __forceinline__ void hrx_topk_moe_f32_wave32_top8_norm_impl(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    constexpr int wave_size = 32;
    constexpr int experts_per_lane = n_experts / wave_size;
    constexpr int n_expert_used = 8;

    float local[experts_per_lane];
    float output_logit = -FLT_MAX;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int row_in_group = static_cast<int>(__builtin_amdgcn_workitem_id_y());
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) *
        static_cast<long long>(__builtin_amdgcn_workgroup_size_y()) + row_in_group;
    if (row >= c.n_rows) {
        return;
    }

    const float * logits_row = reinterpret_cast<const float *>(
        reinterpret_cast<const char *>(logits) + row * c.logits_nb1);
    char * weights_row = reinterpret_cast<char *>(weights) + row * c.weights_nb1;
    char * ids_row = reinterpret_cast<char *>(ids) + row * c.ids_nb1;

    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        local[i] = logits_row[lane + i * wave_size];
    }

    float selected_max = -FLT_MAX;
    #pragma unroll
    for (int k = 0; k < n_expert_used; ++k) {
        float best_v = local[0];
        int best_i = lane;
        #pragma unroll
        for (int i = 1; i < experts_per_lane; ++i) {
            const int expert = lane + i * wave_size;
            if (hrx_topk_take_rhs(best_v, best_i, local[i], expert)) {
                best_v = local[i];
                best_i = expert;
            }
        }

        #pragma unroll
        for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
            const float rhs_v = hrx_topk_top8_shfl_xor_f32(best_v, mask);
            const int rhs_i = hrx_topk_top8_shfl_xor_i32(best_i, mask);
            if (hrx_topk_take_rhs(best_v, best_i, rhs_v, rhs_i)) {
                best_v = rhs_v;
                best_i = rhs_i;
            }
        }

        selected_max = fmaxf(selected_max, best_v);
        if (k == lane) {
            output_logit = best_v;
        }
        if ((best_i & (wave_size - 1)) == lane) {
            local[best_i / wave_size] = -FLT_MAX;
            *reinterpret_cast<int *>(ids_row + k * c.ids_nb_k) = best_i;
        }
    }

    const float output_exp = lane < n_expert_used ? __expf(output_logit - selected_max) : 0.0f;
    float selected_sum = output_exp;
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        selected_sum += hrx_topk_top8_shfl_xor_f32(selected_sum, mask);
    }

    if (lane < n_expert_used) {
        *reinterpret_cast<float *>(weights_row + lane * c.weights_nb_k) = output_exp / selected_sum;
    }
}

extern "C" __global__ void hrx_topk_moe_f32_wave32_n32_top8_norm(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    hrx_topk_moe_f32_wave32_top8_norm_impl<32>(logits, weights, ids, c);
}

extern "C" __global__ void hrx_topk_moe_f32_wave32_n64_top8_norm(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    hrx_topk_moe_f32_wave32_top8_norm_impl<64>(logits, weights, ids, c);
}

extern "C" __global__ void hrx_topk_moe_f32_wave32_n128_top8_norm(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    hrx_topk_moe_f32_wave32_top8_norm_impl<128>(logits, weights, ids, c);
}

extern "C" __global__ void hrx_topk_moe_f32_wave32_n256_top8_norm(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    constexpr int wave_size = 32;
    constexpr int experts_per_lane = 8;
    constexpr int n_expert_used = 8;

    float local[experts_per_lane];
    float output_logit = -FLT_MAX;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int row_in_group = static_cast<int>(__builtin_amdgcn_workitem_id_y());
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) *
        static_cast<long long>(__builtin_amdgcn_workgroup_size_y()) + row_in_group;
    if (row >= c.n_rows) {
        return;
    }

    const float * logits_row = reinterpret_cast<const float *>(
        reinterpret_cast<const char *>(logits) + row * c.logits_nb1);
    char * weights_row = reinterpret_cast<char *>(weights) + row * c.weights_nb1;
    char * ids_row = reinterpret_cast<char *>(ids) + row * c.ids_nb1;

    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        local[i] = logits_row[lane + i * wave_size];
    }

    float selected_max = -FLT_MAX;
    #pragma unroll
    for (int k = 0; k < n_expert_used; ++k) {
        float best_v = local[0];
        int best_i = lane;
        #pragma unroll
        for (int i = 1; i < experts_per_lane; ++i) {
            const int expert = lane + i * wave_size;
            if (hrx_topk_take_rhs(best_v, best_i, local[i], expert)) {
                best_v = local[i];
                best_i = expert;
            }
        }

        #pragma unroll
        for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
            const float rhs_v = __shfl_xor(best_v, mask, wave_size);
            const int rhs_i = __shfl_xor(best_i, mask, wave_size);
            if (hrx_topk_take_rhs(best_v, best_i, rhs_v, rhs_i)) {
                best_v = rhs_v;
                best_i = rhs_i;
            }
        }

        selected_max = fmaxf(selected_max, best_v);
        if (k == lane) {
            output_logit = best_v;
        }
        if ((best_i & (wave_size - 1)) == lane) {
            local[best_i / wave_size] = -FLT_MAX;
            *reinterpret_cast<int *>(ids_row + k * c.ids_nb_k) = best_i;
        }
    }

    const float output_exp = lane < n_expert_used ? __expf(output_logit - selected_max) : 0.0f;
    float selected_sum = output_exp;
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        selected_sum += __shfl_xor(selected_sum, mask, wave_size);
    }

    if (lane < n_expert_used) {
        *reinterpret_cast<float *>(weights_row + lane * c.weights_nb_k) = output_exp / selected_sum;
    }
}
