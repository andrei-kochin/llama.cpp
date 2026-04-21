#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "../../kernels/topk_moe_f32_shared.hip.cpp"

#define HRX_TOPK_MOE_WAVE_KERNEL hrx_topk_moe_f32_wave32
#define HRX_TOPK_MOE_WAVE_SIZE 32
#include "../../kernels/topk_moe_f32_wave.inc"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

static __device__ __forceinline__ int probe_shfl_xor_i32(int value, int lane, int mask) {
    return __builtin_amdgcn_ds_bpermute((lane ^ mask) << 2, value);
}

static __device__ __forceinline__ float probe_shfl_xor_f32(float value, int lane, int mask) {
    union {
        int i;
        float f;
    } tmp;
    tmp.f = value;
    tmp.i = probe_shfl_xor_i32(tmp.i, lane, mask);
    return tmp.f;
}

static __device__ __forceinline__ bool probe_take_rhs(float lhs_v, int lhs_i, float rhs_v, int rhs_i) {
    return rhs_v > lhs_v || (rhs_v == lhs_v && rhs_i < lhs_i);
}

extern "C" __global__ void probe_topk_empty(
        const float *, float *, int *, hrx_topk_moe_f32_constants) {
}

extern "C" __global__ void probe_topk_moe_f32_wave32_256_top8_norm(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    constexpr int wave_size = 32;
    constexpr int experts_per_lane = 8;
    constexpr int n_expert_used = 8;

    float probs[experts_per_lane];
    float selection_probs[experts_per_lane];
    float output_weight = 0.0f;

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
        probs[i] = logits_row[lane + i * wave_size];
    }

    float max_val = probs[0];
    #pragma unroll
    for (int i = 1; i < experts_per_lane; ++i) {
        max_val = fmaxf(max_val, probs[i]);
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        max_val = fmaxf(max_val, probe_shfl_xor_f32(max_val, lane, mask));
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        probs[i] = __expf(probs[i] - max_val);
        sum += probs[i];
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        sum += probe_shfl_xor_f32(sum, lane, mask);
    }

    const float inv_sum = 1.0f / sum;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        probs[i] *= inv_sum;
        selection_probs[i] = probs[i];
    }

    float selected_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < n_expert_used; ++k) {
        float best_v = probs[0];
        float best_s = selection_probs[0];
        int best_i = lane;
        #pragma unroll
        for (int i = 1; i < experts_per_lane; ++i) {
            const int expert = lane + i * wave_size;
            if (probe_take_rhs(best_s, best_i, selection_probs[i], expert)) {
                best_v = probs[i];
                best_s = selection_probs[i];
                best_i = expert;
            }
        }

        #pragma unroll
        for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
            const float rhs_v = probe_shfl_xor_f32(best_v, lane, mask);
            const float rhs_s = probe_shfl_xor_f32(best_s, lane, mask);
            const int rhs_i = probe_shfl_xor_i32(best_i, lane, mask);
            if (probe_take_rhs(best_s, best_i, rhs_s, rhs_i)) {
                best_v = rhs_v;
                best_s = rhs_s;
                best_i = rhs_i;
            }
        }

        if (k == lane) {
            output_weight = best_v;
        }
        if ((best_i & (wave_size - 1)) == lane) {
            selection_probs[best_i / wave_size] = -FLT_MAX;
            *reinterpret_cast<int *>(ids_row + k * c.ids_nb_k) = best_i;
            selected_sum += best_v;
        }
    }

    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        selected_sum += probe_shfl_xor_f32(selected_sum, lane, mask);
    }
    const float denom = selected_sum < c.clamp_min ? c.clamp_min :
        (selected_sum > c.clamp_max ? c.clamp_max : selected_sum);

    if (lane < n_expert_used) {
        *reinterpret_cast<float *>(weights_row + lane * c.weights_nb_k) = output_weight / denom;
    }
}

extern "C" __global__ void probe_topk_moe_f32_wave32_256_top8_norm_hipshfl(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    constexpr int wave_size = 32;
    constexpr int experts_per_lane = 8;
    constexpr int n_expert_used = 8;

    float probs[experts_per_lane];
    float selection_probs[experts_per_lane];
    float output_weight = 0.0f;

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
        probs[i] = logits_row[lane + i * wave_size];
    }

    float max_val = probs[0];
    #pragma unroll
    for (int i = 1; i < experts_per_lane; ++i) {
        max_val = fmaxf(max_val, probs[i]);
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor(max_val, mask, wave_size));
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        probs[i] = __expf(probs[i] - max_val);
        sum += probs[i];
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        sum += __shfl_xor(sum, mask, wave_size);
    }

    const float inv_sum = 1.0f / sum;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        probs[i] *= inv_sum;
        selection_probs[i] = probs[i];
    }

    float selected_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < n_expert_used; ++k) {
        float best_v = probs[0];
        float best_s = selection_probs[0];
        int best_i = lane;
        #pragma unroll
        for (int i = 1; i < experts_per_lane; ++i) {
            const int expert = lane + i * wave_size;
            if (probe_take_rhs(best_s, best_i, selection_probs[i], expert)) {
                best_v = probs[i];
                best_s = selection_probs[i];
                best_i = expert;
            }
        }

        #pragma unroll
        for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
            const float rhs_v = __shfl_xor(best_v, mask, wave_size);
            const float rhs_s = __shfl_xor(best_s, mask, wave_size);
            const int rhs_i = __shfl_xor(best_i, mask, wave_size);
            if (probe_take_rhs(best_s, best_i, rhs_s, rhs_i)) {
                best_v = rhs_v;
                best_s = rhs_s;
                best_i = rhs_i;
            }
        }

        if (k == lane) {
            output_weight = best_v;
        }
        if ((best_i & (wave_size - 1)) == lane) {
            selection_probs[best_i / wave_size] = -FLT_MAX;
            *reinterpret_cast<int *>(ids_row + k * c.ids_nb_k) = best_i;
            selected_sum += best_v;
        }
    }

    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        selected_sum += __shfl_xor(selected_sum, mask, wave_size);
    }
    const float denom = selected_sum < c.clamp_min ? c.clamp_min :
        (selected_sum > c.clamp_max ? c.clamp_max : selected_sum);

    if (lane < n_expert_used) {
        *reinterpret_cast<float *>(weights_row + lane * c.weights_nb_k) = output_weight / denom;
    }
}

extern "C" __global__ void probe_topk_moe_f32_wave32_256_top8_norm_onearray(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    constexpr int wave_size = 32;
    constexpr int experts_per_lane = 8;
    constexpr int n_expert_used = 8;

    float probs[experts_per_lane];
    float output_weight = 0.0f;

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
        probs[i] = logits_row[lane + i * wave_size];
    }

    float max_val = probs[0];
    #pragma unroll
    for (int i = 1; i < experts_per_lane; ++i) {
        max_val = fmaxf(max_val, probs[i]);
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor(max_val, mask, wave_size));
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        probs[i] = __expf(probs[i] - max_val);
        sum += probs[i];
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        sum += __shfl_xor(sum, mask, wave_size);
    }

    const float inv_sum = 1.0f / sum;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        probs[i] *= inv_sum;
    }

    float selected_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < n_expert_used; ++k) {
        float best_v = probs[0];
        int best_i = lane;
        #pragma unroll
        for (int i = 1; i < experts_per_lane; ++i) {
            const int expert = lane + i * wave_size;
            if (probe_take_rhs(best_v, best_i, probs[i], expert)) {
                best_v = probs[i];
                best_i = expert;
            }
        }

        #pragma unroll
        for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
            const float rhs_v = __shfl_xor(best_v, mask, wave_size);
            const int rhs_i = __shfl_xor(best_i, mask, wave_size);
            if (probe_take_rhs(best_v, best_i, rhs_v, rhs_i)) {
                best_v = rhs_v;
                best_i = rhs_i;
            }
        }

        if (k == lane) {
            output_weight = best_v;
        }
        if ((best_i & (wave_size - 1)) == lane) {
            probs[best_i / wave_size] = -FLT_MAX;
            *reinterpret_cast<int *>(ids_row + k * c.ids_nb_k) = best_i;
            selected_sum += best_v;
        }
    }

    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        selected_sum += __shfl_xor(selected_sum, mask, wave_size);
    }
    const float denom = selected_sum < c.clamp_min ? c.clamp_min :
        (selected_sum > c.clamp_max ? c.clamp_max : selected_sum);

    if (lane < n_expert_used) {
        *reinterpret_cast<float *>(weights_row + lane * c.weights_nb_k) = output_weight / denom;
    }
}

extern "C" __global__ void probe_topk_moe_f32_wave32_256_top8_norm_logits(
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
            if (probe_take_rhs(best_v, best_i, local[i], expert)) {
                best_v = local[i];
                best_i = expert;
            }
        }

        #pragma unroll
        for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
            const float rhs_v = __shfl_xor(best_v, mask, wave_size);
            const int rhs_i = __shfl_xor(best_i, mask, wave_size);
            if (probe_take_rhs(best_v, best_i, rhs_v, rhs_i)) {
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

extern "C" __global__ void probe_topk_moe_f32_wave64_256_top8_norm_hipshfl(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    constexpr int wave_size = 64;
    constexpr int experts_per_lane = 4;
    constexpr int n_expert_used = 8;

    float probs[experts_per_lane];
    float selection_probs[experts_per_lane];
    float output_weight = 0.0f;

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
        probs[i] = logits_row[lane + i * wave_size];
    }

    float max_val = probs[0];
    #pragma unroll
    for (int i = 1; i < experts_per_lane; ++i) {
        max_val = fmaxf(max_val, probs[i]);
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor(max_val, mask, wave_size));
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        probs[i] = __expf(probs[i] - max_val);
        sum += probs[i];
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        sum += __shfl_xor(sum, mask, wave_size);
    }

    const float inv_sum = 1.0f / sum;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        probs[i] *= inv_sum;
        selection_probs[i] = probs[i];
    }

    float selected_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < n_expert_used; ++k) {
        float best_v = probs[0];
        float best_s = selection_probs[0];
        int best_i = lane;
        #pragma unroll
        for (int i = 1; i < experts_per_lane; ++i) {
            const int expert = lane + i * wave_size;
            if (probe_take_rhs(best_s, best_i, selection_probs[i], expert)) {
                best_v = probs[i];
                best_s = selection_probs[i];
                best_i = expert;
            }
        }

        #pragma unroll
        for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
            const float rhs_v = __shfl_xor(best_v, mask, wave_size);
            const float rhs_s = __shfl_xor(best_s, mask, wave_size);
            const int rhs_i = __shfl_xor(best_i, mask, wave_size);
            if (probe_take_rhs(best_s, best_i, rhs_s, rhs_i)) {
                best_v = rhs_v;
                best_s = rhs_s;
                best_i = rhs_i;
            }
        }

        if (k == lane) {
            output_weight = best_v;
        }
        if ((best_i & (wave_size - 1)) == lane) {
            selection_probs[best_i / wave_size] = -FLT_MAX;
            *reinterpret_cast<int *>(ids_row + k * c.ids_nb_k) = best_i;
            selected_sum += best_v;
        }
    }

    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        selected_sum += __shfl_xor(selected_sum, mask, wave_size);
    }
    const float denom = selected_sum < c.clamp_min ? c.clamp_min :
        (selected_sum > c.clamp_max ? c.clamp_max : selected_sum);

    if (lane < n_expert_used) {
        *reinterpret_cast<float *>(weights_row + lane * c.weights_nb_k) = output_weight / denom;
    }
}

using kernel_fn_t = void (*)(const float *, float *, int *, hrx_topk_moe_f32_constants);

struct variant {
    const char * name;
    kernel_fn_t fn;
    dim3 block;
};

static void cpu_reference(
        const std::vector<float> & logits,
        int n_rows,
        int n_experts,
        int top_k,
        std::vector<float> & weights,
        std::vector<int> & ids,
        int ids_stride) {
    weights.assign(static_cast<size_t>(n_rows * top_k), 0.0f);
    ids.assign(static_cast<size_t>(n_rows * ids_stride), -1);
    for (int row = 0; row < n_rows; ++row) {
        const float * row_logits = logits.data() + static_cast<size_t>(row) * n_experts;
        float max_value = row_logits[0];
        for (int expert = 1; expert < n_experts; ++expert) {
            max_value = std::max(max_value, row_logits[expert]);
        }
        std::vector<float> probs(static_cast<size_t>(n_experts));
        float sum = 0.0f;
        for (int expert = 0; expert < n_experts; ++expert) {
            probs[static_cast<size_t>(expert)] = std::exp(row_logits[expert] - max_value);
            sum += probs[static_cast<size_t>(expert)];
        }
        for (float & value : probs) {
            value /= sum;
        }
        std::vector<int> order(static_cast<size_t>(n_experts));
        for (int expert = 0; expert < n_experts; ++expert) {
            order[static_cast<size_t>(expert)] = expert;
        }
        std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
            const float lv = probs[static_cast<size_t>(lhs)];
            const float rv = probs[static_cast<size_t>(rhs)];
            if (lv != rv) {
                return lv > rv;
            }
            return lhs < rhs;
        });
        float selected_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            selected_sum += probs[static_cast<size_t>(order[static_cast<size_t>(k)])];
        }
        selected_sum = std::max(selected_sum, 6.103515625e-5f);
        for (int k = 0; k < top_k; ++k) {
            const int expert = order[static_cast<size_t>(k)];
            ids[static_cast<size_t>(row) * ids_stride + k] = expert;
            weights[static_cast<size_t>(row) * top_k + k] =
                probs[static_cast<size_t>(expert)] / selected_sum;
        }
    }
}

static bool check_result(
        const char * name,
        const std::vector<float> & got_weights,
        const std::vector<int> & got_ids,
        const std::vector<float> & ref_weights,
        const std::vector<int> & ref_ids,
        int n_rows,
        int top_k,
        int ids_stride) {
    for (int row = 0; row < n_rows; ++row) {
        for (int k = 0; k < top_k; ++k) {
            const size_t compact = static_cast<size_t>(row) * top_k + k;
            const size_t id_idx = static_cast<size_t>(row) * ids_stride + k;
            if (got_ids[id_idx] != ref_ids[id_idx]) {
                std::fprintf(stderr, "%s id mismatch row=%d k=%d got=%d ref=%d\n",
                    name, row, k, got_ids[id_idx], ref_ids[id_idx]);
                return false;
            }
            const float diff = std::fabs(got_weights[compact] - ref_weights[compact]);
            if (diff > 2.0e-5f) {
                std::fprintf(stderr, "%s weight mismatch row=%d k=%d got=%g ref=%g diff=%g\n",
                    name, row, k, got_weights[compact], ref_weights[compact], diff);
                return false;
            }
        }
    }
    return true;
}

static float bench_variant(
        const variant & v,
        const float * d_logits,
        float * d_weights,
        int * d_ids,
        const hrx_topk_moe_f32_constants & c,
        int n_rows,
        int iters) {
    dim3 grid((n_rows + v.block.y - 1) / v.block.y, 1, 1);
    for (int i = 0; i < 20; ++i) {
        hipLaunchKernelGGL(v.fn, grid, v.block, 0, 0, d_logits, d_weights, d_ids, c);
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        hipLaunchKernelGGL(v.fn, grid, v.block, 0, 0, d_logits, d_weights, d_ids, c);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms * 1000.0f / static_cast<float>(iters);
}

int main(int argc, char ** argv) {
    int n_rows = 1;
    int n_experts = 256;
    int top_k = 8;
    int iters = 20000;
    bool include_wave64 = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_int = [&](int & value) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(2);
            }
            value = std::atoi(argv[++i]);
        };
        if (arg == "--rows") {
            next_int(n_rows);
        } else if (arg == "--experts") {
            next_int(n_experts);
        } else if (arg == "--topk") {
            next_int(top_k);
        } else if (arg == "--iters") {
            next_int(iters);
        } else if (arg == "--include-wave64") {
            include_wave64 = true;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return 2;
        }
    }

    const int ids_stride = n_experts;
    std::vector<float> logits(static_cast<size_t>(n_rows * n_experts));
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (int row = 0; row < n_rows; ++row) {
        for (int expert = 0; expert < n_experts; ++expert) {
            float value = dist(rng);
            if ((expert % 47) == (row % 47)) {
                value += 1.0f;
            }
            logits[static_cast<size_t>(row) * n_experts + expert] = value;
        }
    }

    std::vector<float> ref_weights;
    std::vector<int> ref_ids;
    cpu_reference(logits, n_rows, n_experts, top_k, ref_weights, ref_ids, ids_stride);

    float * d_logits = nullptr;
    float * d_weights = nullptr;
    int * d_ids = nullptr;
    HIP_CHECK(hipMalloc(&d_logits, logits.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weights, static_cast<size_t>(n_rows * top_k) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ids, static_cast<size_t>(n_rows * ids_stride) * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), hipMemcpyHostToDevice));

    hrx_topk_moe_f32_constants c = {};
    c.n_experts = n_experts;
    c.n_rows = n_rows;
    c.n_expert_used = top_k;
    c.logits_nb1 = static_cast<long long>(n_experts * sizeof(float));
    c.weights_nb1 = static_cast<long long>(top_k * sizeof(float));
    c.weights_nb_k = static_cast<long long>(sizeof(float));
    c.ids_nb1 = static_cast<long long>(ids_stride * sizeof(int));
    c.ids_nb_k = static_cast<long long>(sizeof(int));
    c.scale = 1.0f;
    c.clamp_min = 6.103515625e-5f;
    c.clamp_max = std::numeric_limits<float>::infinity();
    c.with_norm = 1;

    std::vector<variant> variants = {
        {"empty", probe_topk_empty, dim3(32, 4, 1)},
        {"baseline64", hrx_topk_moe_f32, dim3(64, 1, 1)},
        {"shared4", hrx_topk_moe_f32_shared4, dim3(64, 4, 1)},
        {"shared8", hrx_topk_moe_f32_shared8, dim3(64, 8, 1)},
        {"wave32", hrx_topk_moe_f32_wave32, dim3(32, 4, 1)},
    };
    if (n_experts == 256 && top_k == 8) {
        variants.push_back({"wave32_256_top8_norm", probe_topk_moe_f32_wave32_256_top8_norm, dim3(32, 4, 1)});
        variants.push_back({"wave32_256_top8_norm_hipshfl", probe_topk_moe_f32_wave32_256_top8_norm_hipshfl, dim3(32, 4, 1)});
        variants.push_back({"wave32_256_top8_norm_onearray", probe_topk_moe_f32_wave32_256_top8_norm_onearray, dim3(32, 4, 1)});
        variants.push_back({"wave32_256_top8_norm_logits", probe_topk_moe_f32_wave32_256_top8_norm_logits, dim3(32, 4, 1)});
        if (include_wave64) {
            variants.push_back({"wave64_256_top8_norm_hipshfl", probe_topk_moe_f32_wave64_256_top8_norm_hipshfl, dim3(64, 1, 1)});
        }
    }

    std::vector<float> got_weights(static_cast<size_t>(n_rows * top_k));
    std::vector<int> got_ids(static_cast<size_t>(n_rows * ids_stride));
    std::printf("rows=%d experts=%d topk=%d ids_stride=%d iters=%d\n",
        n_rows, n_experts, top_k, ids_stride, iters);
    for (const variant & v : variants) {
        HIP_CHECK(hipMemset(d_weights, 0, static_cast<size_t>(n_rows * top_k) * sizeof(float)));
        HIP_CHECK(hipMemset(d_ids, 0xff, static_cast<size_t>(n_rows * ids_stride) * sizeof(int)));
        dim3 grid((n_rows + v.block.y - 1) / v.block.y, 1, 1);
        hipLaunchKernelGGL(v.fn, grid, v.block, 0, 0, d_logits, d_weights, d_ids, c);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(got_weights.data(), d_weights,
            got_weights.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(got_ids.data(), d_ids,
            got_ids.size() * sizeof(int), hipMemcpyDeviceToHost));
        const bool ok = check_result(v.name, got_weights, got_ids, ref_weights, ref_ids,
            n_rows, top_k, ids_stride);
        float us = bench_variant(v, d_logits, d_weights, d_ids, c, n_rows, iters);
        std::printf("  %-30s %s %8.3f us\n", v.name, ok ? "OK" : "FAIL", us);
    }

    HIP_CHECK(hipFree(d_logits));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_ids));
    return 0;
}
