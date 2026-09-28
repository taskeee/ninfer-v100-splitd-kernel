// ninfer::ops::detail - experimental sm_70 D256 Split-D prefill attention route.
//
// Why this file exists: measurement on this machine (2026-09-28, 84,894-token prompt)
// put the vendored llama.cpp MMA flash kernel at 47.4 s of a 117 s prefill -- 41% of the
// wall clock at 31 TFLOP/s with only 2 CTAs/SM resident (8 warps of 64). Tuning that
// kernel's config table did not help (ncols 64: -18%; Q_in_reg: -5.5x), so the route is
// replaced instead of tuned.
//
// The replacement is the 1CatAI D256 Split-D kernel, vendored from fishlikeX/sm70-attn
// (ggml/src/ggml-cuda/fattn-sm70-d256-kernel.cuh, unmodified device code as carried
// there), which was written for exactly this shape: sm_70, head_dim 256, causal, prefill.
// Same machine, same-class A/B in llama.cpp: 176k prefill 372.94 -> 521.93 tok/s (+39.9%).
//
// Interface fit (why the staging is this small):
//   * the kernel builds its own causal mask from (kv_len, kv_offset), so the route no
//     longer needs the [rows x n_kv] mask build;
//   * it wants K/V as contiguous f16 with row stride kv_heads*D and head stride D, which
//     is exactly the layout volta_flash_gather_kv_i8_kernel already produces;
//   * it wants Q as f16 [row][head][D] padded to a multiple of kBlockM (64) -- the only
//     new staging on our side, because our Q arrives as bf16 [D][heads][rows] and, on the
//     int8 KV path, has to cross the same normalized Hadamard the K codes did;
//   * ElementOut is float, matching the f32 staging this route already converts from.
//
// Switch: NINFER_VOLTA_SPLITD=1 selects this kernel. Default off: the vendored llama.cpp
// kernel stays the production path until an A/B on a >=100k prompt says otherwise.

#include "fattn-sm70-d256-kernel.cuh" // vendored; defines FLASH_NAMESPACE, cutlass::half_t

#include "core/device.h"
#include "core/tensor.h"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim     = 256;
constexpr int kD256BlockM  = 64;   // FLASH_NAMESPACE::Sm70D256SplitDTraits::kBlockM
constexpr int kD256Threads = 256;  // ...::kNThreads
// The route's Q-block is kVoltaFlashQBlockTokens (1024) wide, so the padded row count
// never exceeds this; staging is sized once for the maximum.
constexpr int kMaxRows = 1024;

constexpr int round_up_rows(int rows) { return ((rows + kD256BlockM - 1) / kD256BlockM) * kD256BlockM; }

// Upstream's SplitKV3 activation threshold (LLAMA_SM70_SPLITKV3_MIN_KV default). Overridable
// here so the attention contract tests can be pointed at the merge path: their cases are far
// below 2048 keys, so with the production threshold the merge kernel would never be exercised.
constexpr int kSplitKv3MinKv = 2048;

int split3_min_kv() {
    static const int value = [] {
        const char* env = std::getenv("NINFER_SM70_SPLITKV3_MIN_KV");
        return env != nullptr ? std::atoi(env) : kSplitKv3MinKv;
    }();
    return value;
}

// bf16 Q [D][heads][rows] -> f16 [row][head][D] padded to q_pad, zero-filled past `rows`.
// With Int8 the row crosses the registered normalized D256 Hadamard, matching how the K
// codes were rotated at append time; the dot product is invariant under that rotation.
template <bool Int8>
__launch_bounds__(256) __global__ void splitd_stage_q_kernel(
        const __nv_bfloat16* __restrict__ q, cutlass::half_t* __restrict__ qs, int rows,
        int q_pad, int heads) {
    constexpr int kWarps = 8;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    auto* dst_base = reinterpret_cast<half*>(qs);

    // Padding rows are read by the kernel (query_len == q_pad) and must be defined.
    for (int unit = blockIdx.x * kWarps + warp; unit < q_pad * heads;
         unit += gridDim.x * kWarps) {
        const int row = unit / heads;
        const int h   = unit - row * heads;
        if (row < rows) { continue; }
        half* dst = dst_base + (static_cast<std::size_t>(row) * heads + h) * kHeadDim;
        for (int d = lane; d < kHeadDim; d += 32) { dst[d] = __float2half(0.0f); }
    }

    const int unit = blockIdx.x * kWarps + warp;
    if (unit >= rows * heads) { return; }
    const int row = unit / heads;
    const int h   = unit - row * heads;
    const auto* src = q + (static_cast<std::size_t>(row) * heads + h) * kHeadDim;
    half* dst       = dst_base + (static_cast<std::size_t>(row) * heads + h) * kHeadDim;

    if constexpr (Int8) {
        float values[8];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            values[item] = __bfloat162float(src[lane + 32 * item]);
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            dst[lane + 32 * item] = __float2half(values[item]);
        }
    } else {
        for (int d = lane; d < kHeadDim; d += 32) {
            dst[d] = __float2half(__bfloat162float(src[d]));
        }
    }
}

// f32 [row][head][D] (padded) -> bf16 [D][heads][rows], the layout the rest of the engine
// expects from this route's output tensor.
__global__ void splitd_unstage_out_kernel(const float* __restrict__ o,
                                          __nv_bfloat16* __restrict__ out, int rows, int heads) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * heads * kHeadDim;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int d   = static_cast<int>(i % kHeadDim);
        const int h   = static_cast<int>((i / kHeadDim) % heads);
        const int row = static_cast<int>(i / (kHeadDim * static_cast<std::int64_t>(heads)));
        out[i] = __float2bfloat16(o[(static_cast<std::size_t>(row) * heads + h) * kHeadDim + d]);
    }
}

// Staging lives outside the engine workspace on purpose: it is fixed-shape and small
// (1024 rows x 24 heads x 256 dims x 2 B == 12.6 MiB for Q, 25.2 MiB for the f32 output),
// so threading it through the workspace planner would buy nothing.
struct SplitDStaging {
    cutlass::half_t* q = nullptr;
    float*           o = nullptr;
    // SplitKV3 partials: [split][row][D] plus per-row raw max/sum. The kernel documents
    // row = (query_row * heads_q + head_q), which is the same flat order the merge kernel
    // writes into the output staging, so no layout switch is needed between the two paths.
    float*           partial_out = nullptr;
    float*           partial_max = nullptr;
    float*           partial_sum = nullptr;
    int              heads = 0;
};

SplitDStaging& splitd_staging(int heads) {
    static SplitDStaging staging;
    if (staging.q == nullptr || staging.heads != heads) {
        if (staging.q != nullptr) {
            cudaFree(staging.q);
            cudaFree(staging.o);
            cudaFree(staging.partial_out);
            cudaFree(staging.partial_max);
            cudaFree(staging.partial_sum);
            staging.q = nullptr;
        }
        const std::size_t elements = static_cast<std::size_t>(kMaxRows) * heads * kHeadDim;
        const std::size_t rows3    = static_cast<std::size_t>(kMaxRows) * heads;
        CUDA_CHECK(cudaMalloc(&staging.q, elements * sizeof(cutlass::half_t)));
        CUDA_CHECK(cudaMalloc(&staging.o, elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&staging.partial_out, 3 * rows3 * kHeadDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&staging.partial_max, 3 * rows3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&staging.partial_sum, 3 * rows3 * sizeof(float)));
        staging.heads = heads;
    }
    return staging;
}

// cudaFuncSetAttribute cannot run inside a CUDA graph capture (it would invalidate the
// capture), and the engine captures during warmup. Raise it on the first call that is not
// being captured; until then the caller keeps using the llama.cpp kernel.
bool splitd_smem_ready(cudaStream_t stream) {
    using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
    static bool raised = false;
    if (raised) { return true; }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(stream, &capture);
    if (capture != cudaStreamCaptureStatusNone) { return false; }
    const auto kernel =
        FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<cutlass::half_t, false, float, false,
                                                       false, false>;
    const auto kernel_s3 =
        FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<cutlass::half_t, false, float, true,
                                                       false, false>;
    for (const void* kfn : {reinterpret_cast<const void*>(kernel),
                            reinterpret_cast<const void*>(kernel_s3)}) {
        if (cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 Traits::kSmemBytes) != cudaSuccess) {
            (void)cudaGetLastError();
            std::fprintf(stderr, "warning: sm70 split-D kernel smem attribute failed\n");
            return false;
        }
    }
    raised = true;
    return true;
}

} // namespace

bool volta_splitd_requested() {
    // On by default (2026-09-28). The route verifies the positions layout per call and
    // falls back to the llama.cpp kernel whenever the Split-D kernel's mask model does not
    // apply, and ninfer_softmax_attention_test passes with it enabled (the T=66/keys=129
    // public-contract case runs through volta_flash). NINFER_VOLTA_SPLITD=0 is the kill
    // switch back to the vendored llama.cpp kernel.
    static const bool requested = [] {
        const char* env = std::getenv("NINFER_VOLTA_SPLITD");
        return env == nullptr || std::string(env) != "0";
    }();
    return requested;
}

// One Q-block: stage Q, run the Split-D kernel over the gathered K/V, unstage the result.
// `q` is the block's bf16 row base in [D][heads][rows] layout, `out` the matching output
// row base; `k_f16`/`v_f16` are the gathered contiguous FP16 caches [key][kv_heads][D].
// `kv_len` is the padded visible key count and `kv_offset` the absolute key position of
// row 0 (the kernel derives its causal mask from those two).
template <typename Geometry>
bool volta_splitd_block(const void* q_bf16, const void* k_f16, const void* v_f16, int rows,
                        int kv_len, int kv_offset, const void* positions, int position_begin,
                        float scale, bool int8_q, void* out_bf16, cudaStream_t stream) {
    using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
    constexpr int kHeads = Geometry::QHeads;
    constexpr int kKvHeads = Geometry::KVHeads;
    constexpr int kGroup = Geometry::GroupSize;
    static_assert(Geometry::QHeads / Geometry::KVHeads == Geometry::GroupSize, "gqa mismatch");
    static_assert(Geometry::QHeads == 24 || Geometry::QHeads == 16, "unregistered geometry");

    if (rows <= 0 || rows > kMaxRows) { return false; }
    if (!splitd_smem_ready(stream)) { return false; }

    // The vendored kernel derives its causal mask from (kv_len, kv_offset) alone, i.e. it
    // assumes the query rows sit at the tail of the key range at contiguous absolute
    // positions. The public attention contract allows arbitrary positions (the llama.cpp
    // route builds its mask from the positions tensor, so it is correct either way), so
    // verify the layout here and fall back rather than compute a different attention.
    // Cost: one 4 KB D2H copy plus a sync per call, against ~50 ms of attention per call
    // at deep context -- under 0.2%. Inside a graph capture neither is legal, and the
    // capture path keeps the llama.cpp kernel.
    {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(stream, &capture);
        if (capture != cudaStreamCaptureStatusNone) { return false; }
        static int* host_positions = nullptr;
        static int host_rows = 0;
        if (host_rows < rows) {
            if (host_positions != nullptr) { cudaFreeHost(host_positions); }
            host_positions = nullptr;
            if (cudaMallocHost(&host_positions, static_cast<std::size_t>(rows) * sizeof(int)) !=
                cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            host_rows = rows;
        }
        if (positions == nullptr) { return false; }
        const auto* device_positions = static_cast<const std::int32_t*>(positions) + position_begin;
        if (cudaMemcpyAsync(host_positions, device_positions,
                            static_cast<std::size_t>(rows) * sizeof(int),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        for (int i = 0; i < rows; ++i) {
            if (host_positions[i] != kv_offset + i) { return false; }
        }
    }

    const int q_pad = round_up_rows(rows);
    SplitDStaging& staging = splitd_staging(kHeads);

    constexpr int kStageWarps = 8;
    const int stage_blocks    = (rows * kHeads + kStageWarps - 1) / kStageWarps;
    if (int8_q) {
        splitd_stage_q_kernel<true><<<stage_blocks, kStageWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_bf16), staging.q, rows, q_pad, kHeads);
    } else {
        splitd_stage_q_kernel<false><<<stage_blocks, kStageWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_bf16), staging.q, rows, q_pad, kHeads);
    }
    CUDA_CHECK(cudaGetLastError());

    // SplitKV3: three-way KV split for long prefixes. The upstream fork measured +3.7% on a
    // 176k prefill from it (it triples the CTA count so late chunks stop serializing their
    // KV sweep on a saturated grid). Same activation rule as upstream: only for a single
    // batch, above a key threshold, and only when there is a prefix to split (kv_len > q).
    const float softmax_scale_log2 = scale * 1.4426950408889634f;
    const bool use_split3 = split3_min_kv() > 0 && kv_len >= split3_min_kv() && kv_len > q_pad;
    const auto kernel = use_split3
        ? FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<cutlass::half_t, false, float, true,
                                                        false, false>
        : FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<cutlass::half_t, false, float, false,
                                                        false, false>;
    const dim3 grid(q_pad / kD256BlockM, use_split3 ? 3u : 1u,
                    static_cast<unsigned>(kKvHeads * kGroup));
    const dim3 block(kD256Threads);
    kernel<<<grid, block, Traits::kSmemBytes, stream>>>(
        reinterpret_cast<const cutlass::half_t*>(staging.q),
        reinterpret_cast<const cutlass::half_t*>(k_f16),
        reinterpret_cast<const cutlass::half_t*>(v_f16), staging.o,
        // Q/out are laid out [row][head][D] (row stride heads*D, head stride D), which is
        // what splitd_stage_q_kernel / splitd_unstage_out_kernel write and read, and what the
        // SplitKV3 merge kernel writes too (its flat row is query_row*heads_q + head_q). The
        // vendored launcher instead stages head-major (head stride q_pad*D) to match its own
        // stage_q kernel; passing the strides explicitly means either layout works, but they
        // MUST agree with the staging kernels -- an earlier revision passed head-major
        // strides against row-major staging and fed the kernel shuffled rows (caught by
        // ninfer_softmax_attention_test: T=66 keys=129 off by 3-11%).
        /*q_batch_stride*/ static_cast<int>(static_cast<std::int64_t>(q_pad) * kHeads * kHeadDim),
        /*q_row_stride  */ kHeads * kHeadDim,
        /*q_head_stride */ kHeadDim,
        /*k_outer_stride*/ 0,
        /*k_row_stride  */ kKvHeads * kHeadDim,
        /*k_head_stride */ kHeadDim,
        /*v_outer_stride*/ 0,
        /*v_row_stride  */ kKvHeads * kHeadDim,
        /*v_head_stride */ kHeadDim, q_pad, kv_len, kHeads, kKvHeads, kv_offset,
        /*softmax_scale_log2*/ softmax_scale_log2,
        /*block_table*/ nullptr, /*page_size*/ 0, /*block_table_batch_stride*/ 0,
        use_split3 ? staging.partial_out : nullptr,
        use_split3 ? staging.partial_max : nullptr,
        use_split3 ? staging.partial_sum : nullptr);
    CUDA_CHECK(cudaGetLastError());

    if (use_split3) {
        // Merge the three segments straight into the f32 staging the unstage kernel reads.
        const std::int64_t rows3 = static_cast<std::int64_t>(q_pad) * kHeads;
        FLASH_NAMESPACE::sm70_d256_splitkv3_merge_kernel
            <<<dim3(static_cast<unsigned>(rows3)), dim3(kHeadDim), 0, stream>>>(
                staging.partial_out, staging.partial_max, staging.partial_sum, staging.o, rows3,
                softmax_scale_log2);
        CUDA_CHECK(cudaGetLastError());
    }

    const int unstage_blocks = static_cast<int>(
        (static_cast<std::int64_t>(rows) * kHeads * kHeadDim + 255) / 256);
    splitd_unstage_out_kernel<<<unstage_blocks, 256, 0, stream>>>(
        staging.o, static_cast<__nv_bfloat16*>(out_bf16), rows, kHeads);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template bool volta_splitd_block<CausalD256H24Kv4>(const void*, const void*, const void*, int,
                                                   int, int, const void*, int, float, bool, void*,
                                                   cudaStream_t);
template bool volta_splitd_block<CausalD256H16Kv2>(const void*, const void*, const void*, int,
                                                   int, int, const void*, int, float, bool, void*,
                                                   cudaStream_t);

} // namespace ninfer::ops::detail
