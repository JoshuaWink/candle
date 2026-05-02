/// Flash Attention Metal Shader — Fused SDPA for Quantized GQA Models
///
/// Architecture: Tiled online softmax (FlashAttention-2 algorithm)
/// Target: Apple Metal (M1+), single-batch inference
///
/// Eliminates:
///   1. O(N²) attention matrix materialization
///   2. repeat_kv memory copy (GQA broadcast in-kernel)
///   3. Separate mask upload (causal logic in-kernel)
///   4. Multiple GPU dispatches → single kernel
///
/// Performance model:
///   - Naive SDPA: 5 dispatches × 34 layers = 170 command encoder commits
///   - Flash SDPA: 1 dispatch × 34 layers = 34 command encoder commits
///   - Memory: O(N) instead of O(N²) for the attention scores
///
/// Tiling strategy:
///   - Block rows (Br) = 32, Block cols (Bc) = 32
///   - M1 has 128 ALUs per GPU core, 32 threads per SIMD group
///   - Threadgroup size: [32, 1, 1] (one row of tiles per threadgroup)
///   - Each thread handles one query position across all KV positions

#include <metal_stdlib>
using namespace metal;

// ─── Constants ──────────────────────────────────────────────────────

constant int BLOCK_SIZE = 32;
constant float NEG_INF = -1e9f;

// ─── Kernel: flash_attn_f16 ─────────────────────────────────────────
// 
// Computes: softmax(Q×K^T / sqrt(d) + mask) × V
// With online softmax (no materialization of N×N scores)
//
// Inputs:
//   q: [n_heads, seq_q, head_dim] in float16
//   k: [n_kv_heads, seq_kv, head_dim] in float16 (GQA: n_kv_heads ≤ n_heads)
//   v: [n_kv_heads, seq_kv, head_dim] in float16
//
// Output:
//   out: [n_heads, seq_q, head_dim] in float16
//
// The kernel handles GQA by computing which kv_head each q_head maps to:
//   kv_head_idx = q_head_idx / (n_heads / n_kv_heads)

kernel void flash_attn_f16(
    device const half* q       [[buffer(0)]],
    device const half* k       [[buffer(1)]],
    device const half* v       [[buffer(2)]],
    device half* out           [[buffer(3)]],
    constant uint& seq_q       [[buffer(4)]],
    constant uint& seq_kv      [[buffer(5)]],
    constant uint& head_dim    [[buffer(6)]],
    constant uint& n_heads     [[buffer(7)]],
    constant uint& n_kv_heads  [[buffer(8)]],
    constant float& scale      [[buffer(9)]],
    constant uint& sliding_window [[buffer(10)]],  // 0 = disabled
    constant uint& is_causal   [[buffer(11)]],
    uint3 tid                  [[thread_position_in_grid]],
    uint3 tgid                 [[threadgroup_position_in_grid]]
) {
    // tid.x = query position within this head
    // tgid.y = head index
    uint q_pos = tid.x;
    uint head_idx = tgid.y;
    
    if (q_pos >= seq_q || head_idx >= n_heads) return;
    
    // GQA: map query head to kv head
    uint kv_head_idx = head_idx / (n_heads / n_kv_heads);
    
    // Pointers into Q, K, V for this head
    uint q_offset = head_idx * seq_q * head_dim + q_pos * head_dim;
    uint kv_base = kv_head_idx * seq_kv * head_dim;
    
    // Online softmax accumulators
    float row_max = NEG_INF;
    float row_sum = 0.0f;
    
    // Output accumulator (float for precision during accumulation)
    // We accumulate in registers, then write out as f16
    float acc[256]; // head_dim <= 256 for Gemma3
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }
    
    // Process KV in tiles of BLOCK_SIZE
    for (uint kv_start = 0; kv_start < seq_kv; kv_start += BLOCK_SIZE) {
        uint kv_end = min(kv_start + BLOCK_SIZE, seq_kv);
        
        for (uint kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
            // Causal mask: skip future positions
            if (is_causal && kv_pos > q_pos) continue;
            
            // Sliding window: skip positions outside window
            if (sliding_window > 0 && q_pos > kv_pos + sliding_window) continue;
            
            // Compute dot product: q[q_pos] · k[kv_pos]
            float score = 0.0f;
            uint k_offset = kv_base + kv_pos * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                score += float(q[q_offset + d]) * float(k[k_offset + d]);
            }
            score *= scale;
            
            // Online softmax update
            float old_max = row_max;
            row_max = max(row_max, score);
            float exp_diff = exp(old_max - row_max);
            
            // Rescale previous accumulations
            row_sum = row_sum * exp_diff;
            for (uint d = 0; d < head_dim; d++) {
                acc[d] *= exp_diff;
            }
            
            // Add new contribution
            float exp_score = exp(score - row_max);
            row_sum += exp_score;
            
            uint v_offset = kv_base + kv_pos * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                acc[d] += exp_score * float(v[v_offset + d]);
            }
        }
    }
    
    // Normalize by softmax denominator and write output
    uint out_offset = head_idx * seq_q * head_dim + q_pos * head_dim;
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        out[out_offset + d] = half(acc[d] * inv_sum);
    }
}

// ─── Kernel: flash_attn_q4k ────────────────────────────────────────
//
// Variant that dequantizes Q4_K weights on-the-fly.
// Q is already in f16 (after dequant by the linear projection).
// K and V are stored quantized in the KV cache when kv_quant is enabled.
//
// For now (M0/M1), KV cache is f16. This kernel is for future KV cache
// quantization where K,V are stored as Q8_0 in the cache.

// TODO: Implement when KV cache quantization is ready (optimization #9)
