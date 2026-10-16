//
//  KimiDeltaAttention.swift
//  mlx-swift-lm
//
//  Kimi Delta Attention (KDA): the gated delta rule with a decay per key channel.
//  Follows `recurrent_kimi_delta_attention` in transformers' `modeling_glm5_next.py`.
//

import Foundation
import MLX
import MLXNN

// MARK: - Decay

private let computeKDALowerBoundDecay:
    @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
        f, aLog, dtBias, lowerBound in
        exp(
            lowerBound
                * sigmoid(
                    exp(aLog.asType(.float32))
                        * (f.asType(.float32) + dtBias.asType(.float32))))
    }

private let computeKDASoftplusDecay: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { f, aLog, dtBias in
        exp(-exp(aLog.asType(.float32)) * softplus(f.asType(.float32) + dtBias.asType(.float32)))
    }

/// Per-key-channel decay multipliers `exp(g)` in float32, shaped like `f`.
///
/// - Parameters:
///   - f: forget-gate logits `[B, T, H, Dk]`.
///   - aLog: `A_log`, broadcastable to `[H, 1]`.
///   - dtBias: `dt_bias`, broadcastable to `[H, Dk]`.
///   - lowerBound: the "safe gate" bound. With it, `g = lowerBound * sigmoid(exp(A_log) *
///     (f + dt_bias))`; without it, `g = -exp(A_log) * softplus(f + dt_bias)`.
public func kimiDeltaDecay(
    f: MLXArray, aLog: MLXArray, dtBias: MLXArray, lowerBound: Float?
) -> MLXArray {
    if let lowerBound {
        return computeKDALowerBoundDecay(f, aLog, dtBias, MLXArray(lowerBound))
    }
    return computeKDASoftplusDecay(f, aLog, dtBias)
}

// MARK: - Metal kernel

private func makeKimiDeltaKernel(masked: Bool, checkpoints: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = masked ? "mask[b_idx * T + t]" : "true"
    let checkpointSource =
        checkpoints
        ? """
        if (t + 1 < T) {
          auto c_state = checkpoint_states + ((t * B * H + n) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            c_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
          }
        }
        """
        : ""

    // One SIMD-group owns one value row of one head's [Dv, Dk] state; lanes split Dk.
    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / H;
            auto h_idx = n % H;
            constexpr int n_per_t = Dk / 32;
            const float scale = metal::rsqrt(static_cast<float>(Dk));

            // q, k, g: [B, T, H, Dk]; v, y: [B, T, H, Dv]; beta: [B, T, H]
            auto q_ = q + (b_idx * T * H + h_idx) * Dk;
            auto k_ = k + (b_idx * T * H + h_idx) * Dk;
            auto g_ = g + (b_idx * T * H + h_idx) * Dk;
            auto v_ = v + (b_idx * T * H + h_idx) * Dv;
            y += (b_idx * T * H + h_idx) * Dv;
            auto beta_ = beta + b_idx * T * H;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // state_in, state_out: [B, H, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float qv[n_per_t];
                float kv[n_per_t];
                float q_sq = 0.0f;
                float k_sq = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  qv[i] = static_cast<float>(q_[s_idx]);
                  kv[i] = static_cast<float>(k_[s_idx]);
                  q_sq += qv[i] * qv[i];
                  k_sq += kv[i] * kv[i];
                }
                float q_scale = scale;
                float k_scale = 1.0f;
                if (NormalizeQK) {
                  q_scale = scale / metal::sqrt(simd_sum(q_sq) + 1e-6f);
                  k_scale = 1.0f / metal::sqrt(simd_sum(k_sq) + 1e-6f);
                }

                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  kv[i] = kv[i] * k_scale;
                  state[i] = state[i] * static_cast<float>(g_[n_per_t * dk_idx + i]);
                  kv_mem += state[i] * kv[i];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                    * static_cast<float>(beta_[h_idx]);

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  state[i] = state[i] + kv[i] * delta;
                  out += state[i] * qv[i];
                }
                out = simd_sum(out) * q_scale;
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              } else if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(0);
              }
              \(checkpointSource)
              q_ += H * Dk;
              k_ += H * Dk;
              g_ += H * Dk;
              v_ += H * Dv;
              y += H * Dv;
              beta_ += H;
            }
            for (int i = 0; i < n_per_t; ++i) {
              o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T", "B"]
    if masked {
        inputNames.append("mask")
    }
    var outputNames = ["y", "state_out"]
    if checkpoints {
        outputNames.append("checkpoint_states")
    }
    let suffix = (masked ? "_mask" : "") + (checkpoints ? "_checkpoints" : "")
    return MLXFast.metalKernel(
        name: "kimi_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: outputNames,
        source: source)
}

private final class KimiDeltaKernels: Sendable {
    static let shared = KimiDeltaKernels()

    let plain: MLXFast.MLXFastKernel?
    let masked: MLXFast.MLXFastKernel?
    let checkpointing: MLXFast.MLXFastKernel?

    private init() {
        plain = makeKimiDeltaKernel(masked: false, checkpoints: false)
        masked = makeKimiDeltaKernel(masked: true, checkpoints: false)
        checkpointing = makeKimiDeltaKernel(masked: false, checkpoints: true)
    }
}

/// Runs the fused recurrence. Returns the output, the final state, and, when requested, the
/// state after each of the first `T - 1` positions stacked as `[T - 1, B, H, Dv, Dk]`.
func kimiDeltaKernel(
    q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray, beta: MLXArray, state: MLXArray,
    mask: MLXArray? = nil, normalizeQK: Bool, checkpoints: Bool = false
) -> (MLXArray, MLXArray, MLXArray?) {
    let (batch, length, heads, keyDim) = k.shape4
    let valueDim = v.dim(3)
    precondition(keyDim % 32 == 0, "the KDA kernel needs a key head dim divisible by 32")
    precondition(!(checkpoints && mask != nil), "checkpoints need an unmasked sequence")

    let kernels = KimiDeltaKernels.shared
    let kernel = checkpoints ? kernels.checkpointing : mask != nil ? kernels.masked : kernels.plain
    guard let kernel else { fatalError("Kimi delta kernel not available") }

    var inputs = [q, k, v, decay, beta, state, MLXArray(length), MLXArray(batch)]
    if let mask { inputs.append(mask) }
    var outputShapes = [[batch, length, heads, valueDim], state.shape]
    var outputDTypes = [q.dtype, state.dtype]
    if checkpoints {
        outputShapes.append([Swift.max(length - 1, 1)] + state.shape)
        outputDTypes.append(state.dtype)
    }
    let outputs = kernel(
        inputs,
        template: [
            ("InT", q.dtype), ("StT", state.dtype), ("Dk", keyDim), ("Dv", valueDim),
            ("H", heads), ("NormalizeQK", normalizeQK),
        ],
        grid: (32, valueDim, batch * heads),
        threadGroup: (32, 4, 1),
        outputShapes: outputShapes,
        outputDTypes: outputDTypes)
    return (outputs[0], outputs[1], checkpoints ? outputs[2] : nil)
}

// MARK: - Ops fallback

/// The KDA recurrence written with MLX ops, one position at a time. Used when the kernel
/// does not apply (CPU, key dims not divisible by 32) and as the kernel's test reference.
public func kimiDeltaOps(
    q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray, beta: MLXArray,
    state: MLXArray? = nil, mask: MLXArray? = nil, normalizeQK: Bool = true
) -> (output: MLXArray, state: MLXArray, checkpoints: [MLXArray]) {
    let (batch, length, heads, keyDim) = k.shape4
    var q = q.asType(.float32)
    var k = k.asType(.float32)
    if normalizeQK {
        q = q / sqrt((q * q).sum(axis: -1, keepDims: true) + 1e-6)
        k = k / sqrt((k * k).sum(axis: -1, keepDims: true) + 1e-6)
    }
    q = q * pow(Float(keyDim), -0.5)
    let v = v.asType(.float32)
    let decay = decay.asType(.float32)
    let beta = beta.asType(.float32)

    var s =
        state?.asType(.float32)
        ?? MLXArray.zeros([batch, heads, v.dim(3), keyDim], dtype: .float32)
    var ys = [MLXArray]()
    var checkpoints = [MLXArray]()
    for t in 0 ..< length {
        let previous = s
        let kt = k[0..., t].expandedDimensions(axis: -2)
        s = s * decay[0..., t].expandedDimensions(axis: -2)
        let kvMem = (s * kt).sum(axis: -1)
        let delta = (v[0..., t] - kvMem) * beta[0..., t].expandedDimensions(axis: -1)
        s = s + kt * delta.expandedDimensions(axis: -1)
        var y = (s * q[0..., t].expandedDimensions(axis: -2)).sum(axis: -1)
        if let mask {
            let keep = mask[0..., t].reshaped(batch, 1, 1, 1)
            s = MLX.where(keep, s, previous)
            y = MLX.where(keep.squeezed(axis: -1), y, MLXArray(Float(0)))
        }
        ys.append(y)
        if t + 1 < length {
            checkpoints.append(s)
        }
    }
    return (MLX.stacked(ys, axis: 1), s, checkpoints)
}

// MARK: - Public API

/// Kimi Delta Attention: the gated delta rule with a decay per key channel.
///
/// For each position: `S = S * decay`, `S += k (beta (v - S k))`, `y = S (q / sqrt(Dk))`,
/// with the recurrent state `S` kept in float32 as `[B, H, Dv, Dk]`.
///
/// - Parameters:
///   - q, k: `[B, T, H, Dk]`. With `normalizeQK`, both are L2-normalized in float32 first.
///   - v: `[B, T, H, Dv]`.
///   - decay: per-channel decay multipliers `[B, T, H, Dk]` from ``kimiDeltaDecay(f:aLog:dtBias:lowerBound:)``.
///   - beta: write strengths `[B, T, H]`, usually `sigmoid(b)`.
///   - state: the previous state, or `nil` for zeros.
///   - mask: `[B, T]`; masked positions leave the state unchanged and output zeros.
/// - Returns: the output in `q`'s dtype and the final state.
public func kimiDeltaUpdate(
    q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray, beta: MLXArray,
    state: MLXArray? = nil, mask: MLXArray? = nil, normalizeQK: Bool = true,
    useKernel: Bool = true
) -> (MLXArray, MLXArray) {
    let state = initialKimiDeltaState(q: q, v: v, state: state)
    if useKernel, canUseKimiDeltaKernel(q) {
        let (y, s, _) = kimiDeltaKernel(
            q: q, k: k, v: v, decay: decay.asType(.float32), beta: beta.asType(.float32),
            state: state, mask: mask, normalizeQK: normalizeQK)
        return (y, s)
    }
    let result = kimiDeltaOps(
        q: q, k: k, v: v, decay: decay, beta: beta, state: state, mask: mask,
        normalizeQK: normalizeQK)
    return (result.output.asType(q.dtype), result.state)
}

/// Like ``kimiDeltaUpdate(q:k:v:decay:beta:state:mask:normalizeQK:useKernel:)`` for an
/// unmasked sequence, and also returns the state after each of the first `T - 1` positions.
/// Speculative verification uses these states to rewind rejected positions.
public func kimiDeltaUpdateWithCheckpoints(
    q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray, beta: MLXArray,
    state: MLXArray? = nil, normalizeQK: Bool = true, useKernel: Bool = true
) -> (output: MLXArray, state: MLXArray, checkpoints: [MLXArray]) {
    let state = initialKimiDeltaState(q: q, v: v, state: state)
    let length = q.dim(1)
    if useKernel, canUseKimiDeltaKernel(q) {
        let (y, s, stacked) = kimiDeltaKernel(
            q: q, k: k, v: v, decay: decay.asType(.float32), beta: beta.asType(.float32),
            state: state, normalizeQK: normalizeQK, checkpoints: true)
        let checkpoints = length > 1 ? (0 ..< (length - 1)).map { stacked![$0] } : []
        return (y, s, checkpoints)
    }
    let result = kimiDeltaOps(
        q: q, k: k, v: v, decay: decay, beta: beta, state: state, normalizeQK: normalizeQK)
    return (result.output.asType(q.dtype), result.state, result.checkpoints)
}

private func initialKimiDeltaState(q: MLXArray, v: MLXArray, state: MLXArray?) -> MLXArray {
    let state = state ?? MLXArray.zeros([q.dim(0), v.dim(2), v.dim(3), q.dim(3)], dtype: .float32)
    return state.dtype == .float32 ? state : state.asType(.float32)
}

private func canUseKimiDeltaKernel(_ q: MLXArray) -> Bool {
    q.dim(3) % 32 == 0 && Device.defaultDevice().deviceType == .gpu
}
