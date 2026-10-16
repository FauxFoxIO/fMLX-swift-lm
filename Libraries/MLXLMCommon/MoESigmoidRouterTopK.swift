import MLX

// MARK: - Fused sigmoid + bias router top-k

/// One-kernel router tail for DeepSeek-style `noaux_tc` routing with a single group.
///
/// Each row runs in one threadgroup with one thread per expert. A thread computes
/// `sigmoid(logit)`, ranks `sigmoid + bias` against the row, and the winners write their
/// unbiased score. Thread 0 then normalizes (optional) and applies the routed scale.
/// Ties rank the lower expert index first.
private let sigmoidRouterTopKSource = """
    uint row = threadgroup_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;

    threadgroup ulong sk[E_];
    threadgroup float top_v[K_];

    float s = 1.0f / (1.0f + metal::exp(-static_cast<float>(logits[row * E_ + t])));
    float v = s + static_cast<float>(bias[t]);
    uint b = (v == 0.0f) ? 0u : as_type<uint>(v);
    uint mono = isnan(v) ? 0xFFFFFFFFu : (b ^ ((uint)(((int)b) >> 31) | 0x80000000u));
    ulong key = (((ulong)mono) << 32) | (ulong)(0xFFFFFFFFu - t);
    sk[t] = key;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int above = 0;
    for (uint j = 0; j < E_; ++j) {
        above += (sk[j] > key) ? 1 : 0;
    }
    if (above < K_) {
        top_v[above] = s;
        inds[row * K_ + above] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t == 0) {
        float scale = static_cast<float>(scaling[0]);
        if (NORMALIZE_) {
            float acc = 0.0f;
            for (int q = 0; q < K_; ++q) {
                acc += top_v[q];
            }
            scale = scale / (acc + 1e-20f);
        }
        for (int q = 0; q < K_; ++q) {
            scores[row * K_ + q] = top_v[q] * scale;
        }
    }
    """

private final class SigmoidRouterTopKKernel: Sendable {
    static let shared = SigmoidRouterTopKKernel()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "sigmoid_bias_router_topk",
            inputNames: ["logits", "bias", "scaling"],
            outputNames: ["inds", "scores"],
            source: sigmoidRouterTopKSource
        )
    }
}

/// Whether `logits` can use the fused kernel: one thread per expert must fit a threadgroup.
package func supportsFusedSigmoidRouterTopK(_ logits: MLXArray, k: Int) -> Bool {
    let e = logits.dim(-1)
    return e > 0 && e <= maxFusedRouterExperts && k > 0 && k <= e && logits.size > 0
        && logits.dtype.isFloatingPoint
}

/// Fused sigmoid, bias-corrected top-k, unbiased score gather, optional normalization,
/// and routed scaling. Returns `uint32` indices in descending biased-score order and
/// float32 scores, both shaped `[..., k]`.
package func fusedSigmoidRouterTopK(
    logits: MLXArray, bias: MLXArray, k: Int, normalize: Bool, scalingFactor: Float
) -> (indices: MLXArray, scores: MLXArray) {
    let e = logits.dim(-1)
    precondition(e <= maxFusedRouterExperts, "fused router exceeds Metal threadgroup limit")
    precondition(k > 0 && k <= e, "invalid fused router top-k")
    precondition(bias.size == e, "router bias must have one value per expert")

    let rows = logits.size / e
    let shape = Array(logits.shape.dropLast()) + [k]
    let out = SigmoidRouterTopKKernel.shared.kernel(
        [logits, bias.asType(.float32), MLXArray([scalingFactor])],
        template: [
            ("E_", e),
            ("K_", k),
            ("NORMALIZE_", normalize ? 1 : 0),
        ],
        grid: (e, rows, 1),
        threadGroup: (e, 1, 1),
        outputShapes: [shape, shape],
        outputDTypes: [.uint32, .float32]
    )
    return (out[0], out[1])
}

/// Unfused reference: the same routing as `fusedSigmoidRouterTopK` with generic MLX ops.
/// Index order within the top-k is unspecified.
package func chainSigmoidRouterTopK(
    logits: MLXArray, bias: MLXArray, k: Int, normalize: Bool, scalingFactor: Float
) -> (indices: MLXArray, scores: MLXArray) {
    let scores = sigmoid(logits.asType(.float32))
    let selection = scores + bias.asType(.float32)
    let indices = argPartition(-selection, kth: k - 1, axis: -1)[.ellipsis, ..<k]
    var selected = takeAlong(scores, indices, axis: -1)
    if normalize {
        selected = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)
    }
    return (indices, selected * scalingFactor)
}

/// `noaux_tc` routing with one expert group: selects on `sigmoid(logits) + bias` and
/// weights experts by `sigmoid(logits)`, normalized if requested, times `scalingFactor`.
package func sigmoidRouterTopK(
    logits: MLXArray, bias: MLXArray, k: Int, normalize: Bool, scalingFactor: Float
) -> (indices: MLXArray, scores: MLXArray) {
    if supportsFusedSigmoidRouterTopK(logits, k: k) {
        return fusedSigmoidRouterTopK(
            logits: logits, bias: bias, k: k, normalize: normalize, scalingFactor: scalingFactor)
    }
    return chainSigmoidRouterTopK(
        logits: logits, bias: bias, k: k, normalize: normalize, scalingFactor: scalingFactor)
}
