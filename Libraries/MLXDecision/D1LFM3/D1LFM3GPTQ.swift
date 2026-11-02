// Copyright © 2026 Faux Fox, Inc.
// GPTQ (Frantar et al. 2022): quantize a layer's columns in order and push each column's error
// onto the columns not yet quantized, weighted by the inverse Hessian of the layer's inputs.

import Foundation
import MLX
import MLXNN

extension D1LFM3Quantizer {
    /// Groups of layers that read the same input share one Hessian, stored under the first path.
    public static func hessianOwner(_ path: String) -> String {
        for (sibling, owner) in [
            (".k_proj", ".q_proj"), (".v_proj", ".q_proj"), (".gate_proj", ".q_proj"),
            (".feed_forward.w3", ".feed_forward.w1"),
        ] where path.hasSuffix(sibling) {
            return String(path.dropLast(sibling.count)) + owner
        }
        return path
    }

    /// GPTQ for `w` [O, I] with input Hessian `hessian` [I, I] (the mean of `x xᵀ`). Each group's
    /// scale and minimum come from the importance-weighted search over the group's error-updated
    /// weights, with the Hessian's diagonal as importance.
    public static func gptq(
        _ w: MLXArray, hessian: MLXArray, setting: D1LFM3Configuration.Quantization.Setting,
        scaleType: DType, blockSize: Int = 128, damping: Float = 0.01
    ) -> Quantized {
        let (o, i) = (w.dim(0), w.dim(1))
        let g = setting.groupSize
        let nmax = Float((1 << setting.bits) - 1)
        precondition(blockSize % g == 0 && i % blockSize == 0)

        var h = hessian.asType(.float32)
        let diagonal = MLX.diagonal(h)
        let dead = diagonal .<= 0
        var weights = MLX.where(dead[.newAxis], MLXArray(Float(0)), w.asType(.float32))
        let damp = damping * MLX.where(dead, MLXArray(Float(0)), diagonal).mean()
        h =
            h + MLXArray.eye(i) * damp
            + MLX.diag(MLX.where(dead, MLXArray(Float(1)), MLXArray(Float(0))))
        let cpu = StreamOrDevice.cpu
        let lower = MLXLinalg.cholesky(h, stream: cpu)
        let inverse = MLXLinalg.choleskyInv(lower, stream: cpu)
        let hinv = MLXLinalg.cholesky(inverse, upper: true, stream: cpu)
        let importance = MLX.where(dead, MLXArray(Float(0)), diagonal)
        eval(hinv, weights, importance)

        var codes = [MLXArray]()
        var scales = [MLXArray]()
        var minimums = [MLXArray]()
        for start in stride(from: 0, to: i, by: blockSize) {
            let end = start + blockSize
            var block = weights[0..., start ..< end]
            let hb = hinv[start ..< end, start ..< end]
            var errors = [MLXArray]()
            var scale = MLXArray(Float(1))
            var minimum = MLXArray(Float(0))
            for c in 0 ..< blockSize {
                if c % g == 0 {
                    let group = block[0..., c ..< c + g].reshaped(o, 1, g)
                    let imp = importance[(start + c) ..< (start + c + g)].reshaped(1, 1, g)
                    (scale, minimum) = fitAffine(
                        group, weight: imp, bits: setting.bits, scaleType: scaleType)
                    scale = scale.reshaped(o)
                    minimum = minimum.reshaped(o)
                    scales.append(scale)
                    minimums.append(minimum)
                }
                let column = block[0..., c]
                let q = clip(round((column - minimum) / scale), min: 0, max: nmax)
                let error = (column - (q * scale + minimum)) / hb[c, c]
                block =
                    block - error.expandedDimensions(axis: 1) * hb[c].expandedDimensions(axis: 0)
                codes.append(q)
                errors.append(error)
            }
            let e = stacked(errors, axis: 1)
            if end < i {
                weights[0..., end...] =
                    weights[0..., end...] - matmul(e, hinv[start ..< end, end...])
            }
            eval([weights] + Array(codes.suffix(blockSize)) + scales + minimums)
        }
        let q = stacked(codes, axis: 1).asType(.uint32)
        return Quantized(
            weight: pack(q, bits: setting.bits),
            scales: stacked(scales, axis: 1).asType(scaleType),
            biases: stacked(minimums, axis: 1).asType(scaleType))
    }
}

extension D1LFM3Engine {
    /// Runs `body` while summing `x xᵀ` of the inputs of the linear layers at `paths`; returns each
    /// layer's mean over tokens [I, I].
    public func collectHessians(_ paths: Set<String>, _ body: () throws -> Void) rethrows
        -> [String: MLXArray]
    {
        var sums = [String: MLXArray]()
        var counts = [String: Int]()
        let originals = network.leafModules().flattened().compactMap {
            path, module -> (String, Linear)? in
            guard paths.contains(path), let linear = module as? Linear, !(linear is QuantizedLinear)
            else { return nil }
            return (path, linear)
        }
        let observed = originals.map { path, linear -> (String, Linear) in
            (
                path,
                D1LFM3ObservedLinear(linear) { x in
                    let flat = x.reshaped(-1, x.dim(-1)).asType(.float32)
                    let s = matmul(flat.transposed(), flat)
                    sums[path] = sums[path].map { $0 + s } ?? s
                    counts[path, default: 0] += flat.dim(0)
                    eval(sums[path]!)
                }
            )
        }
        replaceLinear(Dictionary(uniqueKeysWithValues: observed))
        defer { replaceLinear(Dictionary(uniqueKeysWithValues: originals)) }
        try body()
        return sums.reduce(into: [:]) { $0[$1.key] = $1.value / Float(max(1, counts[$1.key] ?? 1)) }
    }
}
