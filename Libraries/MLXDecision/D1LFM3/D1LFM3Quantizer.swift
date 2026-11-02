// Copyright © 2026 Faux Fox, Inc.
// The weighted scale and minimum search follows llama.cpp's `make_qkx2_quants` and its
// importance-matrix weights (ggml-quants.c, MIT license).

import Foundation
import MLX
import MLXNN

/// How an export quantizes each tensor: a default and first-match rules over module paths.
public struct D1LFM3Recipe: Codable, Sendable {
    public struct Rule: Codable, Sendable {
        /// A regular expression matched against the whole module path.
        public var match: String
        /// nil keeps the matched modules in floating point.
        public var setting: D1LFM3Configuration.Quantization.Setting?

        public init(match: String, setting: D1LFM3Configuration.Quantization.Setting?) {
            self.match = match
            self.setting = setting
        }
    }

    public enum Method: String, Codable, Sendable {
        /// MLX's own min/max scales.
        case minMax = "minmax"
        /// llama.cpp's weighted search over scale and minimum, with importance weights when given.
        case search
        /// GPTQ error compensation with input Hessians; group scales from the weighted search.
        case gptq
    }

    public var defaults: D1LFM3Configuration.Quantization.Setting?
    public var rules: [Rule]
    public var method: Method
    /// Fold activation-aware channel scales into the trunk before quantizing (needs an
    /// importance matrix).
    public var scaling: Bool?
    /// GPTQ's Hessian damping, a fraction of its mean diagonal (default 0.01).
    public var damping: Float?

    public init(
        defaults: D1LFM3Configuration.Quantization.Setting?, rules: [Rule] = [],
        method: Method = .search
    ) {
        self.defaults = defaults
        self.rules = rules
        self.method = method
    }

    public func setting(for path: String) -> D1LFM3Configuration.Quantization.Setting? {
        for rule in rules where path.wholeMatch(of: try! Regex(rule.match)) != nil {
            return rule.setting
        }
        return defaults
    }
}

public enum D1LFM3Quantizer {
    public struct Quantized {
        public let weight: MLXArray
        public let scales: MLXArray
        public let biases: MLXArray?
    }

    /// Quantizes `w` [O, I]. `importance` [I] is the mean square of each input channel over
    /// calibration tokens (an importance matrix); `scaleType` is the stored type of scales and biases.
    public static func quantize(
        _ w: MLXArray, setting: D1LFM3Configuration.Quantization.Setting,
        method: D1LFM3Recipe.Method, importance: MLXArray? = nil, scaleType: DType
    ) -> Quantized {
        let mode = setting.quantizationMode
        guard mode == .affine, method != .minMax else {
            let (q, s, b) = MLX.quantized(
                w.asType(scaleType), groupSize: setting.groupSize, bits: setting.bits, mode: mode)
            return Quantized(weight: q, scales: s, biases: b)
        }
        // Rows in chunks keep the search's temporaries small.
        let rows = w.dim(0)
        let chunk = max(1, (1 << 22) / w.dim(1))
        var weights = [MLXArray]()
        var scales = [MLXArray]()
        var biases = [MLXArray]()
        for start in stride(from: 0, to: rows, by: chunk) {
            let part = w[start ..< min(rows, start + chunk)]
            let (q, s, b) = searchAffine(
                part, groupSize: setting.groupSize, bits: setting.bits, importance: importance,
                scaleType: scaleType)
            eval(q, s, b)
            weights.append(q)
            scales.append(s)
            biases.append(b)
        }
        return Quantized(
            weight: concatenated(weights), scales: concatenated(scales),
            biases: concatenated(biases))
    }

    /// Per group: try scales around min/max, fit the scale and minimum by weighted least squares
    /// for each candidate's codes, and keep the lowest weighted error.
    static func searchAffine(
        _ w: MLXArray, groupSize: Int, bits: Int, importance: MLXArray?, scaleType: DType
    ) -> (MLXArray, MLXArray, MLXArray) {
        let (o, i) = (w.dim(0), w.dim(1))
        let groups = i / groupSize
        let x = w.asType(.float32).reshaped(o, groups, groupSize)
        let nmax = Float((1 << bits) - 1)
        let (scale, minimum) = fitAffine(
            x, weight: importance?.asType(.float32).reshaped(1, groups, groupSize), bits: bits,
            scaleType: scaleType)
        let codes = clip(round((x - minimum) / scale), min: 0, max: nmax).asType(.uint32)
        return (
            pack(codes.reshaped(o, i), bits: bits), scale.reshaped(o, groups).asType(scaleType),
            minimum.reshaped(o, groups).asType(scaleType)
        )
    }

    /// The scale and minimum of each group of `x` [O, G, g], float32 values already rounded to
    /// `scaleType`. `importance` broadcasts against `x`.
    static func fitAffine(_ x: MLXArray, weight importance: MLXArray?, bits: Int, scaleType: DType)
        -> (MLXArray, MLXArray)
    {
        let nmax = Float((1 << bits) - 1)
        // llama.cpp: weights = qw * sqrt(sigma2 + x²) with an importance matrix, else sqrt(sigma2 + x²).
        let sigma2 = (x * x).mean(axis: -1, keepDims: true)
        var weight = sqrt(sigma2 + x * x)
        if let importance { weight = weight * importance }
        let lo = x.min(axis: -1, keepDims: true)
        let hi = x.max(axis: -1, keepDims: true)
        let range = maximum(hi - lo, MLXArray(Float(1e-12)))

        func fit(_ codes: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let sw = weight.sum(axis: -1, keepDims: true)
            let sl = (weight * codes).sum(axis: -1, keepDims: true)
            let sl2 = (weight * codes * codes).sum(axis: -1, keepDims: true)
            let sx = (weight * x).sum(axis: -1, keepDims: true)
            let sxl = (weight * codes * x).sum(axis: -1, keepDims: true)
            let d = sw * sl2 - sl * sl
            let ok = d .> 0
            let safe = MLX.where(ok, d, MLXArray(Float(1)))
            let scale = MLX.where(ok, (sw * sxl - sx * sl) / safe, range / nmax)
            let minimum = MLX.where(ok, (sl2 * sx - sl * sxl) / safe, lo)
            let e = scale * codes + minimum - x
            return (scale, minimum, (weight * e * e).sum(axis: -1, keepDims: true))
        }

        var bestScale = range / nmax
        var bestMin = lo
        var bestError = { () -> MLXArray in
            let codes = clip(round((x - lo) / bestScale), min: 0, max: nmax)
            let e = bestScale * codes + lo - x
            return (weight * e * e).sum(axis: -1, keepDims: true)
        }()
        for step in 0 ... 20 {
            let iscale = (nmax - 1 + 0.1 * Float(step)) / range
            let codes = clip(round((x - lo) * iscale), min: 0, max: nmax)
            let (scale, minimum, error) = fit(codes)
            let better = error .< bestError
            bestScale = MLX.where(better, scale, bestScale)
            bestMin = MLX.where(better, minimum, bestMin)
            bestError = MLX.where(better, error, bestError)
        }
        // Stored in `scaleType`; codes are chosen against the stored values.
        var scale = bestScale.asType(scaleType).asType(.float32)
        scale = MLX.where(abs(scale) .< 1e-30, MLXArray(Float(1e-8)), scale)
        return (scale, bestMin.asType(scaleType).asType(.float32))
    }

    /// MLX's affine packing: codes LSB first, `32 / bits` per word for powers of two, else a bit stream.
    static func pack(_ codes: MLXArray, bits: Int) -> MLXArray {
        let (o, i) = (codes.dim(0), codes.dim(1))
        if 32 % bits == 0 {
            let per = 32 / bits
            let shifts = MLXArray((0 ..< per).map { UInt32($0 * bits) })
            return (codes.reshaped(o, i / per, per) << shifts).sum(axis: -1).asType(.uint32)
        }
        let bitPlanes =
            (codes[.ellipsis, .newAxis] >> MLXArray((0 ..< bits).map { UInt32($0) }))
            & MLXArray(UInt32(1))
        let stream = bitPlanes.reshaped(o, -1, 32)
        return (stream << MLXArray((0 ..< 32).map { UInt32($0) })).sum(axis: -1).asType(.uint32)
    }
}
