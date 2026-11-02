// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXDecision

/// `profile [--length 2048] [--bits 4] [--group 64] [--iterations 20]`: times the trunk's kernel
/// classes on random data at one row length: every linear shape in float, as `quantizedMM`, and as
/// dequantize-then-matmul; attention with each mask form; the short convolution.
func profile(_ arguments: Arguments) throws {
    let length = arguments.int("length") ?? 2048
    let bits = arguments.int("bits") ?? 4
    let group = arguments.int("group") ?? 64
    let iterations = arguments.int("iterations") ?? 20
    let dtype = arguments.dtype ?? .float16
    let d = 2048

    func time(_ name: String, count: Int = 1, _ body: () -> MLXArray) -> Double {
        for _ in 0 ..< 3 { eval(body()) }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< iterations { eval(body()) }
        let ms = milliseconds(since: start) / Double(iterations)
        log(
            String(
                format: "%-44@ %8.3f ms  ×%2d = %8.2f ms", name as NSString, ms, count,
                ms * Double(count)))
        return ms * Double(count)
    }

    let x = MLXRandom.normal([1, length, d]).asType(dtype)
    // (out, in, count per trunk): 13 conv layers, 11 attention layers, 24 feed-forwards.
    let shapes: [(String, Int, Int, Int)] = [
        ("conv.in_proj", 3 * d, d, 13), ("conv.out_proj", d, d, 13),
        ("attn.q_proj", d, d, 11), ("attn.k/v_proj", 512, d, 22), ("attn.gate_proj", 16, d, 11),
        ("attn.qkvg fused", d + 1024 + 16, d, 11), ("attn.o_proj", d, d, 11),
        ("ffn.w1/w3", 7680, d, 47), ("ffn.w1w3 fused", 15360, d, 23), ("ffn.w2", d, 7680, 23),
    ]
    var totals = [String: Double]()
    for (name, o, i, count) in shapes {
        let input = i == d ? x : MLXRandom.normal([1, length, i]).asType(dtype)
        let w = (MLXRandom.normal([o, i]) * 0.02).asType(dtype)
        let (wq, s, b) = quantized(w, groupSize: group, bits: bits)
        eval(w, wq, s, b!)
        let skip = name.contains("fused")
        let f = time("\(name) float", count: count) { matmul(input, w.transposed()) }
        let q = time("\(name) qmm \(bits)b g\(group)", count: count) {
            quantizedMM(
                input, wq, scales: s, biases: b, transpose: true, groupSize: group, bits: bits)
        }
        let dq = time("\(name) dequant+matmul", count: count) {
            matmul(
                input,
                dequantized(wq, scales: s, biases: b, groupSize: group, bits: bits).transposed())
        }
        if !skip {
            totals["float", default: 0] += f
            totals["qmm", default: 0] += q
            totals["dequant", default: 0] += dq
        }
    }

    let q = MLXRandom.normal([1, 16, length, 128]).asType(dtype)
    let k = MLXRandom.normal([1, 4, length, 128]).asType(dtype)
    let v = MLXRandom.normal([1, 4, length, 128]).asType(dtype)
    let positions = MLXArray(Int32(0) ..< Int32(length))
    let boolMask = positions[.newAxis] .<= positions[0..., .newAxis] + 300
    let additive = MLX.where(boolMask, MLXArray(Float(0)), MLXArray(Float(-40000))).asType(dtype)
    eval(q, k, v, boolMask, additive)
    let scale = Float(pow(128.0, -0.5))
    totals["attention"] = time("sdpa bool mask", count: 11) {
        MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(boolMask))
    }
    _ = time("sdpa additive mask", count: 11) {
        MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(additive))
    }
    _ = time("sdpa no mask", count: 11) {
        MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
    }
    _ = time("sdpa causal", count: 11) {
        MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)
    }
    let bcx = MLXRandom.normal([1, length, 3 * d]).asType(dtype)
    let taps = (MLXRandom.normal([7, d]) * 0.1).asType(dtype)
    eval(bcx, taps)
    totals["conv"] = time("short conv (ops)", count: 13) {
        D1LFM3Engine.shortConvolution(bcx, weight: taps, fused: false)
    }
    _ = time("short conv (kernel)", count: 13) {
        D1LFM3Engine.shortConvolution(bcx, weight: taps, fused: true)
    }
    let normWeight = MLXArray.ones([d]).asType(dtype)
    totals["norms"] = time("rms norm", count: 97) {
        MLXFast.rmsNorm(x, weight: normWeight, eps: 1e-5)
    }
    log("totals at L=\(length): \(totals.mapValues { Int($0.rounded()) })")
}
