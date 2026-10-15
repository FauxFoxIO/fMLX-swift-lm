// Copyright © 2026 Faux Fox, Inc.
// Port of the d1-omni reference `audio.py` (LiquidAI/d1-omni-600M; NeMo FastConformer lineage)
// and MLX-VLM `d1_omni/audio.py` with its Parakeet encoder.

import Foundation
import MLX
import MLXNN

/// NeMo's log-mel front end in eval mode: preemphasis, |STFT|^2, Slaney mel, log, per-bin norm.
enum D1OmniMelFrontend {
    static let sampleRate = 16_000
    static let minimumSamples = 8_000
    static let maximumSeconds = 30
    static let fftSize = 512
    static let window = 400
    static let hop = 160

    /// Mono 16 kHz samples cut to 30 s and padded with silence to 0.5 s.
    static func clip(_ samples: [Float]) -> [Float] {
        var x = Array(samples.prefix(maximumSeconds * sampleRate))
        if x.count < minimumSamples {
            x += [Float](repeating: 0, count: minimumSamples - x.count)
        }
        return x
    }

    /// Features [frames + 1, bins] in float32 and the valid frame count.
    static func features(_ clip: [Float], bins: Int) -> (MLXArray, Int) {
        let n = clip.count
        let frames = n / hop
        var emphasized = clip
        for i in stride(from: n - 1, to: 0, by: -1) {
            emphasized[i] = clip[i] - 0.97 * clip[i - 1]
        }
        let padding = fftSize / 2
        let padded =
            [Float](repeating: 0, count: padding) + emphasized
            + [Float](repeating: 0, count: padding)
        let count = 1 + (padded.count - fftSize) / hop
        let signal = MLXArray(padded)
        let windows = asStrided(signal, [count, fftSize], strides: [hop, 1])
        let spectrum = MLXFFT.rfft(windows * MLXArray(hannWindow()), axis: -1)
        let power = (spectrum.realPart().square() + spectrum.imaginaryPart().square()).asType(
            .float32)
        let filters = MLXArray(slaneyFilterbank(bins: bins), [bins, fftSize / 2 + 1])
        var mel = log(matmul(filters, power.transposed()) + Float(pow(2.0, -24.0)))
        let valid = (MLXArray(Int32(0) ..< Int32(count)) .< Int32(frames))[.newAxis]
        let validf = valid.asType(.float32)
        let mean = (mel * validf).sum(axis: 1, keepDims: true) / Float(frames)
        let variance =
            ((mel - mean).square() * validf).sum(axis: 1, keepDims: true)
            / Float(max(frames - 1, 1))
        mel = (mel - mean) / (sqrt(variance) + Float(1e-5))
        mel = MLX.where(valid, mel, MLXArray(Float(0)))
        return (mel.transposed(), frames)
    }

    /// `hann_window(400, periodic=False)` centred in 512 samples.
    static func hannWindow() -> [Float] {
        var result = [Float](repeating: 0, count: fftSize)
        let left = (fftSize - window) / 2
        for k in 0 ..< window {
            result[left + k] = Float(
                0.5 - 0.5 * cos(2 * Double.pi * Double(k) / Double(window - 1)))
        }
        return result
    }

    /// `librosa.filters.mel(sr=16000, n_fft=512, n_mels=bins, norm="slaney")` as float32.
    static func slaneyFilterbank(bins: Int) -> [Float] {
        let fSp = 200.0 / 3
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logStep = Foundation.log(6.4) / 27
        func hzToMel(_ f: Double) -> Double {
            f >= minLogHz ? minLogMel + Foundation.log(f / minLogHz) / logStep : f / fSp
        }
        func melToHz(_ m: Double) -> Double {
            m >= minLogMel ? minLogHz * Foundation.exp(logStep * (m - minLogMel)) : fSp * m
        }
        let frequencies = fftSize / 2 + 1
        let start = hzToMel(0)
        let stop = hzToMel(Double(sampleRate) / 2)
        let step = (stop - start) / Double(bins + 1)
        var melF = (0 ..< bins + 2).map { melToHz(Double($0) * step + start) }
        melF[bins + 1] = melToHz(stop)
        let fftFreqs = (0 ..< frequencies).map { Double($0) * Double(sampleRate) / Double(fftSize) }
        var weights = [Float](repeating: 0, count: bins * frequencies)
        for i in 0 ..< bins {
            let lowerWidth = melF[i + 1] - melF[i]
            let upperWidth = melF[i + 2] - melF[i + 1]
            let norm = 2 / (melF[i + 2] - melF[i])
            for k in 0 ..< frequencies {
                let lower = -(melF[i] - fftFreqs[k]) / lowerWidth
                let upper = (melF[i + 2] - fftFreqs[k]) / upperWidth
                let base = Float(max(0, min(lower, upper)))
                weights[i * frequencies + k] = Float(Double(base) * norm)
            }
        }
        return weights
    }
}

/// 8x time subsampling: a 3x3 stride-2 conv, then two depthwise-separable stride-2 convs.
final class D1OmniSubsampling: Module {
    @ModuleInfo var layers: [UnaryLayer]
    @ModuleInfo var linear: Linear

    init(_ config: D1OmniConfiguration.Audio) {
        let c = config.subsamplingChannels
        _layers.wrappedValue = [
            Conv2d(inputChannels: 1, outputChannels: c, kernelSize: 3, stride: 2, padding: 1),
            ReLU(),
            Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1, groups: c
            ),
            Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1),
            ReLU(),
            Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1, groups: c
            ),
            Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1),
            ReLU(),
        ]
        var frequencies = config.melBins
        for _ in 0 ..< 3 { frequencies = (frequencies - 1) / 2 + 1 }
        _linear.wrappedValue = Linear(c * frequencies, config.modelSize)
    }

    /// `features` [1, T, F] -> ([1, T', D], valid T').
    func callAsFunction(_ features: MLXArray, valid: Int) -> (MLXArray, Int) {
        var x = features[.ellipsis, .newAxis]
        var length = valid
        for layer in layers {
            x = layer(x)
            if let conv = layer as? Conv2d {
                if conv.stride.0 != 1 { length = (length - 1) / 2 + 1 }
                let mask = MLXArray(Int32(0) ..< Int32(x.dim(1))) .< Int32(length)
                x = x * mask.asType(x.dtype)[.newAxis, 0..., .newAxis, .newAxis]
            }
        }
        let (b, t) = (x.dim(0), x.dim(1))
        return (linear(x.transposed(0, 1, 3, 2).reshaped(b, t, -1)), length)
    }
}

final class D1OmniFeedForward: Module, UnaryLayer {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(_ size: Int, _ hidden: Int) {
        _linear1.wrappedValue = Linear(size, hidden)
        _linear2.wrappedValue = Linear(hidden, size)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// Transformer-XL relative-position attention with per-layer biases u and v.
final class D1OmniRelativeAttention: Module {
    let heads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ParameterInfo(key: "bias_u") var biasU: MLXArray
    @ParameterInfo(key: "bias_v") var biasV: MLXArray

    init(size: Int, heads: Int) {
        self.heads = heads
        headDim = size / heads
        _qProj.wrappedValue = Linear(size, size)
        _kProj.wrappedValue = Linear(size, size)
        _vProj.wrappedValue = Linear(size, size)
        _oProj.wrappedValue = Linear(size, size)
        _relativeKProj.wrappedValue = Linear(size, size, bias: false)
        _biasU.wrappedValue = MLXArray.zeros([heads, size / heads])
        _biasV.wrappedValue = MLXArray.zeros([heads, size / heads])
    }

    /// `x` [B, T, D], `positions` [1, 2T - 1, D].
    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(b, t, heads, headDim)
        let k = kProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let p = relativeKProj(positions).reshaped(1, -1, heads, headDim).transposed(0, 2, 1, 3)
        let ac = matmul((q + biasU).transposed(0, 2, 1, 3), k.transposed(0, 1, 3, 2))
        var bd = matmul((q + biasV).transposed(0, 2, 1, 3), p.transposed(0, 1, 3, 2))
        let span = bd.dim(-1)
        bd = padded(bd, widths: [0, 0, 0, [1, 0]])
        bd = bd.reshaped(b, heads, span + 1, t)[0..., 0..., 1...].reshaped(b, heads, t, span)
        let scores = (ac + bd[.ellipsis, ..<t]) / Float(Double(headDim).squareRoot())
        let attention = softmax(scores.asType(.float32), axis: -1).asType(v.dtype)
        let y = matmul(attention, v).transposed(0, 2, 1, 3).reshaped(b, t, -1)
        return oProj(y)
    }
}

final class D1OmniConvolutionModule: Module {
    let padding: Int

    @ModuleInfo(key: "pointwise_conv1") var pointwise1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo var norm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwise2: Conv1d

    init(size: Int, kernel: Int) {
        padding = (kernel - 1) / 2
        _pointwise1.wrappedValue = Conv1d(
            inputChannels: size, outputChannels: 2 * size, kernelSize: 1)
        _depthwise.wrappedValue = Conv1d(
            inputChannels: size, outputChannels: size, kernelSize: kernel, padding: padding,
            groups: size)
        _norm.wrappedValue = BatchNorm(featureCount: size)
        _pointwise2.wrappedValue = Conv1d(inputChannels: size, outputChannels: size, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = pointwise1(x)
        let half = h.dim(-1) / 2
        let gated = h[.ellipsis, ..<half] * sigmoid(h[.ellipsis, half...])
        return pointwise2(silu(norm(depthwise(gated))))
    }
}

final class D1OmniConformerLayer: Module {
    @ModuleInfo(key: "feed_forward1") var feedForward1: D1OmniFeedForward
    @ModuleInfo(key: "self_attn") var attention: D1OmniRelativeAttention
    @ModuleInfo var conv: D1OmniConvolutionModule
    @ModuleInfo(key: "feed_forward2") var feedForward2: D1OmniFeedForward
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "norm_self_att") var normAttention: LayerNorm
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(_ config: D1OmniConfiguration.Audio) {
        let d = config.modelSize
        let hidden = d * config.feedForwardExpansion
        _feedForward1.wrappedValue = D1OmniFeedForward(d, hidden)
        _attention.wrappedValue = D1OmniRelativeAttention(size: d, heads: config.heads)
        _conv.wrappedValue = D1OmniConvolutionModule(size: d, kernel: config.convolutionKernel)
        _feedForward2.wrappedValue = D1OmniFeedForward(d, hidden)
        _normFeedForward1.wrappedValue = LayerNorm(dimensions: d)
        _normAttention.wrappedValue = LayerNorm(dimensions: d)
        _normConv.wrappedValue = LayerNorm(dimensions: d)
        _normFeedForward2.wrappedValue = LayerNorm(dimensions: d)
        _normOut.wrappedValue = LayerNorm(dimensions: d)
    }

    func callAsFunction(_ input: MLXArray, positions: MLXArray) -> MLXArray {
        var x = input + feedForward1(normFeedForward1(input)) * 0.5
        x = x + attention(normAttention(x), positions: positions)
        x = x + conv(normConv(x))
        x = x + feedForward2(normFeedForward2(x)) * 0.5
        return normOut(x)
    }
}

final class D1OmniConformer: Module {
    let size: Int

    @ModuleInfo var subsampling: D1OmniSubsampling
    @ModuleInfo var layers: [D1OmniConformerLayer]

    init(_ config: D1OmniConfiguration.Audio) {
        size = config.modelSize
        _subsampling.wrappedValue = D1OmniSubsampling(config)
        _layers.wrappedValue = (0 ..< config.layers).map { _ in D1OmniConformerLayer(config) }
    }

    /// Sinusoids for relative positions `t - 1` down to `-(t - 1)`, as NeMo builds them.
    func positions(_ t: Int, dtype: DType) -> MLXArray {
        let relative = MLXArray(stride(from: t - 1, through: -(t - 1), by: -1).map { Float($0) })
        let exponents = MLXArray(stride(from: 0, to: size, by: 2).map { Float($0) })
        let div = exp(exponents * Float(-(Foundation.log(10000.0) / Double(size))))
        let angles = relative[0..., .newAxis] * div[.newAxis]
        let pe = stacked([sin(angles), cos(angles)], axis: -1).reshaped(2 * t - 1, size)
        return pe[.newAxis].asType(dtype)
    }

    /// `features` [1, T, F] -> the valid encoder frames [1, T', D].
    func callAsFunction(_ features: MLXArray, valid: Int) -> MLXArray {
        let (encoded, length) = subsampling(features, valid: valid)
        // Valid frames never read padded ones, so the encoder runs on the valid span alone.
        var x = encoded[0..., ..<length]
        let positions = positions(length, dtype: x.dtype)
        for layer in layers {
            x = layer(x, positions: positions)
        }
        return x
    }
}

final class D1OmniAudioAdapter: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(input: Int, output: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: input)
        _linear1.wrappedValue = Linear(input, output)
        _linear2.wrappedValue = Linear(output, output)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(norm(x)))) }
}

/// `x + up(GELU(down(LN(x))))`, trained with the trunk to fit the adapter's output to it.
final class D1OmniAudioResidual: Module {
    @ModuleInfo var ln: LayerNorm
    @ModuleInfo var down: Linear
    @ModuleInfo var up: Linear

    init(size: Int, width: Int) {
        _ln.wrappedValue = LayerNorm(dimensions: size)
        _down.wrappedValue = Linear(size, width)
        _up.wrappedValue = Linear(width, size)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x + up(gelu(down(ln(x)))) }
}

final class D1OmniAudio: Module {
    let bins: Int

    @ModuleInfo var encoder: D1OmniConformer
    @ModuleInfo var adapter: D1OmniAudioAdapter
    @ModuleInfo var residual: D1OmniAudioResidual

    init(_ config: D1OmniConfiguration) {
        bins = config.audio.melBins
        _encoder.wrappedValue = D1OmniConformer(config.audio)
        _adapter.wrappedValue = D1OmniAudioAdapter(
            input: config.audio.modelSize, output: config.text.hiddenSize)
        _residual.wrappedValue = D1OmniAudioResidual(
            size: config.text.hiddenSize, width: config.audio.residualWidth)
    }

    /// One clip of mono 16 kHz samples -> prefix embeddings [P, D], one per 80 ms.
    func callAsFunction(_ samples: [Float]) -> MLXArray {
        let (features, frames) = D1OmniMelFrontend.features(
            D1OmniMelFrontend.clip(samples), bins: bins)
        let dtype = adapter.norm.weight?.dtype ?? .float32
        let x = encoder(features[.newAxis].asType(dtype), valid: frames)
        return residual(adapter(x))[0]
    }
}
