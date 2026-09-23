//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4
    var mtpNumHiddenLayers: Int = 0
    var mtpUseDedicatedEmbeddings: Bool = false

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
        case mtpUseDedicatedEmbeddings = "mtp_use_dedicated_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        self.mtpNumHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0
        self.mtpUseDedicatedEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .mtpUseDedicatedEmbeddings) ?? false

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }
}

// MARK: - GatedDeltaNet

final class Qwen35GatedDeltaNet: Module {
    typealias SpeculativeCheckpoint = (
        index: Int, conv: MLXArray, recurrent: MLXArray
    )
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    // Inference-only physical projection. The four registered modules remain
    // as views so checkpoint, adapter, and parameter paths do not change.
    private let fusedInputProjection = FusedQuantizedLinearProjectionCache()
    var fusedInputProjectionEnabled = qwen35FourGDNEnabled
    private var recoverLoRATargets = [String: Edge0Qwen35RecoverLoRATarget]()

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        let inputProjectionPrefixes = [
            "in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a.",
        ]
        let replacesInputProjection = parameters.flattened().contains { key, _ in
            inputProjectionPrefixes.contains(where: key.hasPrefix)
        }
        defer {
            // Parameter updates are incremental and can throw after changing an
            // earlier tensor. Invalidate on both success and failure so a stale
            // physical projection can never remain published.
            if replacesInputProjection {
                fusedInputProjection.invalidate()
            }
        }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        let replacesInputProjection =
            key == "in_proj_qkv" || key == "in_proj_z"
            || key == "in_proj_b" || key == "in_proj_a"
        defer {
            // This is conservative when the setter itself rejects the value,
            // and necessary when a bulk update changed an earlier key first.
            if replacesInputProjection {
                fusedInputProjection.invalidate()
            }
        }
        try super.updateModule(key: key, value)
    }

    var hasFusedInputProjection: Bool { fusedInputProjection.isPrepared }

    /// Build one physical quantized projection while retaining the four named
    /// module paths as storage-sharing views. This runs at most once between
    /// parameter/module updates; failed eligibility checks are not repeated on
    /// every token. The model loader calls this before publishing the model;
    /// forward passes never invoke it.
    @discardableResult
    func prepareFusedInputProjection() throws -> Bool {
        try fusedInputProjection.prepare(
            enabled: fusedInputProjectionEnabled,
            linears: [
                inProjQKV, inProjZ, inProjB, inProjA,
            ]
        ) { sourceViews in
            try update(
                modules: ModuleChildren(values: [
                    "in_proj_qkv": .value(sourceViews[0]),
                    "in_proj_z": .value(sourceViews[1]),
                    "in_proj_b": .value(sourceViews[2]),
                    "in_proj_a": .value(sourceViews[3]),
                ]), verify: [])
        }
    }

    func projectInputs(_ inputs: MLXArray, batch: Int, sequence: Int) -> (
        qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
    ) {
        guard recoverLoRATargets.isEmpty,
            fusedInputProjectionEnabled, let fusedInProj = fusedInputProjection.fused
        else {
            return (
                recoverProjection(inProjQKV, input: inputs, target: "in_proj_qkv"),
                recoverProjection(inProjZ, input: inputs, target: "in_proj_z").reshaped(
                    batch, sequence, numVHeads, headVDim),
                recoverProjection(inProjB, input: inputs, target: "in_proj_b"),
                recoverProjection(inProjA, input: inputs, target: "in_proj_a")
            )
        }

        let projected = fusedInProj(inputs)
        let qkvEnd = keyDim * 2 + valueDim
        let zEnd = qkvEnd + valueDim
        let bEnd = zEnd + numVHeads
        let aEnd = bEnd + numVHeads
        return (
            projected[0..., 0..., ..<qkvEnd],
            projected[0..., 0..., qkvEnd ..< zEnd].reshaped(
                batch, sequence, numVHeads, headVDim),
            projected[0..., 0..., zEnd ..< bEnd],
            projected[0..., 0..., bEnd ..< aEnd]
        )
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil,
        checkpointAfter: Int? = nil,
        checkpointIndices: [Int] = []
    ) -> MLXArray {
        let convState =
            cache?[0] ?? zeroStates(batch: inputs.dim(0), dtype: inputs.dtype).conv
        let (out, newConvState, newRecState, checkpoint) = forward(
            inputs, convState: convState, recState: cache?[1], mask: mask,
            checkpointAfter: checkpointAfter, checkpointIndices: checkpointIndices)
        if let cache {
            cache[0] = newConvState
            cache[1] = newRecState
            for checkpoint in checkpoint {
                cache.saveSpeculativeCheckpoint(
                    convState: checkpoint.conv, recurrentState: checkpoint.recurrent,
                    advancedBy: checkpoint.index,
                    rewinding: inputs.dim(1) - checkpoint.index)
            }
            cache.advance(inputs.dim(1))
        }
        return out
    }

    /// Zero conv/recurrent state — the shapes `callAsFunction` and
    /// `gatedDeltaUpdate` otherwise build implicitly, made explicit for the
    /// traced decode path.
    func zeroStates(batch: Int, dtype: DType) -> (conv: MLXArray, rec: MLXArray) {
        (
            MLXArray.zeros([batch, convKernelSize - 1, convDim], dtype: dtype),
            MLXArray.zeros([batch, numVHeads, headVDim, headKDim], dtype: .float32)
        )
    }

    /// The GDN body with state passed explicitly in and out so it can be
    /// traced.
    func forward(
        _ x: MLXArray,
        convState: MLXArray,
        recState: MLXArray?,
        mask: MLXArray?,
        checkpointAfter: Int? = nil,
        checkpointIndices: [Int] = [],
        fusedCheckpoint: Bool = false
    ) -> (
        output: MLXArray,
        convState: MLXArray,
        recurrentState: MLXArray,
        checkpoints: [SpeculativeCheckpoint]
    ) {
        let B = x.dim(0)
        let S = x.dim(1)

        var (qkv, z, b, a) = projectInputs(x, batch: B, sequence: S)

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let fusedDecode =
            S == 1 && mask == nil && (qkv.dtype == .float16 || qkv.dtype == .bfloat16)
        let (convPre, newConvState) =
            fusedDecode
            ? decodeConv(convState: convState, qkv: qkv)
            : generalConv(convState: convState, qkv: qkv)
        let convOut = silu(convPre)

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let checkpointIndices = Array(
            Set(checkpointIndices + (checkpointAfter.map { [$0] } ?? []))
        ).filter { $0 > 0 && $0 < S }.sorted()

        func checkpointConv(at split: Int) -> MLXArray {
            guard convKernelSize > 1 else {
                return MLXArray.zeros([B, 0, convDim], dtype: qkv.dtype)
            }
            let convInput = concatenated([convState, qkv], axis: 1)
            return contiguous(convInput[0..., split ..< (split + convKernelSize - 1), 0...])
        }

        let out: MLXArray
        let newRecState: MLXArray
        let checkpoints: [SpeculativeCheckpoint]
        if S == 4, checkpointIndices == [1, 2, 3], mask == nil {
            let prefixStates: [MLXArray]
            (out, newRecState, prefixStates) = gatedDeltaUpdateFourCheckpoints(
                q: qNormed, k: kNormed, v: v, a: a, b: b,
                aLog: aLog, dtBias: dtBias, state: recState)
            checkpoints = zip(checkpointIndices, prefixStates).map { split, recurrent in
                (split, checkpointConv(at: split), recurrent)
            }
        } else if checkpointIndices.count == 1 {
            let split = checkpointIndices[0]
            let prefixState: MLXArray
            if fusedCheckpoint, S == 2, split == 1, mask == nil, headKDim % 32 == 0 {
                (out, newRecState, prefixState) = gatedDeltaUpdateCheckpoint(
                    q: qNormed, k: kNormed, v: v, a: a, b: b,
                    aLog: aLog, dtBias: dtBias, state: recState)
            } else {
                let prefixMask = mask.map { $0[0..., ..<split] }
                let suffixMask = mask.map { $0[0..., split...] }
                let (prefixOut, intermediateState) = gatedDeltaUpdate(
                    q: qNormed[0..., ..<split, 0..., 0...],
                    k: kNormed[0..., ..<split, 0..., 0...],
                    v: v[0..., ..<split, 0..., 0...],
                    a: a[0..., ..<split, 0...],
                    b: b[0..., ..<split, 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: recState,
                    mask: prefixMask)
                let (suffixOut, suffixState) = gatedDeltaUpdate(
                    q: qNormed[0..., split..., 0..., 0...],
                    k: kNormed[0..., split..., 0..., 0...],
                    v: v[0..., split..., 0..., 0...],
                    a: a[0..., split..., 0...],
                    b: b[0..., split..., 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: intermediateState,
                    mask: suffixMask)
                out = concatenated([prefixOut, suffixOut], axis: 1)
                newRecState = suffixState
                prefixState = intermediateState
            }

            checkpoints = [(split, checkpointConv(at: split), prefixState)]
        } else {
            (out, newRecState) = gatedDeltaUpdate(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: recState,
                mask: mask)
            checkpoints = checkpointIndices.map { split in
                let (_, recurrent) = gatedDeltaUpdate(
                    q: qNormed[0..., ..<split, 0..., 0...],
                    k: kNormed[0..., ..<split, 0..., 0...],
                    v: v[0..., ..<split, 0..., 0...],
                    a: a[0..., ..<split, 0...],
                    b: b[0..., ..<split, 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: recState,
                    mask: mask.map { $0[0..., ..<split] })
                return (split, checkpointConv(at: split), recurrent)
            }
        }

        let gated = norm(out, gate: z)
        let outputInput = gated.reshaped(B, S, -1)
        return (
            recoverProjection(outProj, input: outputInput, target: "out_proj"),
            newConvState, newRecState, checkpoints
        )
    }

    func attachRecoverLoRA(_ targets: [String: Edge0Qwen35RecoverLoRATarget]) {
        recoverLoRATargets = targets
        fusedInputProjection.invalidate()
    }

    private func recoverProjection(_ projection: Linear, input: MLXArray, target: String)
        -> MLXArray
    {
        let output = projection(input)
        guard let adapter = recoverLoRATargets[target] else { return output }
        return adapter.applying(to: output, input: input)
    }

    /// The S == 1 depthwise conv as elementwise multiply-adds, so `compile`
    /// folds it into the surrounding segment. f32 accumulation with a single
    /// final round matches `generalConv`'s `Convolution` kernel bit-for-bit
    /// for f16/bf16 (pinned by `Qwen35GDNDecodeBitwiseTests`); the kernel's
    /// own f32 accumulation orders differently, so f32 input stays on
    /// `generalConv`.
    func decodeConv(
        convState: MLXArray, qkv: MLXArray
    ) -> (conv: MLXArray, state: MLXArray) {
        var acc =
            convState[0..., 0, 0...].asType(.float32)
            * conv1d.weight[0..., 0, 0].asType(.float32)
        for tap in 1 ..< convKernelSize {
            let row =
                tap < convKernelSize - 1
                ? convState[0..., tap, 0...] : qkv[0..., 0, 0...]
            acc = acc + row.asType(.float32) * conv1d.weight[0..., tap, 0].asType(.float32)
        }
        return (
            acc.asType(qkv.dtype).reshaped(convState.dim(0), 1, convDim),
            concatenated([convState[0..., 1..., 0...], qkv], axis: 1)
        )
    }

    /// The sliding-window conv via MLX's `Convolution` kernel — the reference
    /// `decodeConv` is pinned against.
    func generalConv(
        convState: MLXArray, qkv: MLXArray
    ) -> (conv: MLXArray, state: MLXArray) {
        let convInput = concatenated([convState, qkv], axis: 1)
        return (
            conv1d(convInput),
            contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])
        )
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer
    private var recoverLoRATargets = [String: Edge0Qwen35RecoverLoRATarget]()

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        positionOffset: Int? = nil
    ) -> MLXArray {
        let (q, gate, k, values) = projectPreRope(x)

        let offset = positionOffset.map(RoPEOffset.scalar) ?? cache?.ropeOffset
        let queries = applyRotaryPosition(rope, to: q, offset: offset)
        let keys = applyRotaryPosition(rope, to: k, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )

        return mergeHeadsAndProject(attention: output, gate: gate)
    }

    /// Projections up to (not including) rope: `x` → (queries, gate, keys,
    /// values). Rope stays outside the traced decode path on purpose: its
    /// offset moves every token, and a trace would bake it in as a constant.
    func projectPreRope(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = recoverProjection(qProj, input: x, target: "q_proj")
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = recoverProjection(kProj, input: x, target: "k_proj")
        var values = recoverProjection(vProj, input: x, target: "v_proj")

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        return (queries, gate, keys, values)
    }

    /// Attention tail: head merge → output gate → output projection.
    func mergeHeadsAndProject(attention: MLXArray, gate: MLXArray) -> MLXArray {
        let merged =
            attention
            .transposed(0, 2, 1, 3)
            .reshaped(attention.dim(0), attention.dim(2), -1)
        let projectionInput = sigmoidMultiply(merged, gate)
        return recoverProjection(oProj, input: projectionInput, target: "o_proj")
    }

    func attachRecoverLoRA(_ targets: [String: Edge0Qwen35RecoverLoRATarget]) {
        recoverLoRATargets = targets
    }

    private func recoverProjection(_ projection: Linear, input: MLXArray, target: String)
        -> MLXArray
    {
        let output = projection(input)
        guard let adapter = recoverLoRATargets[target] else { return output }
        return adapter.applying(to: output, input: input)
    }
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") private var residentSwitchMLP: SwitchGLU?

    /// The registered resident expert layer. Streamed models deliberately have
    /// no parameters under `switch_mlp`; their weights live in the expert store.
    var switchMLP: SwitchGLU { residentSwitchMLP! }

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    private var streamedSwitchGLU: StreamedSwitchGLU?
    private var streamedExpertExecution:
        Qwen35StreamedExpertExecution<StreamedQuantizedExpertWeights>?
    private let streamedPlaceholder: Bool
    private let streamedInputDims: Int
    private let streamedHiddenDims: Int
    private var streamedResidentBytes = 0
    private var recoverLoRAEnabled = false

    init(_ args: Qwen35TextConfiguration, streamedExperts: Bool = false) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok
        self.streamedPlaceholder = streamedExperts
        self.streamedInputDims = args.hiddenSize
        self.streamedHiddenDims = args.moeIntermediateSize

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        if !streamedExperts {
            _residentSwitchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.numExperts
            )
        }
        self.streamedSwitchGLU = nil

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    var usesStreamedExperts: Bool { streamedExpertExecution != nil }
    var streamedAdditionalResidentWeightBytes: Int { streamedResidentBytes }

    func configureStreamedExperts(
        store: ExpertWeightStore<Int, StreamedQuantizedExpertWeights>, groupSize: Int,
        maximumResidentBytes: Int
    ) throws {
        guard streamedPlaceholder, streamedSwitchGLU == nil,
            streamedExpertExecution == nil, [32, 64].contains(groupSize),
            maximumResidentBytes >= 0
        else {
            throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
        }
        streamedSwitchGLU = StreamedSwitchGLU(
            inputDims: streamedInputDims, hiddenDims: streamedHiddenDims, groupSize: groupSize)
        streamedExpertExecution = Qwen35StreamedExpertExecution(store: store)
        streamedResidentBytes = maximumResidentBytes
    }

    @discardableResult
    func prepareFusedSharedExpertProjection() throws -> Bool {
        try sharedExpert.prepareFusedGateUpProjection()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Decode and two-token MTP verification run through a compiled trace.
        // Longer prefill stays unfused because it is GEMM-bound.
        if recoverLoRAEnabled || x.dim(1) > 2 {
            return forward(x)
        }
        return compiledForward(self, x)
    }

    /// The body stays inside this block, so the trace's default state (the
    /// block's own weights) is complete.
    private let compiledForward = CompiledTrace<Qwen35SparseMoeBlock> { block, arguments in
        [block.forward(arguments[0])]
    }

    /// The uncompiled body; an enclosing layer trace inlines it rather than
    /// nesting this block's own compiled wrapper.
    func forward(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let (inds, scores) = moeRouterTopK(
            gates, k: topK, normalize: normTopkProb)

        let tokenCount = x.size / x.dim(-1)
        let flatX = x.reshaped(tokenCount, x.dim(-1))
        let flatIndices = inds.reshaped(tokenCount, topK)
        let flatScores = scores.reshaped(tokenCount, topK)
        let combined = switchMLP.callAndWeightedReduce(
            flatX, flatIndices, weights: flatScores, fuseSortedReduction: true
        ).reshaped(x.shape)

        var sharedY = sharedExpert(x)
        let gateOutput = sharedExpertGate(x)
        sharedY = sigmoid(gateOutput) * sharedY

        return combined + sharedY
    }

    func attachRecoverLoRA(_ targets: [String: Edge0Qwen35RecoverLoRATarget]) {
        sharedExpert.attachRecoverLoRA(targets)
        recoverLoRAEnabled = !targets.isEmpty
    }

    /// Route, materialize and remap outside every compiled Qwen trace. The
    /// ordinary `SwitchGLU` owns no routed weight arrays in this configuration.
    nonisolated(nonsending) func streamedForward(_ x: MLXArray) async throws -> MLXArray {
        try await streamedForward(x, prediction: nil).output
    }

    nonisolated(nonsending) func streamedForward(
        _ x: MLXArray, prediction: Edge0Qwen35PrerouterPrediction?
    ) async throws -> (output: MLXArray, executedExpertIDs: MLXArray) {
        guard let execution = streamedExpertExecution, let streamedSwitchGLU else {
            throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
        }
        guard x.dim(0) == 1 else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "Streaming requires a batch of one")
        }
        let sequenceLength = x.dim(1)
        guard (1 ... 2).contains(sequenceLength) else {
            throw Qwen35StreamedExpertExecutionError.requiresResidentExperts(
                sequenceLength: sequenceLength)
        }

        let indices: MLXArray
        let scores: MLXArray
        if let prediction {
            guard prediction.expertIDs.shape == [1, sequenceLength, topK],
                prediction.scores.shape == prediction.expertIDs.shape
            else {
                throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
            }
            indices = prediction.expertIDs
            scores = prediction.scores
        } else {
            var gates = gate(x)
            gates = MLX.softmax(gates, axis: -1, precise: true)
            (indices, scores) = moeRouterTopK(
                gates, k: topK, normalize: normTopkProb)
        }
        // The IDs cross the MLX/I/O boundary; scores remain MLX values for the
        // exact weighted reduction after the selected slots are materialized.
        eval([indices, scores])
        let routerExpertIDs = indices.asArray(UInt32.self).map(Int.init)
        let plan = try await execution.prepareDecode(
            sequenceLength: sequenceLength, topK: topK, expertCount: numExperts,
            routerExpertIDs: routerExpertIDs)

        let flatX = x.reshaped(sequenceLength, x.dim(-1))
        let flatScores = scores.reshaped(sequenceLength, topK)
        let combined = try plan.withExpertWeights { weights in
            try streamedSwitchGLU.callAndWeightedReduce(
                flatX, compactRouterIDs: plan.compactRouterIDs, topK: topK,
                weights: flatScores, experts: weights.map { $0.withValue { $0 } })
        }.reshaped(x.shape)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY
        return (combined + sharedY, indices)
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module
    private var recoverLoRAEnabled = false

    init(
        _ args: Qwen35TextConfiguration, layerIdx: Int, forceFullAttention: Bool = false,
        streamedExperts: Bool = false
    ) {
        self.isLinear =
            forceFullAttention ? false : (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args, streamedExperts: streamedExperts)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    var usesStreamedExperts: Bool {
        (mlp as? Qwen35SparseMoeBlock)?.usesStreamedExperts ?? false
    }
    var streamedAdditionalResidentWeightBytes: Int {
        (mlp as? Qwen35SparseMoeBlock)?.streamedAdditionalResidentWeightBytes ?? 0
    }

    func configureStreamedExperts(
        store: ExpertWeightStore<Int, StreamedQuantizedExpertWeights>, groupSize: Int,
        maximumResidentBytes: Int
    ) throws {
        guard let moe = mlp as? Qwen35SparseMoeBlock else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "The selected Qwen layer is not routed")
        }
        try moe.configureStreamedExperts(
            store: store, groupSize: groupSize, maximumResidentBytes: maximumResidentBytes)
    }

    func attachRecoverLoRA(
        linearTargets: [String: Edge0Qwen35RecoverLoRATarget],
        attentionTargets: [String: Edge0Qwen35RecoverLoRATarget],
        sharedExpertTargets: [String: Edge0Qwen35RecoverLoRATarget]
    ) {
        linearAttn?.attachRecoverLoRA(linearTargets)
        selfAttn?.attachRecoverLoRA(attentionTargets)
        if let moe = mlp as? Qwen35SparseMoeBlock {
            moe.attachRecoverLoRA(sharedExpertTargets)
        }
        recoverLoRAEnabled =
            !linearTargets.isEmpty || !attentionTargets.isEmpty
            || !sharedExpertTargets.isEmpty
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        positionOffset: Int? = nil,
        checkpointAfter: Int? = nil,
        checkpointIndices: [Int] = []
    ) -> MLXArray {
        let verificationIndices = Array(
            Set(checkpointIndices + (checkpointAfter.map { [$0] } ?? []))
        ).sorted()
        let isCompiledVerificationShape =
            (x.dim(1) == 2 && verificationIndices == [1])
            || (x.dim(1) == 4 && verificationIndices == [1, 2, 3])
        if !recoverLoRAEnabled,
            compiledVerificationEnabled, isCompiledVerificationShape, ssmMask == nil,
            positionOffset == nil
        {
            if isLinear, let mambaCache = cache as? MambaCache,
                mambaCache[0] != nil, mambaCache[1] != nil
            {
                return verifyLinearLayer(
                    x, cache: mambaCache, checkpointIndices: verificationIndices)
            }
            if !isLinear, let cache, usesPlainAttentionCacheRoute(cache) {
                return decodeAttentionLayer(x, mask: attentionMask, cache: cache)
            }
        }
        // Single-token unmasked decode runs the layer as one traced function
        // (two for full attention, split at the KV write). Everything else
        // takes the general body below.
        if !recoverLoRAEnabled, x.dim(1) == 1, ssmMask == nil {
            if isLinear, let mambaCache = cache as? MambaCache {
                return decodeLinearLayer(x, cache: mambaCache)
            }
            if !isLinear, let cache, usesPlainAttentionCacheRoute(cache) {
                return decodeAttentionLayer(x, mask: attentionMask, cache: cache)
            }
        }

        let r: MLXArray
        if isLinear {
            r = linearAttn!(
                inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                checkpointAfter: checkpointAfter, checkpointIndices: checkpointIndices)
        } else {
            r = selfAttn!(
                inputLayerNorm(x), mask: attentionMask, cache: cache,
                positionOffset: positionOffset)
        }

        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    /// The stream boundary sits after each attention/GDN result and before
    /// routed experts. No compiled layer or decode-segment trace is entered.
    nonisolated(nonsending) func streamedForward(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?, cache: KVCache?
    ) async throws -> MLXArray {
        let r: MLXArray
        if isLinear {
            r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }
        let h = x + r
        let mlpInput = postAttentionLayerNorm(h)
        if let moe = mlp as? Qwen35SparseMoeBlock {
            return h + (try await moe.streamedForward(mlpInput))
        }
        return h + (mlp as! UnaryLayer)(mlpInput)
    }

    nonisolated(nonsending) func edge0StreamedForward(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?, cache: KVCache?,
        prediction: Edge0Qwen35PrerouterPrediction?
    ) async throws -> (hidden: MLXArray, mlpInput: MLXArray, executedExpertIDs: MLXArray) {
        let r: MLXArray
        if isLinear {
            r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }
        let h = x + r
        let mlpInput = postAttentionLayerNorm(h)
        guard let moe = mlp as? Qwen35SparseMoeBlock else {
            throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
        }
        let routed = try await moe.streamedForward(mlpInput, prediction: prediction)
        return (h + routed.output, mlpInput, routed.executedExpertIDs)
    }

    @discardableResult
    func prepareFusedMLPProjection() throws -> Bool {
        if let denseMLP = mlp as? Qwen3NextMLP {
            return try denseMLP.prepareFusedGateUpProjection()
        }
        if let sparseMLP = mlp as? Qwen35SparseMoeBlock {
            return try sparseMLP.prepareFusedSharedExpertProjection()
        }
        return false
    }

    // MARK: - Compiled decode blocks

    var compiledVerificationEnabled = true
    var fusedVerificationCheckpointEnabled = true
    var hasCompiledVerificationLayer: Bool {
        compiledVerificationLayer.isCompiled || compiledCheckpointVerificationLayer.isCompiled
            || compiledFourCheckpointVerificationLayer.isCompiled
    }

    private let compiledVerificationLayer = CompiledTrace<Qwen35DecoderLayer> { layer, args in
        layer.verificationLayerBody(args, fusedCheckpoint: false)
    }

    private let compiledCheckpointVerificationLayer = CompiledTrace<Qwen35DecoderLayer> {
        layer, args in
        layer.verificationLayerBody(args, fusedCheckpoint: true)
    }

    private let compiledFourCheckpointVerificationLayer = CompiledTrace<Qwen35DecoderLayer> {
        layer, args in
        layer.verificationLayerBody(args, checkpointIndices: [1, 2, 3])
    }

    private func verificationLayerBody(
        _ args: [MLXArray], fusedCheckpoint: Bool = false,
        checkpointIndices: [Int] = [1]
    ) -> [MLXArray] {
        let (r, conv, recurrent, checkpoints) = linearAttn!.forward(
            inputLayerNorm(args[0]), convState: args[1], recState: args[2],
            mask: nil, checkpointIndices: checkpointIndices,
            fusedCheckpoint: fusedCheckpoint)
        let h = args[0] + r
        return [h + mlpForward(postAttentionLayerNorm(h)), conv, recurrent]
            + checkpoints.flatMap { [$0.conv, $0.recurrent] }
    }

    // Keep the committed-token checkpoint explicit; cache mutation stays outside the trace.
    private func verifyLinearLayer(
        _ x: MLXArray, cache: MambaCache, checkpointIndices: [Int]
    ) -> MLXArray {
        let trace: CompiledTrace<Qwen35DecoderLayer>
        switch x.dim(1) {
        case 2:
            trace =
                fusedVerificationCheckpointEnabled
                ? compiledCheckpointVerificationLayer : compiledVerificationLayer
        case 4:
            trace = compiledFourCheckpointVerificationLayer
        default:
            preconditionFailure("Unsupported compiled verification width")
        }
        let out = trace(self, [x, cache[0]!, cache[1]!])
        cache[0] = out[1]
        cache[1] = out[2]
        for (position, index) in checkpointIndices.enumerated() {
            cache.saveSpeculativeCheckpoint(
                convState: out[3 + 2 * position], recurrentState: out[4 + 2 * position],
                advancedBy: index, rewinding: x.dim(1) - index)
        }
        cache.advance(x.dim(1))
        return out[0]
    }

    // Every body stays inside this layer, so each trace's default state (the
    // layer's own weights) is complete.
    private let compiledLinearLayer = CompiledTrace<Qwen35DecoderLayer> { layer, arguments in
        let (out, newConvState, newRecState) = layer.linearLayerBody(
            x: arguments[0], convState: arguments[1], recState: arguments[2])
        return [out, newConvState, newRecState]
    }

    private let compiledAttentionPre = CompiledTrace<Qwen35DecoderLayer> { layer, arguments in
        let (queries, gate, keys, values) = layer.attentionPreBody(x: arguments[0])
        return [queries, gate, keys, values]
    }

    private let compiledAttentionPost = CompiledTrace<Qwen35DecoderLayer> { layer, arguments in
        [
            layer.attentionPostBody(
                x: arguments[0], attention: arguments[1], gate: arguments[2])
        ]
    }

    /// GDN decode layer as one traced function. A compiled function must be
    /// pure, so conv/recurrent state crosses the boundary explicitly.
    private func decodeLinearLayer(_ x: MLXArray, cache: MambaCache) -> MLXArray {
        let zero = linearAttn!.zeroStates(batch: x.dim(0), dtype: x.dtype)
        let convState = cache[0] ?? zero.conv
        let recState = cache[1] ?? zero.rec

        let out = compiledLinearLayer(self, [x, convState, recState])
        cache[0] = out[1]
        cache[1] = out[2]
        cache.advance(1)
        return out[0]
    }

    /// Full-attention decode layer: two traced functions around the KV write,
    /// which cannot live inside a trace because the cache grows every token.
    private func decodeAttentionLayer(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache
    ) -> MLXArray {
        let projected = compiledAttentionPre(self, [x])
        let attention = attentionCacheStep(
            queries: projected[0], keys: projected[2], values: projected[3],
            cache: cache, mask: mask)
        return compiledAttentionPost(self, [x, attention, projected[1]])[0]
    }

    /// The part of a full-attention decode step that cannot be traced: rope
    /// (its offset moves every token), the KV write, and the SDPA over the
    /// grown cache.
    func attentionCacheStep(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        cache: KVCache, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let attn = selfAttn!
        let offset = cache.ropeOffset
        return attentionWithCacheUpdate(
            queries: applyRotaryPosition(attn.rope, to: queries, offset: offset),
            keys: applyRotaryPosition(attn.rope, to: keys, offset: offset),
            values: values,
            cache: cache,
            scale: attn.scale,
            mask: mask
        )
    }

    // MARK: - Layer bodies

    func linearLayerBody(x: MLXArray, convState: MLXArray, recState: MLXArray) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        let (r, newConvState, newRecState, _) = linearAttn!.forward(
            inputLayerNorm(x), convState: convState, recState: recState, mask: nil)
        let h = x + r
        return (h + mlpForward(postAttentionLayerNorm(h)), newConvState, newRecState)
    }

    func attentionPreBody(x: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        selfAttn!.projectPreRope(inputLayerNorm(x))
    }

    /// `x` is the layer input — the residual branch around the attention block.
    func attentionPostBody(x: MLXArray, attention: MLXArray, gate: MLXArray) -> MLXArray {
        let r = selfAttn!.mergeHeadsAndProject(attention: attention, gate: gate)
        let h = x + r
        return h + mlpForward(postAttentionLayerNorm(h))
    }

    private func mlpForward(_ x: MLXArray) -> MLXArray {
        if let moe = mlp as? Qwen35SparseMoeBlock {
            return moe.forward(x)
        }
        return (mlp as! UnaryLayer)(x)
    }
}

private final class Edge0Qwen35Runtime {
    let profile: Edge0Qwen35InferenceProfile
    let prerouter: Edge0Qwen35Prerouter
    let state: Edge0Qwen35InferenceRequestState
    let phaseController = Edge0Qwen35PhaseController()

    init(profile: Edge0Qwen35InferenceProfile, prerouter: Edge0Qwen35Prerouter) throws {
        self.profile = profile
        self.prerouter = prerouter
        self.state = try Edge0Qwen35InferenceRequestState(profile: profile)
    }
}

private func profilePrediction(
    _ runtime: Edge0Qwen35Runtime, forConsumer layer: Int
) -> Edge0Qwen35PrerouterPrediction? {
    guard runtime.profile.predictedConsumers.contains(layer) else { return nil }
    return try? runtime.state.prediction(forConsumer: layer)
}

// MARK: - Text Model

public class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int
    private var recoverLoRAEnabled = false
    private let edge0Configuration: Qwen35TextConfiguration

    init(_ args: Qwen35TextConfiguration, streamedExperts: Bool = false) {
        precondition(args.vocabularySize > 0)
        self.edge0Configuration = args

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        let layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx, streamedExperts: streamedExperts)
        }
        self.layers = layers

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        let segments = CompiledDecodeSegment.schedule(
            linearLayers: layers.map(\.isLinear))
        self.decodeSegments = segments
        self.compiledSegments = CompiledDecodeSegmentCache(
            count: segments.count,
            state: { model, index in
                // Everything `segmentBody` reads: the layers it runs, the
                // embedding it starts from, the final norm it ends with.
                var modules: [Module] = segments[index].layerIndices.map { model.layers[$0] }
                if index == 0 {
                    modules.append(model.embedTokens)
                }
                if index == segments.count - 1 {
                    modules.append(model.norm)
                }
                return modules
            },
            body: { model, index, arguments in
                model.segmentBody(at: index, arguments)
            })
        super.init()
    }

    var usesStreamedExperts: Bool { layers.contains { $0.usesStreamedExperts } }
    private var streamedResidentBytes = 0
    private var edge0Runtime: Edge0Qwen35Runtime?
    var streamedAdditionalResidentWeightBytes: Int { streamedResidentBytes }
    var usesEdge0: Bool { edge0Runtime != nil }

    func configureStreamedExperts(
        stores: [ExpertWeightStore<Int, StreamedQuantizedExpertWeights>], groupSize: Int,
        maximumResidentBytesPerLayer: Int
    ) throws {
        guard stores.count == layers.count, maximumResidentBytesPerLayer >= 0 else {
            throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
        }
        for (layer, store) in zip(layers, stores) {
            try layer.configureStreamedExperts(
                store: store, groupSize: groupSize,
                maximumResidentBytes: maximumResidentBytesPerLayer)
        }
        let (bytes, overflow) = layers.reduce(into: (0, false)) { total, layer in
            let (next, additionOverflow) = total.0.addingReportingOverflow(
                layer.streamedAdditionalResidentWeightBytes)
            total = (next, total.1 || additionOverflow)
        }
        guard !overflow else {
            throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
        }
        streamedResidentBytes = bytes
    }

    func configureEdge0(
        profile: Edge0Qwen35InferenceProfile,
        prerouter: Edge0Qwen35Prerouter
    ) throws {
        try profile.validate()
        try profile.validateAttachment(prerouter)
        guard usesStreamedExperts, layers.count == 40,
            layers.allSatisfy({ ($0.mlp as? Qwen35SparseMoeBlock)?.topK == profile.topK })
        else {
            throw Edge0Qwen35InferenceProfileError.invalidProfile
        }
        edge0Runtime = try Edge0Qwen35Runtime(profile: profile, prerouter: prerouter)
    }

    func beginEdge0Request() {
        beginEdge0Request(promptTokenCount: 0)
    }

    func beginEdge0Request(promptTokenCount: Int) {
        edge0Runtime?.state.resetForNewRequest()
        edge0Runtime?.phaseController.begin(promptTokenCount: promptTokenCount)
    }

    func endEdge0Request() {
        edge0Runtime?.state.resetForNewRequest()
        edge0Runtime?.phaseController.reset()
    }

    func edge0PrefillPhase(forwardTokenCount: Int) -> Edge0Qwen35ForwardPhase {
        edge0Runtime?.phaseController.prefillPhase(forwardTokenCount: forwardTokenCount)
            ?? .prefillRealRouter
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        forward(inputs, cache: cache, applyFinalNorm: true)
    }

    /// Backbone hidden states with optional final normalization.
    ///
    /// MTP state emission needs access to both the residual and the final
    /// normalized representation. The paired Qwen MTP head consumes the same
    /// post-final-norm hidden representation used by the target LM head.
    func forward(
        _ inputs: MLXArray,
        cache: [KVCache?]? = nil,
        applyFinalNorm: Bool,
        checkpointAfter: Int? = nil,
        checkpointIndices: [Int] = []
    ) -> MLXArray {
        forwardCapturing(
            inputs, cache: cache, applyFinalNorm: applyFinalNorm,
            checkpointAfter: checkpointAfter, checkpointIndices: checkpointIndices,
            layerIDs: []
        ).hidden
    }

    func forwardCapturing(
        _ inputs: MLXArray,
        cache: [KVCache?]? = nil,
        applyFinalNorm: Bool,
        checkpointAfter: Int? = nil,
        checkpointIndices: [Int] = [],
        layerIDs: [Int]
    ) -> (hidden: MLXArray, captures: [MLXArray]) {
        precondition(
            Set(layerIDs).count == layerIDs.count
                && layerIDs.allSatisfy { layers.indices.contains($0) },
            "Qwen 3.5 hidden capture layer IDs must be unique and in range")
        if layerIDs.isEmpty, !recoverLoRAEnabled,
            applyFinalNorm, inputs.dim(1) == 1, let caches = cache,
            let step = decodeStep(inputs, caches)
        {
            return (step, [])
        }

        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        let requested = Set(layerIDs)
        var captured = [Int: MLXArray]()
        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i],
                checkpointAfter: checkpointAfter, checkpointIndices: checkpointIndices)
            if requested.contains(i) { captured[i] = hiddenStates }
        }

        return (
            applyFinalNorm ? norm(hiddenStates) : hiddenStates,
            layerIDs.compactMap { captured[$0] }
        )
    }

    func attachRecoverLoRA(_ adapter: Edge0Qwen35RecoverLoRA) throws {
        guard configurationIsCompatibleWithEdge0RecoverLoRA else {
            throw Edge0Qwen35RecoverLoRAAttachmentError.incompatibleModel
        }
        let expectedPaths = Set(Edge0Qwen35RecoverLoRA.expectedTargetPaths)
        let actualPaths = Set(adapter.targetPaths)
        guard actualPaths == expectedPaths else {
            throw Edge0Qwen35RecoverLoRAAttachmentError.targetPathsMismatch(
                expected: expectedPaths.count, actual: actualPaths.count)
        }

        for (index, layer) in layers.enumerated() {
            let prefix = "language_model.model.layers.\(index)."
            let linearTargets = adapter.adapters(for: [
                prefix + "linear_attn.in_proj_qkv",
                prefix + "linear_attn.in_proj_z",
                prefix + "linear_attn.in_proj_b",
                prefix + "linear_attn.in_proj_a",
                prefix + "linear_attn.out_proj",
            ]).reduce(into: [String: Edge0Qwen35RecoverLoRATarget]()) { targets, entry in
                targets[String(entry.key.dropFirst(prefix.count + "linear_attn.".count))] =
                    entry.value
            }
            let attentionTargets = adapter.adapters(for: [
                prefix + "self_attn.q_proj",
                prefix + "self_attn.k_proj",
                prefix + "self_attn.v_proj",
                prefix + "self_attn.o_proj",
            ]).reduce(into: [String: Edge0Qwen35RecoverLoRATarget]()) { targets, entry in
                targets[String(entry.key.dropFirst(prefix.count + "self_attn.".count))] =
                    entry.value
            }
            let sharedExpertTargets = adapter.adapters(for: [
                prefix + "mlp.shared_expert.gate_proj",
                prefix + "mlp.shared_expert.up_proj",
                prefix + "mlp.shared_expert.down_proj",
            ]).reduce(into: [String: Edge0Qwen35RecoverLoRATarget]()) { targets, entry in
                targets[String(entry.key.dropFirst(prefix.count + "mlp.shared_expert.".count))] =
                    entry.value
            }
            layer.attachRecoverLoRA(
                linearTargets: linearTargets, attentionTargets: attentionTargets,
                sharedExpertTargets: sharedExpertTargets)
        }
        recoverLoRAEnabled = true
        invalidateCompiledTraces()
    }

    private var configurationIsCompatibleWithEdge0RecoverLoRA: Bool {
        let attentionHeadDim = edge0Configuration.headDim ?? 0
        return layers.count == 40 && edge0Configuration.hiddenSize == 2048
            && edge0Configuration.fullAttentionInterval == 4
            && edge0Configuration.numExperts == 256 && edge0Configuration.numExpertsPerTok == 4
            && edge0Configuration.moeIntermediateSize == 512
            && edge0Configuration.linearNumValueHeads == 32
            && edge0Configuration.linearNumKeyHeads == 16
            && edge0Configuration.linearKeyHeadDim == 128
            && edge0Configuration.linearValueHeadDim == 128
            && edge0Configuration.linearConvKernelDim == 4
            && edge0Configuration.sharedExpertIntermediateSize == 512
            && edge0Configuration.attentionHeads == 16 && attentionHeadDim == 256
            && edge0Configuration.kvHeads == 2
    }

    nonisolated(nonsending) func streamedForward(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        edge0Phase: Edge0Qwen35ForwardPhase = .decode
    ) async throws -> MLXArray {
        guard usesStreamedExperts, inputs.dim(0) == 1, (1 ... 2).contains(inputs.dim(1)) else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "Streaming requires an installed routed Qwen model with one or two input tokens")
        }
        var hiddenStates = embedTokens(inputs)
        let cacheArray: [KVCache?]
        if let cache {
            cacheArray = cache.map { $0 }
        } else {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }
        guard cacheArray.count == layers.count else {
            throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
        }
        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray[ssmIdx] as? MambaCache)

        if let edge0Runtime {
            guard inputs.dim(1) == 1 else {
                throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                    "Edge0 routing requires one-token forwards")
            }
            if edge0Phase == .prefillRealRouter {
                for (index, layer) in layers.enumerated() {
                    let result = try await layer.edge0StreamedForward(
                        hiddenStates,
                        attentionMask: layer.isLinear ? .none : faMask,
                        ssmMask: layer.isLinear ? ssmMask : nil,
                        cache: cacheArray[index], prediction: nil)
                    hiddenStates = edge0Runtime.profile.stabilizedHidden(result.hidden)
                }
                return norm(hiddenStates)
            }
            var hiddenByOwner = [Int: MLXArray]()
            for (index, layer) in layers.enumerated() {
                let prediction = profilePrediction(edge0Runtime, forConsumer: index)
                let result = try await layer.edge0StreamedForward(
                    hiddenStates,
                    attentionMask: layer.isLinear ? .none : faMask,
                    ssmMask: layer.isLinear ? ssmMask : nil,
                    cache: cacheArray[index], prediction: prediction)
                try edge0Runtime.state.recordExecuted(result.executedExpertIDs, for: index)
                if edge0Runtime.profile.artifactOwners.contains(index) {
                    hiddenByOwner[index] = result.mlpInput
                }
                hiddenStates = edge0Runtime.profile.stabilizedHidden(result.hidden)
            }
            try edge0Runtime.state.stage(
                prerouter: edge0Runtime.prerouter, hiddenByOwner: hiddenByOwner)
            edge0Runtime.state.advance()
        } else {
            for (index, layer) in layers.enumerated() {
                hiddenStates = try await layer.streamedForward(
                    hiddenStates,
                    attentionMask: layer.isLinear ? .none : faMask,
                    ssmMask: layer.isLinear ? ssmMask : nil,
                    cache: cacheArray[index])
            }
        }
        return norm(hiddenStates)
    }

    // MARK: - Whole-step decode schedule

    /// One traced piece of a decode step: the tail of the previous
    /// full-attention layer, a run of GDN layers, then the head of the next
    /// one (whose SDPA runs between this segment and the next).
    private let decodeSegments: [CompiledDecodeSegment]
    private let compiledSegments: CompiledDecodeSegmentCache<Qwen35TextModelInner>
    var compiledDecodeSegmentCount: Int { compiledSegments.compiledCount }

    /// Flat argument/result lists because `compile` takes `[MLXArray]`.
    /// In: `[x]` (token ids for segment 0), then `[attention, gate]` when
    /// opening with a full-attention tail, then `[convState, recState]` per
    /// GDN layer. Out: `[x]`, then `[newConvState, newRecState]` per GDN
    /// layer, then `[queries, gate, keys, values]` when closing with a head.
    private func segmentBody(at index: Int, _ args: [MLXArray]) -> [MLXArray] {
        let segment = decodeSegments[index]
        var hiddenStates = index == 0 ? embedTokens(args[0]) : args[0]

        if let post = segment.attentionPostLayer {
            hiddenStates = layers[post].attentionPostBody(
                x: hiddenStates, attention: args[1], gate: args[2])
        }

        var states: [MLXArray] = []
        for (i, layerIndex) in segment.linearLayers.enumerated() {
            let slot = segment.stateInputOffset + 2 * i
            let (out, newConvState, newRecState) = layers[layerIndex].linearLayerBody(
                x: hiddenStates, convState: args[slot], recState: args[slot + 1])
            hiddenStates = out
            states.append(newConvState)
            states.append(newRecState)
        }

        if let pre = segment.attentionPreLayer {
            let (queries, gate, keys, values) = layers[pre].attentionPreBody(x: hiddenStates)
            // The next segment needs the attention layer's input for its residual.
            return [hiddenStates] + states + [queries, gate, keys, values]
        }

        if index == decodeSegments.count - 1 {
            hiddenStates = norm(hiddenStates)
        }
        return [hiddenStates] + states
    }

    /// One decode step through the compiled segments, or nil when this is not
    /// the plain single-token case the schedule assumes — the caller then
    /// takes the general path. Segments split at each KV write: the cache
    /// update's in-place slice_update cannot live inside a trace, and
    /// everything else on a decode step is static-shaped, so the segments
    /// compile concretely.
    private func decodeStep(_ inputs: MLXArray, _ cache: [KVCache?]) -> MLXArray? {
        guard cache.count == layers.count else { return nil }
        // The schedule is only valid when the masks the general path would
        // build both come out empty.
        if createSSMMask(h: inputs, cache: cache[ssmIdx] as? MambaCache) != nil { return nil }
        guard let faCache = cache[faIdx],
            case .none = createAttentionMask(h: inputs, cache: faCache)
        else { return nil }

        // Cache kinds can change mid-generation (`maybeQuantizeKVCache` swaps
        // array entries), so eligibility is re-checked every step.
        var mambaCaches = [MambaCache?](repeating: nil, count: layers.count)
        for (i, layer) in layers.enumerated() {
            if layer.isLinear {
                // No GDN state yet (a single-token prompt): the general path
                // builds the zero states.
                guard let mambaCache = cache[i] as? MambaCache, mambaCache[0] != nil,
                    mambaCache[1] != nil
                else { return nil }
                mambaCaches[i] = mambaCache
            } else {
                guard let kv = cache[i], usesPlainAttentionCacheRoute(kv) else { return nil }
            }
        }

        var carry = inputs
        var pendingAttention: [MLXArray] = []

        for (segmentIndex, segment) in decodeSegments.enumerated() {
            var args: [MLXArray] = [carry] + pendingAttention
            for layerIndex in segment.linearLayers {
                let mambaCache = mambaCaches[layerIndex]!
                args.append(mambaCache[0]!)
                args.append(mambaCache[1]!)
            }

            let outputs = compiledSegments(self, at: segmentIndex, args)

            carry = outputs[0]
            for (i, layerIndex) in segment.linearLayers.enumerated() {
                let mambaCache = mambaCaches[layerIndex]!
                mambaCache[0] = outputs[1 + 2 * i]
                mambaCache[1] = outputs[2 + 2 * i]
                mambaCache.advance(1)
            }

            pendingAttention = []
            if let pre = segment.attentionPreLayer {
                let head = segment.attentionOutputOffset
                let attention = layers[pre].attentionCacheStep(
                    queries: outputs[head], keys: outputs[head + 2],
                    values: outputs[head + 3], cache: cache[pre]!, mask: .none)
                pendingAttention = [attention, outputs[head + 1]]
            }
        }

        return carry
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration
    private let mixedPreservedNorms: Bool

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(
        _ args: Qwen35TextConfiguration, mixedPreservedNorms: Bool = false,
        streamedExperts: Bool = false
    ) {
        self.configuration = args
        self.mixedPreservedNorms = mixedPreservedNorms
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args, streamedExperts: streamedExperts)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    var usesStreamedExperts: Bool { model.usesStreamedExperts }
    var usesEdge0: Bool { model.usesEdge0 }
    var streamedExpertResidentWeightBytes: Int {
        model.streamedAdditionalResidentWeightBytes
    }

    func configureStreamedExperts(
        stores: [ExpertWeightStore<Int, StreamedQuantizedExpertWeights>], groupSize: Int,
        maximumResidentBytesPerLayer: Int
    ) throws {
        try model.configureStreamedExperts(
            stores: stores, groupSize: groupSize,
            maximumResidentBytesPerLayer: maximumResidentBytesPerLayer)
    }

    func configureEdge0(
        profile: Edge0Qwen35InferenceProfile,
        prerouter: Edge0Qwen35Prerouter
    ) throws {
        try model.configureEdge0(profile: profile, prerouter: prerouter)
    }

    func beginEdge0Request() {
        model.beginEdge0Request()
    }

    func beginEdge0Request(promptTokenCount: Int) {
        model.beginEdge0Request(promptTokenCount: promptTokenCount)
    }

    func endEdge0Request() {
        model.endEdge0Request()
    }

    nonisolated(nonsending) func streamedScheduledForward(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray {
        let hiddenStates = try await model.streamedForward(inputs, cache: cache)
        return projectLogits(hiddenStates)
    }

    /// Projects only the final hidden state of an ordinary prefill. The cache
    /// still receives every row; avoiding the earlier vocabulary projections is
    /// safe because scheduled generation samples only the final row.
    public nonisolated(nonsending) func scheduledFinalPrefillForward(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray {
        guard qwen35FinalPrefillLogitEnabled || usesEdge0 else {
            return try await scheduledForward(inputs, cache: cache)
        }
        let hiddenStates: MLXArray
        if usesStreamedExperts {
            let phase =
                usesEdge0
                ? model.edge0PrefillPhase(forwardTokenCount: inputs.dim(1))
                : .decode
            hiddenStates = try await model.streamedForward(
                inputs, cache: cache, edge0Phase: phase)
        } else {
            hiddenStates = model(inputs, cache: cache)
        }
        guard hiddenStates.dim(1) > 0 else {
            throw ConcurrentTextRuntimeError.invalidRequest
        }
        let last = hiddenStates[
            0..., (hiddenStates.dim(1) - 1) ..< hiddenStates.dim(1), 0...
        ]
        return projectLogits(last)
    }

    private func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let emitDrafterState = state?[mtpEmitFlagKey] ?? false
        let checkpointOnly = state?[speculativeCheckpointOnlyKey] ?? false
        let finalPrefillLogitsOnly =
            emitDrafterState && (state?[mtpFinalPrefillLogitsOnlyKey] ?? false)
        let hiddenStates: MLXArray
        let drafterHiddenStates: MLXArray
        if emitDrafterState || checkpointOnly {
            let captured = model.forwardCapturing(
                input.tokens, cache: cache, applyFinalNorm: false,
                checkpointAfter: state?[mtpCacheCheckpointIndexKey],
                checkpointIndices: state?[mtpCacheCheckpointIndicesKey] ?? [],
                layerIDs: state?[mtpHiddenLayerIDsKey] ?? [])
            hiddenStates = model.norm(captured.hidden)
            drafterHiddenStates =
                captured.captures.isEmpty
                ? hiddenStates : concatenated(captured.captures, axis: -1)
        } else {
            hiddenStates = model(input.tokens, cache: cache)
            drafterHiddenStates = hiddenStates
        }

        let logitsInput: MLXArray
        if finalPrefillLogitsOnly {
            logitsInput = hiddenStates[0..., (-1)..., 0...]
        } else {
            logitsInput = hiddenStates
        }
        let logits = projectLogits(logitsInput)

        guard emitDrafterState else {
            return LMOutput(logits: logits)
        }

        var outState = state ?? LMOutput.State()
        outState[mtpFinalPrefillLogitsOnlyKey] = nil
        outState[mtpLastHiddenStatesKey] = drafterHiddenStates
        outState[mtpSharedKVStatesKey] = qwen35SharedKVState(
            cache: cache, fullAttentionIndex: model.faIdx)
        outState[mtpSharedKVOffsetsKey] = qwen35SharedKVOffsets(
            cache: cache, fullAttentionIndex: model.faIdx)
        outState[mtpSharedKVSourceIndicesKey] = ["full_attention": model.faIdx]
        return LMOutput(logits: logits, state: outState)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            // Full-attention layers honor maxKVSize; GDN / linear layers keep
            // a fixed recurrent state that cannot be token-windowed.
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    public func prepare() throws {
        for layer in model.layers {
            if let linearAttn = layer.linearAttn {
                _ = try linearAttn.prepareFusedInputProjection()
            }
            _ = try layer.prepareFusedMLPProjection()
        }
    }

    /// Attaches Edge0's fixed Recover-LoRA sidecar without merging it into the base model.
    public func attachRecoverLoRA(_ adapter: Edge0Qwen35RecoverLoRA) throws {
        try model.attachRecoverLoRA(adapter)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        // Ordinary converted checkpoints advertise their norm convention through
        // convolution layout. JANG MXFP bundles are the typed exception: projections
        // and convolutions are converted while source `(1 + weight)` norms are retained.
        let shouldShiftNormWeights = hasUnsanitizedConv1d

        var weights = weights.filter { !$0.key.contains("mtp.") }

        weights = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            let isNorm = v.ndim == 1 && normKeys.contains(where: { k.hasSuffix($0) })
            if isNorm && (shouldShiftNormWeights || mixedPreservedNorms) {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }
}

private func qwen35SharedKVState(
    cache: [KVCache]?,
    fullAttentionIndex: Int
) -> [String: (MLXArray, MLXArray)] {
    guard let cache, fullAttentionIndex < cache.count else {
        return [:]
    }
    let state = cache[fullAttentionIndex].state
    guard state.count == 2 else {
        return [:]
    }
    return ["full_attention": (state[0], state[1])]
}

private func qwen35SharedKVOffsets(
    cache: [KVCache]?,
    fullAttentionIndex: Int
) -> [String: Int]? {
    guard let cache, fullAttentionIndex < cache.count else {
        return nil
    }
    return ["full_attention": cache[fullAttentionIndex].offset]
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

extension Qwen35TextModel: PromptLookupHybridModel {
    public var maximumNativeTargetCacheRewind: Int { 3 }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration, streamedExperts: Bool = false) {
        let textModel = Qwen35TextModel(
            args.textConfig, mixedPreservedNorms: args.mixedPreservedNorms,
            streamedExperts: streamedExperts)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    var usesStreamedExperts: Bool { languageModel.usesStreamedExperts }
    var usesEdge0: Bool { languageModel.usesEdge0 }
    var streamedExpertResidentWeightBytes: Int {
        languageModel.streamedExpertResidentWeightBytes
    }

    func configureStreamedExperts(
        stores: [ExpertWeightStore<Int, StreamedQuantizedExpertWeights>], groupSize: Int,
        maximumResidentBytesPerLayer: Int
    ) throws {
        try languageModel.configureStreamedExperts(
            stores: stores, groupSize: groupSize,
            maximumResidentBytesPerLayer: maximumResidentBytesPerLayer)
    }

    func configureEdge0(
        profile: Edge0Qwen35InferenceProfile,
        prerouter: Edge0Qwen35Prerouter
    ) throws {
        try languageModel.configureEdge0(profile: profile, prerouter: prerouter)
    }

    func beginEdge0Request() {
        languageModel.beginEdge0Request()
    }

    func beginEdge0Request(promptTokenCount: Int) {
        languageModel.beginEdge0Request(promptTokenCount: promptTokenCount)
    }

    func endEdge0Request() {
        languageModel.endEdge0Request()
    }

    nonisolated(nonsending) func streamedScheduledForward(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray {
        try await languageModel.streamedScheduledForward(inputs, cache: cache)
    }

    public nonisolated(nonsending) func scheduledFinalPrefillForward(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray {
        try await languageModel.scheduledFinalPrefillForward(inputs, cache: cache)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        languageModel(input, cache: cache, state: state)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func prepare() throws {
        try languageModel.prepare()
    }

    /// Attaches Edge0's fixed Recover-LoRA sidecar without merging it into the base model.
    public func attachRecoverLoRA(_ adapter: Edge0Qwen35RecoverLoRA) throws {
        try languageModel.attachRecoverLoRA(adapter)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

extension Qwen35Model: PromptLookupHybridModel {
    public var maximumNativeTargetCacheRewind: Int { 3 }
}

// MARK: - Chat conventions

// `Qwen35MoEModel` subclasses `Qwen35Model` and inherits both declarations.
extension Qwen35Model {
    public var toolCallFormat: ToolCallFormat? { .qwen35 }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}

extension Qwen35TextModel {
    public var toolCallFormat: ToolCallFormat? { .qwen35 }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
