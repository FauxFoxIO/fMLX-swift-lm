import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXVLM

@Test func combinedQwenPreservedHeadConvertsMixedNorms() throws {
    let config = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
    let model = MLXLLM.Qwen35MTPDraftModel(config, mixedPreservedNorms: true)
    let result = model.sanitize(weights: [
        "mtp.norm.weight": MLXArray.full([16], values: MLXArray(Float(2.1))),
        "mtp.pre_fc_norm_embedding.weight": MLXArray.full([16], values: MLXArray(Float(-0.5))),
        "mtp.layers.0.input_layernorm.weight": MLXArray.zeros([16]),
    ])
    #expect(
        try #require(result["mtp.norm.weight"]).asArray(Float.self)
            == Array(repeating: Float(3.1), count: 16))
    #expect(
        try #require(result["mtp.pre_fc_norm_embedding.weight"]).asArray(Float.self)
            == Array(repeating: Float(0.5), count: 16))
    #expect(
        try #require(result["mtp.layers.0.input_layernorm.weight"]).asArray(Float.self)
            == Array(repeating: Float(1), count: 16))
}

@Test func combinedQuantizedMoECheckpointLoadsTargetAndMTP() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "combined-qwen-\(UUID())")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let data = Data(
        """
        {"model_type":"qwen3_5_moe","text_config":{
        "model_type":"qwen3_5_moe_text","hidden_size":64,"num_hidden_layers":2,
        "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":32,"linear_num_value_heads":2,"linear_num_key_heads":1,
        "linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,
        "vocab_size":100,"full_attention_interval":2,"mtp_num_hidden_layers":1,
        "tie_word_embeddings":true,"rope_theta":10000000.0,"partial_rotary_factor":0.25,
        "num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":64,"shared_expert_intermediate_size":64},
        "quantization":{"bits":4,"group_size":32,"mode":"affine","vision":"fp16_passthrough",
        "mtp":"preserved","quantization_backend":"mx.quantize"}}
        """.utf8)
    try data.write(to: directory.appendingPathComponent("config.json"))
    let configuration = try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data)
    do {
        let target = MLXLLM.Qwen35MoEModel(configuration)
        let head = MLXLLM.Qwen35MTPDraftModel(configuration, preconvertedNorms: true)
        quantize(model: target, groupSize: 32, bits: 4)
        quantize(model: head, groupSize: 32, bits: 4)
        var arrays = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        arrays.merge(Dictionary(uniqueKeysWithValues: head.parameters().flattened())) { _, new in
            new
        }
        arrays["vision_tower.unused.weight"] = MLXArray.zeros([2, 2])
        try save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
    }
    let target = try await NativeTextModelLoader.load(directory: directory)
    let head = try #require(try await NativeTextModelLoader.loadCombinedMTP(directory: directory))
    let runtime = try ConcurrentTextRuntime(
        model: target,
        identity: .init(
            modelRevision: "fixture", tokenizerRevision: "fixture", chatTemplateRevision: "fixture",
            adapterRevision: "none", cacheLayoutRevision: "fixture"),
        configuration: .init(
            memoryBudgetBytes: 32_000_000, prefixCacheBytes: 1_000_000,
            workingMemoryBytes: 1_000_000, prefillChunkSize: 2), drafter: head)
    func generate(speculative: Bool) async throws -> [Int] {
        let stream = try await runtime.generate(
            .init(tokens: [1, 2, 3, 4], maxTokens: 12, speculative: speculative))
        var tokens: [Int] = []
        var rounds = 0
        for try await event in stream.events {
            if case .token(let token) = event { tokens.append(token) }
            if case .speculation(let telemetry) = event { rounds += telemetry.roundCount }
        }
        if speculative { #expect(rounds > 0) }
        return tokens
    }
    let baseline = try await generate(speculative: false)
    let speculative = try await generate(speculative: true)
    #expect(baseline == speculative)
    await runtime.shutdown()
}

@Test
func testQwen35TextConfigurationDecodesMTPFields() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))

    #expect(cfg.mtpNumHiddenLayers == 1)
    #expect(cfg.mtpUseDedicatedEmbeddings == false)
}

@Test
func testQwen35VLMTextConfigurationDecodesMTPFields() throws {
    let cfg = try JSONDecoder().decode(
        MLXVLM.Qwen35Configuration.TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))

    #expect(cfg.mtpNumHiddenLayers == 1)
    #expect(cfg.mtpUseDedicatedEmbeddings == false)
}

@Test
func testQwen35MTPDraftSanitizeKeepsAndShiftsMTPNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    let sanitized = drafter.sanitize(weights: [
        "mtp.norm.weight": MLXArray.zeros([16]),
        "mtp.pre_fc_norm_embedding.weight": MLXArray.zeros([16]),
        "mtp.layers.0.self_attn.q_proj.weight": MLXArray.zeros([32, 16]),
        "mtp.layers.0.mlp.experts.gate_up_proj": MLXArray.zeros([2, 32, 16]),
        "mtp.layers.0.mlp.experts.down_proj": MLXArray.zeros([2, 16, 16]),
        "model.embed_tokens.weight": MLXArray.zeros([16, 16]),
    ])

    #expect(sanitized["model.embed_tokens.weight"] == nil)
    #expect(sanitized["mtp.layers.0.self_attn.q_proj.weight"] != nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.gate_up_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.down_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
    let norm = try #require(sanitized["mtp.norm.weight"])
    let pre = try #require(sanitized["mtp.pre_fc_norm_embedding.weight"])
    eval(norm, pre)
    #expect(allClose(norm, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
    #expect(allClose(pre, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
}

@Test
func testQwen35StandaloneMTPDoesNotDoubleShiftConvertedNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35Configuration.self,
        from: Data(qwen35StandaloneMTPConfigJSON().utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg, preconvertedNorms: true)

    let weight = MLXArray.zeros([16])
    let sanitized = drafter.sanitize(weights: ["mtp.norm.weight": weight])
    let norm = try #require(sanitized["mtp.norm.weight"])
    eval(norm)
    #expect(allClose(norm, weight, rtol: 0, atol: 0).item(Bool.self))
}

@Test
func testQwen35StandaloneBareMTPNamesPreserveConvertedNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg, preconvertedNorms: true)
    let weights = [
        "fc.weight": MLXArray.zeros([16, 32]),
        "pre_fc_norm_hidden.weight": MLXArray.ones([16]),
        "norm.weight": MLXArray.ones([16]),
        "layers.0.self_attn.q_proj.scales": MLXArray.ones([32, 1]),
        "model.embed_tokens.weight": MLXArray.zeros([16, 16]),
    ]
    let sanitized = drafter.sanitize(weights: weights)
    #expect(sanitized.count == 4)
    #expect(sanitized["mtp.fc.weight"]?.shape == [16, 32])
    #expect(sanitized["mtp.layers.0.self_attn.q_proj.scales"] != nil)
    #expect(
        try #require(sanitized["mtp.norm.weight"]).asArray(Float.self)
            == Array(repeating: 1, count: 16))
}

@Test
func testQwen35MTPDraftSanitizeStacksPerExpertMoEWeights() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1, numExperts: 2).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    let weights: [String: MLXArray] = [
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.gate_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.up_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.down_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.down_proj.weight": MLXArray.ones([16, 16]),
    ]

    let sanitized = drafter.sanitize(weights: weights)

    #expect(sanitized["mtp.layers.0.mlp.experts.0.gate_proj.weight"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
}

@Test
func testQwen35MTPDraftInstantiatesDedicatedEmbeddingWhenConfigured() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(
            qwen35TextConfigJSON(mtpLayers: 1, mtpUseDedicatedEmbeddings: true).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    #expect(drafter.mtp.embedTokens != nil)
    let sanitized = drafter.sanitize(weights: [
        "mtp.embed_tokens.weight": MLXArray.zeros([16, 16]),
        "model.embed_tokens.weight": MLXArray.ones([16, 16]),
    ])
    #expect(sanitized["mtp.embed_tokens.weight"] != nil)
    #expect(sanitized["model.embed_tokens.weight"] == nil)
}

@Suite(.serialized)
struct Qwen35MTPMetalTests {
    @Test(arguments: [false, true], [0, 2])
    func testCompiledVerificationPreservesRecurrentCheckpoint(
        fusedCheckpoint: Bool, numExperts: Int
    ) throws {
        var cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1, numExperts: numExperts).utf8))
        cfg.fullAttentionInterval = 4
        cfg.linearKeyHeadDim = 128
        cfg.linearValueHeadDim = 128
        for dtype in [DType.float32, .bfloat16] {
            let layer = withRandomState(MLXRandom.RandomState(seed: 914)) {
                MLXLLM.Qwen35DecoderLayer(cfg, layerIdx: 0)
            }
            layer.fusedVerificationCheckpointEnabled = fusedCheckpoint
            layer.update(parameters: layer.parameters().mapValues { $0.asType(dtype) })
            layer.train(false)
            let seed = MambaCache()
            let prefix = MLXArray.ones([1, 3, 16], dtype: dtype)
            eval(layer(prefix, attentionMask: .none, ssmMask: nil, cache: seed))
            let compiledCache = seed.copy() as! MambaCache
            let referenceCache = seed.copy() as! MambaCache
            let input = MLXArray((0 ..< 32).map { Float($0 % 11 - 5) / 7 })
                .reshaped(1, 2, 16).asType(dtype)
            let compiled = layer(
                input, attentionMask: .none, ssmMask: nil, cache: compiledCache,
                checkpointAfter: 1)
            let mixed = layer.linearAttn!(
                layer.inputLayerNorm(input), cache: referenceCache, checkpointAfter: 1)
            let residual = input + mixed
            let reference =
                residual + (layer.mlp as! UnaryLayer)(layer.postAttentionLayerNorm(residual))
            eval(compiled, reference)
            #expect(layer.hasCompiledVerificationLayer)
            #expect(allClose(compiled, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(compiledCache.hasSpeculativeCheckpoint)
            for (actual, expected) in zip(compiledCache.state, referenceCache.state) {
                #expect(allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            }
            #expect(compiledCache.restoreSpeculativeCheckpoint())
            #expect(referenceCache.restoreSpeculativeCheckpoint())
            for (actual, expected) in zip(compiledCache.state, referenceCache.state) {
                #expect(actual.dtype == expected.dtype)
                #expect(allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            }
            let continuation = input[0..., ..<1, 0...]
            let actual = layer(
                continuation, attentionMask: .none, ssmMask: nil, cache: compiledCache)
            let expected = layer(
                continuation, attentionMask: .none, ssmMask: nil, cache: referenceCache)
            #expect(allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    @Test(arguments: [false, true])
    func testCompiledVerificationAttentionAndWeightInvalidation(fusedCheckpoint: Bool) throws {
        var cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        cfg.fullAttentionInterval = 4
        for index in [0, 3] {
            let layer = withRandomState(MLXRandom.RandomState(seed: 117)) {
                MLXLLM.Qwen35DecoderLayer(cfg, layerIdx: index)
            }
            layer.fusedVerificationCheckpointEnabled = fusedCheckpoint
            layer.train(false)
            let seed: KVCache = index == 0 ? MambaCache() : KVCacheSimple()
            let prefix = MLXArray.ones([1, 3, 16])
            eval(layer(prefix, attentionMask: .causal, ssmMask: nil, cache: seed))
            let input = MLXArray((0 ..< 32).map { Float($0 % 7) / 7 }).reshaped(1, 2, 16)
            for updated in [false, true] {
                if updated {
                    layer.update(parameters: layer.parameters().mapValues { $0 * 0.75 })
                    layer.invalidateCompiledTraces()
                    #expect(!layer.hasCompiledVerificationLayer)
                }
                let compiledCache = seed.copy()
                let referenceCache = seed.copy()
                layer.compiledVerificationEnabled = true
                let actual = layer(
                    input, attentionMask: .causal, ssmMask: nil, cache: compiledCache,
                    checkpointAfter: 1)
                layer.compiledVerificationEnabled = false
                let expected = layer(
                    input, attentionMask: .causal, ssmMask: nil, cache: referenceCache,
                    checkpointAfter: 1)
                #expect(allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
                #expect(compiledCache.offset == referenceCache.offset)
                for (a, b) in zip(compiledCache.state, referenceCache.state) {
                    #expect(allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self))
                }
            }
        }
    }

    @Test(arguments: [false, true])
    func testCompiledVerificationDoesNotRetainLayer(fusedCheckpoint: Bool) throws {
        var cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        cfg.fullAttentionInterval = 4
        weak var reference: MLXLLM.Qwen35DecoderLayer?
        do {
            let layer = MLXLLM.Qwen35DecoderLayer(cfg, layerIdx: 0)
            layer.fusedVerificationCheckpointEnabled = fusedCheckpoint
            let cache = MambaCache()
            eval(
                layer(MLXArray.ones([1, 3, 16]), attentionMask: .none, ssmMask: nil, cache: cache))
            eval(
                layer(
                    MLXArray.ones([1, 2, 16]), attentionMask: .none, ssmMask: nil, cache: cache,
                    checkpointAfter: 1))
            reference = layer
            #expect(layer.hasCompiledVerificationLayer)
        }
        #expect(reference == nil)
    }

    @Test
    func testScheduledMTPChunkedPrefillMatchesReferenceAndGreedy() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let target = withRandomState(MLXRandom.RandomState(seed: 112)) {
            MLXLLM.Qwen35TextModel(cfg)
        }
        let drafter = withRandomState(MLXRandom.RandomState(seed: 113)) {
            MLXLLM.Qwen35MTPDraftModel(cfg)
        }
        let prompt = [1, 2, 3, 4, 5]
        let parameters = GenerateParameters(maxTokens: 12, temperature: 0)
        var reference = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), mainModel: target, drafter: drafter,
            parameters: parameters, blockSize: 2)
        reference.jointGreedyVerificationEnabled = false
        let expected = Array(reference)
        var chunked = try MTPSpeculativeTokenIterator(
            scheduledPrompt: prompt,
            mainModel: target, drafter: drafter, mainCache: target.newCache(parameters: nil),
            parameters: parameters, blockSize: 2)
        try chunked.prepareScheduledChunk([1, 2], nextPromptToken: 3)
        try chunked.prepareScheduledChunk([3, 4], nextPromptToken: 5)
        try chunked.prepareScheduledChunk([5], nextPromptToken: nil)
        var actual: [Int] = []
        while let token = chunked.next() { actual.append(token) }
        #expect(actual == expected)
        #expect(chunked.jointGreedyVerificationCount > 0)
        let greedy = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: target, parameters: parameters)
        #expect(Array(greedy) == expected)
    }

    @Test
    func testIntermediateLogitsElisionPreservesAllScheduledState() throws {
        var cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        cfg.hiddenLayers = 4
        cfg.fullAttentionInterval = 4
        cfg.linearKeyHeadDim = 128
        cfg.linearValueHeadDim = 128
        let target = withRandomState(MLXRandom.RandomState(seed: 221)) {
            MLXLLM.Qwen35TextModel(cfg)
        }
        let drafter = withRandomState(MLXRandom.RandomState(seed: 222)) {
            MLXLLM.Qwen35MTPDraftModel(cfg)
        }
        let prompt = [1, 2, 3, 4, 5]
        let parameters = GenerateParameters(maxTokens: 12, temperature: 0)
        var reference = try MTPSpeculativeTokenIterator(
            scheduledPrompt: prompt, mainModel: target, drafter: drafter,
            mainCache: target.newCache(parameters: nil), parameters: parameters, blockSize: 2)
        var optimized = try MTPSpeculativeTokenIterator(
            scheduledPrompt: prompt, mainModel: target, drafter: drafter,
            mainCache: target.newCache(parameters: nil), parameters: parameters, blockSize: 2)
        for position in stride(from: 0, to: prompt.count, by: 2) {
            let end = min(position + 2, prompt.count)
            let chunk = Array(prompt[position ..< end])
            let next = end < prompt.count ? prompt[end] : nil
            try reference.prepareScheduledChunk(
                chunk, nextPromptToken: next, evaluateIntermediateLogits: true)
            try optimized.prepareScheduledChunk(chunk, nextPromptToken: next)
            let actual = optimized.scheduledResidentArrays
            let expected = reference.scheduledResidentArrays
            #expect(actual.count == expected.count)
            for (a, b) in zip(actual, expected) {
                #expect(a.dtype == b.dtype)
                #expect(arrayEqual(a, b).item(Bool.self))
            }
        }
        #expect(Array(optimized) == Array(reference))
    }

    @Test
    func testQwen35MTPPredictorAdvancesEveryLayerCachePerToken() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 2).utf8))
        let predictor = MLXLLM.Qwen35MTPPredictor(cfg)
        let cache = predictor.newCache()
        let embeds = MLXArray.zeros([1, 1, 16])
        let hidden = MLXArray.zeros([1, 1, 16])

        let first = predictor(
            inputsEmbeds: embeds, hiddenStates: hidden, cache: cache,
            positionOffset: 128)
        eval(first)
        #expect(cache[0].offset == 1)
        #expect(cache[1].offset == 1)

        let second = predictor(
            inputsEmbeds: embeds, hiddenStates: first, cache: cache,
            positionOffset: 129)
        eval(second)
        #expect(cache[0].offset == 2)
        #expect(cache[1].offset == 2)
    }

    @Test
    func testQwen35VLMMTPPositionIdsApplyMultimodalDelta() throws {
        let positionIds = MLXVLM.qwen35MTPPositionIds(
            offset: 10,
            batchSize: 2,
            positionDeltas: MLXArray([Int32(3), 5])
        )

        eval(positionIds)
        #expect(positionIds.shape == [3, 2, 1])
        #expect(positionIds[0, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[1, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[2, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[1, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[2, 1, 0].item(Int32.self) == 15)
    }

    @Test
    func testQwen35VLMMTPPositionIdsRepeatAndTrimShortBatchDeltas() throws {
        let positionIds = MLXVLM.qwen35MTPPositionIds(
            offset: 10,
            batchSize: 4,
            positionDeltas: MLXArray([Int32(3), 5])
        )

        eval(positionIds)
        #expect(positionIds.shape == [3, 4, 1])
        #expect(positionIds[0, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[0, 2, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 3, 0].item(Int32.self) == 15)
    }

    @Test
    func testQwen35TextModelEmitDrafterStateBySynthetic() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(cfg)
        let cache = try model.newCache(parameters: nil as GenerateParameters?)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let input = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4]))
        let out = model(input, cache: cache, state: state)

        let hidden = try #require(out.state?[mtpLastHiddenStatesKey])
        let sharedKV = try #require(out.state?[mtpSharedKVStatesKey])
        let sharedKVOffsets = try #require(out.state?[mtpSharedKVOffsetsKey])
        eval(out.logits, hidden)
        #expect(out.logits.shape == [1, 4, 16])
        #expect(hidden.shape == [1, 4, 16])
        #expect(Set(sharedKV.keys) == ["full_attention"])
        #expect(sharedKVOffsets == ["full_attention": 4])
        let full = try #require(sharedKV["full_attention"])
        eval(full.0, full.1)
        #expect(full.0.shape.count == 4)
        #expect(full.1.shape.count == 4)
    }

    @Test
    func testQwen35TextModelEmitsPostFinalNormHiddenState() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
        let expected = model.model.forward(tokens, applyFinalNorm: false)
        let normalized = model.model.norm(expected)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let output = model(LMInput.Text(tokens: tokens), cache: nil, state: state)
        let emitted = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(expected, normalized, emitted)

        #expect(allClose(emitted, normalized, rtol: 0, atol: 0).item(Bool.self))
        #expect(!allClose(emitted, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test
    func testQwen35VLMEmitsPostFinalNormHiddenState() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8))
        let model = MLXVLM.Qwen35(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
        let base = MLXArray(0 ..< 4).asType(.int32).reshaped([1, 1, 4])
        let positionIds = broadcast(base, to: [3, 1, 4])
        let expected = model.languageModel.model(
            tokens, positionIds: positionIds, applyFinalNorm: false)
        let normalized = model.languageModel.model.norm(expected)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let output = model.languageModel(
            tokens, cache: nil, state: state, positionIds: positionIds)
        let emitted = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(expected, normalized, emitted)

        #expect(allClose(emitted, normalized, rtol: 0, atol: 0).item(Bool.self))
        #expect(!allClose(emitted, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test
    func testQwen35DrafterCacheTracksVerifiedSequenceAcrossAcceptRejectPatterns() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(cfg)
        let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])

        var targetState = LMOutput.State()
        targetState[mtpEmitFlagKey] = true
        let targetOutput = target(LMInput.Text(tokens: prompt), cache: nil, state: targetState)
        let promptHidden = try #require(targetOutput.state?[mtpLastHiddenStatesKey])

        for pattern in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            var state = drafter.makeState(parameters: nil)
            var bonus = MLXArray([Int32(4)])
            drafter.prepareDrafterState(
                target: target, promptTokens: prompt, targetHidden: promptHidden,
                firstBonus: bonus, positionDeltas: nil, state: &state, sampler: sampler)
            eval(state.seedToken!, state.seedHidden!)
            #expect(state.cache.allSatisfy { $0.offset == 3 })
            #expect(state.nextPosition == 3)

            var expectedPosition = 3
            for accepted in pattern {
                let proposal = drafter.draftBlock(
                    target: target, lastToken: bonus,
                    lastHidden: promptHidden[0..., (-1)..., 0...], sharedKV: [:],
                    positionDeltas: nil, queryOffset: expectedPosition, blockSize: 2,
                    state: &state, sampler: sampler)
                eval(proposal.tokens, proposal.logits)
                #expect(state.cache.allSatisfy { $0.offset == expectedPosition })

                let verifyHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
                let finalToken = MLXArray([Int32(8 + accepted)])
                drafter.commitDrafterState(
                    target: target, targetHidden: verifyHidden, draftTokens: proposal.tokens,
                    acceptedCount: accepted, finalToken: finalToken, positionDeltas: nil,
                    state: &state, sampler: sampler)
                eval(state.seedToken!, state.seedHidden!)
                expectedPosition += accepted + 1
                #expect(state.cache.allSatisfy { $0.offset == expectedPosition })
                #expect(state.nextPosition == expectedPosition)
                bonus = finalToken
            }
        }
    }

    @Test
    func testQwen35GDNCheckpointMatchesPrefixWithoutReplayingProjections() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let layer = MLXLLM.Qwen35GatedDeltaNet(cfg)
        let input = MLXRandom.normal([1, 2, 16])
        let fullCache = MambaCache()
        let speculativeCache = MambaCache()
        let prefixCache = MambaCache()

        let full = layer(input, cache: fullCache)
        let speculative = layer(input, cache: speculativeCache, checkpointAfter: 1)
        _ = layer(input[0..., ..<1, 0...], cache: prefixCache)
        eval(full, speculative)

        #expect(allClose(speculative, full, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        #expect(speculativeCache.hasSpeculativeCheckpoint)
        #expect(speculativeCache.restoreSpeculativeCheckpoint())

        let restored = speculativeCache.state
        let expected = prefixCache.state
        #expect(restored.count == expected.count)
        for (actual, reference) in zip(restored, expected) {
            eval(actual, reference)
            #expect(allClose(actual, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    @Test
    func testQwen35VLMGDNCheckpointMatchesPrefix() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let layer = MLXVLM.Qwen35Language.GatedDeltaNet(cfg)
        let input = MLXRandom.normal([1, 2, 16])
        let fullCache = MambaCache()
        let speculativeCache = MambaCache()
        let prefixCache = MambaCache()

        let full = layer(input, cache: fullCache)
        let speculative = layer(input, cache: speculativeCache, checkpointAfter: 1)
        _ = layer(input[0..., ..<1, 0...], cache: prefixCache)
        eval(full, speculative)

        #expect(allClose(speculative, full, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        #expect(speculativeCache.restoreSpeculativeCheckpoint())
        for (actual, reference) in zip(speculativeCache.state, prefixCache.state) {
            eval(actual, reference)
            #expect(allClose(actual, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    @Test
    func testQwen35HybridCacheRewindRestoresAttentionAndRecurrentStateAtomically() {
        let attention = KVCacheSimple()
        let keys = MLXArray.zeros([1, 1, 3, 2])
        _ = attention.update(keys: keys, values: keys)

        let recurrent = MambaCache()
        let checkpointConv = MLXArray.ones([1, 1, 4])
        let checkpointState = MLXArray.ones([1, 2, 2, 2])
        recurrent.saveSpeculativeCheckpoint(
            convState: checkpointConv, recurrentState: checkpointState, advancedBy: 1)
        recurrent[0] = MLXArray.zeros([1, 1, 4])
        recurrent[1] = MLXArray.zeros([1, 2, 2, 2])

        let rewound = rewindSpeculativePromptCache([attention, recurrent], numTokens: 1)
        #expect(rewound == 1)
        #expect(attention.offset == 2)
        #expect(!recurrent.hasSpeculativeCheckpoint)

        let restored = recurrent.state
        #expect(restored.count == 2)
        eval(restored[0], restored[1])
        #expect(allClose(restored[0], checkpointConv, rtol: 0, atol: 0).item(Bool.self))
        #expect(allClose(restored[1], checkpointState, rtol: 0, atol: 0).item(Bool.self))
    }
}

@Suite(.serialized)
struct Qwen35MTPRegistrationTests {
    @Test
    func registrationsCreateTextAndVLMDrafters() async throws {
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8),
            modelType: "qwen3_5_text")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)

        await MLXVLM.Qwen35VLMMTPRegistration.register()

        let wrappedTextModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35WrappedTextConfigJSON(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(wrappedTextModel is MLXLLM.Qwen35MTPDraftModel)

        let vlmModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let standalone = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35StandaloneMTPConfigJSON().utf8),
            modelType: "qwen3_5_mtp")
        #expect(standalone is MLXLLM.Qwen35MTPDraftModel)
        #expect(standalone.maximumBlockSize == 2)
        #expect(standalone.requiresPromptPrefill)
        #expect(!standalone.requiresSharedTargetKV)
    }

    @Test
    func registrationsAreOrderIndependentForSharedModelTypes() async throws {
        await MLXVLM.Qwen35VLMMTPRegistration.register()
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let vlmModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35WrappedTextConfigJSON(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)
    }
}

private func qwen35TextConfigJSON(
    mtpLayers: Int,
    mtpUseDedicatedEmbeddings: Bool = false,
    numExperts: Int = 0
) -> String {
    """
    {
      "model_type": "qwen3_5_text",
      "hidden_size": 16,
      "num_hidden_layers": 1,
      "intermediate_size": 32,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 8,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 1,
      "linear_key_head_dim": 8,
      "linear_value_head_dim": 8,
      "linear_conv_kernel_dim": 2,
      "rms_norm_eps": 1e-6,
      "vocab_size": 16,
      "rope_theta": 100000.0,
      "partial_rotary_factor": 0.25,
      "max_position_embeddings": 64,
      "tie_word_embeddings": true,
      "attention_bias": false,
      "full_attention_interval": 1,
      "mtp_num_hidden_layers": \(mtpLayers),
      "mtp_use_dedicated_embeddings": \(mtpUseDedicatedEmbeddings),
      "num_experts": \(numExperts),
      "num_experts_per_tok": \(numExperts == 0 ? 0 : 1),
      "moe_intermediate_size": 16,
      "shared_expert_intermediate_size": 16,
      "rope_parameters": {
        "type": "default",
        "rope_theta": 100000.0,
        "partial_rotary_factor": 0.25
      }
    }
    """
}

private func qwen35VLMConfigJSON(mtpLayers: Int) -> String {
    """
    {
      "model_type": "qwen3_5",
      "text_config": \(qwen35TextConfigJSON(mtpLayers: mtpLayers)),
      "vision_config": {
        "model_type": "qwen3_5_vit",
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "out_hidden_size": 16,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "num_position_embeddings": 16
      }
    }
    """
}

private func qwen35WrappedTextConfigJSON(modelType: String) -> String {
    """
    {
      "model_type": "\(modelType)",
      "text_config": \(qwen35TextConfigJSON(mtpLayers: 1))
    }
    """
}

private func qwen35StandaloneMTPConfigJSON() -> String {
    """
    {
      "model_type": "qwen3_5_mtp",
      "block_size": 3,
      "text_config": \(qwen35TextConfigJSON(mtpLayers: 1)),
      "tie_word_embeddings": true,
      "vision_config": {}
    }
    """
}
