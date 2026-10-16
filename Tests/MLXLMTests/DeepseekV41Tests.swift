import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// DeepSeek-V4.1 against mlx-lm (plus DeepSeek's Engram) on the tiny native-format fixture
/// written by `scripts/parity/deepseek_v41.py`. The 24-token sequence evicts window rows,
/// selects 3 of up to 24 compressed entries, and reaches both Engram layers.
struct DeepseekV41ParityTests {
    @Test("Prefill logits match the reference")
    func prefillMatchesReference() async throws {
        let fixture = try ParityFixture("deepseek_v41")
        let model = try await fixture.loadLanguageModel()
        let logits = model(fixture.inputIDs, cache: nil)
        let difference = fixture.maxAbsDifference(logits)
        print("deepseek_v41 prefill max |diff| = \(difference)")
        #expect(difference < 1e-4)
    }

    @Test("Chunked prefill and cached decode match full-sequence logits")
    func cachedDecodeMatchesReference() async throws {
        let fixture = try ParityFixture("deepseek_v41")
        let model = try await fixture.loadLanguageModel()
        let cache = try model.newCache(parameters: nil)
        let length = fixture.inputIDs.dim(1)
        // Odd chunk sizes leave compressor groups open across calls.
        let bounds = [0, 5, 9, 14] + Array(15 ... length)
        var rows = [MLXArray]()
        for (start, end) in zip(bounds, bounds.dropFirst()) {
            rows.append(model(fixture.inputIDs[0..., start ..< end], cache: cache))
        }
        let difference = fixture.maxAbsDifference(concatenated(rows, axis: 1))
        print("deepseek_v41 chunked max |diff| = \(difference)")
        #expect(cache[0].offset == length)
        #expect(difference < 1e-4)
    }

    @Test("Rewinding the cache and feeding the same tokens again reproduces the logits")
    func rewindMatchesReference() async throws {
        let fixture = try ParityFixture("deepseek_v41")
        let model = try await fixture.loadLanguageModel()
        let cache = try model.newCache(parameters: nil)
        let length = fixture.inputIDs.dim(1)
        _ = model(fixture.inputIDs[0..., ..<11], cache: cache)
        for t in 11 ..< 19 {
            _ = model(fixture.inputIDs[0..., t ..< (t + 1)], cache: cache)
        }
        // Drop 5 positions, which reopens a compressor group, then replay them.
        for layer in cache {
            #expect(layer.trim(5) == 5)
        }
        #expect(cache[0].offset == 14)
        var rows = [model(fixture.inputIDs[0..., 14 ..< 17], cache: cache)]
        for t in 17 ..< length {
            rows.append(model(fixture.inputIDs[0..., t ..< (t + 1)], cache: cache))
        }
        let expected = fixture.logits[0..., 14...]
        let difference = abs(concatenated(rows, axis: 1) - expected).max().item(Float.self)
        print("deepseek_v41 rewind max |diff| = \(difference)")
        #expect(difference < 1e-4)
    }

    @Test("A copied cache continues like the original")
    func copiedCacheMatches() async throws {
        let fixture = try ParityFixture("deepseek_v41")
        let model = try await fixture.loadLanguageModel()
        let cache = try model.newCache(parameters: nil)
        _ = model(fixture.inputIDs[0..., ..<13], cache: cache)
        let copy = cache.map { $0.copy() }
        let restored = try model.newCache(parameters: nil)
        for (source, target) in zip(cache, restored) {
            var target = target
            target.state = source.state
            target.metaState = source.metaState
        }
        let next = fixture.inputIDs[0..., 13 ..< 16]
        let expected = fixture.logits[0..., 13 ..< 16]
        for candidate in [copy, restored] {
            let difference = abs(model(next, cache: candidate) - expected).max().item(Float.self)
            #expect(difference < 1e-4)
        }
    }

    @Test("Engram n-gram hashes match DeepSeek's NgramHashState")
    func engramHashesMatch() async throws {
        let fixture = try ParityFixture("deepseek_v41")
        let expected = try loadArrays(
            url: fixture.directory.appending(component: "expected.safetensors"))["engram_hashes"]!
        let model = try await fixture.loadLanguageModel() as! DeepseekV41Model
        let hasher = try #require(model.engramMemory.hasher)
        let tokens = fixture.inputIDs.asArray(Int32.self)
        let full = hasher.hashes(tokens: tokens, history: [], start: 0)
        #expect(full == expected.asArray(Int32.self).map(Int.init))

        // Decode one token at a time from a short history.
        var history = [Int32]()
        var stepwise = [Int]()
        for (t, token) in tokens.enumerated() {
            stepwise += hasher.hashes(tokens: [token], history: history, start: t)
            history = Array((history + [hasher.compressed(token)]).suffix(hasher.maxNgram - 1))
        }
        #expect(stepwise == full)
    }
}

struct DeepseekV41ComponentTests {
    /// The FP4 E2M1 values of codes 0 ..< 16.
    static let fp4Values: [Float] = [
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6,
    ]

    @Test("Native FP4 experts repack losslessly as mxfp4")
    func fp4Repack() throws {
        let (rows, columns) = (3, 64)
        var codes = [UInt8]()
        for i in 0 ..< rows * columns {
            codes.append(UInt8((i * 7 + i / 5) % 16))
        }
        // Two codes per byte, the even element in the low nibble.
        let packed = stride(from: 0, to: codes.count, by: 2).map {
            Int8(bitPattern: codes[$0] | (codes[$0 + 1] << 4))
        }
        let exponents: [UInt8] = [120, 127, 130, 126, 131, 125]
        let (weight, scales, option) = try DeepseekV41Checkpoint.repack(
            weight: MLXArray(packed, [rows, columns / 2]), scale: MLXArray(exponents, [rows, 2]))
        #expect(option.mode == .mxfp4 && option.bits == 4 && option.groupSize == 32)
        let actual = dequantized(
            weight, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4,
            dtype: .float32
        ).asArray(Float.self)
        for i in 0 ..< rows * columns {
            let scale = Float(pow(2, Double(Int(exponents[i / 32]) - 127)))
            #expect(actual[i] == Self.fp4Values[Int(codes[i])] * scale)
        }
    }

    @Test("Native FP8 tiles repack losslessly as mxfp8 with one scale per row")
    func fp8Repack() throws {
        let (rows, columns) = (40, 64)  // two tile rows, the second one partial
        let values = MLXRandom.normal([rows, columns], key: MLXRandom.key(1))
        let fp8 = toFP8(values)
        let exponents: [UInt8] = [125, 127, 129, 131]
        let (weight, scales, option) = try DeepseekV41Checkpoint.repack(
            weight: fp8, scale: MLXArray(exponents, [2, 2]))
        #expect(option.mode == .mxfp8 && option.bits == 8)
        #expect(scales.shape == [rows, 2])
        let actual = dequantized(
            weight, scales: scales, biases: nil, groupSize: 32, bits: 8, mode: .mxfp8,
            dtype: .float32)
        let tile = MLXArray(exponents.map { Float(pow(2, Double(Int($0) - 127))) }, [2, 2])
        let expanded = repeated(repeated(tile, count: 32, axis: 0), count: 32, axis: 1)[..<rows]
        let expected = fromFP8(fp8, dtype: .float32) * expanded
        #expect(abs(actual - expected).max().item(Float.self) == 0)
    }

    @Test("Safetensors I8, F8_E4M3 and F8_E8M0 tensors load as the bytes the repack expects")
    func nativeDTypesLoad() throws {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "deepseek_v41_dtypes_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeSafetensors(
            url,
            [
                ("i8", "I8", [4], [0x01, 0xFE, 0x7F, 0x80]),
                ("e4m3", "F8_E4M3", [2], [0x38, 0xB0]),  // 1.0, -0.5
                ("e8m0", "F8_E8M0", [2], [127, 130]),
            ])
        let arrays = try loadArrays(url: url)
        #expect(arrays["i8"]!.dtype == .int8)
        #expect(arrays["i8"]!.asArray(Int8.self) == [1, -2, 127, -128])
        #expect(arrays["e4m3"]!.dtype == .uint8)
        #expect(fromFP8(arrays["e4m3"]!, dtype: .float32).asArray(Float.self) == [1, -0.5])
        #expect(arrays["e8m0"]!.asArray(UInt8.self) == [127, 130])
    }

    @Test("Native names map to module paths")
    func renaming() {
        let cases: [(String, String?)] = [
            ("embed.weight", "model.embed_tokens.weight"),
            ("head.weight", "lm_head.weight"),
            ("layers.3.attn.wq_a.scale", "model.layers.3.self_attn.q_a_proj.scale"),
            (
                "layers.3.attn.compressor.norm.weight",
                "model.layers.3.self_attn.compressor.kv_norm.weight"
            ),
            (
                "layers.3.attn.compressor.wkv.weight",
                "model.layers.3.self_attn.compressor.kv_proj.weight"
            ),
            ("layers.3.attn.indexer.wk.weight", "model.layers.3.self_attn.indexer.k_proj.weight"),
            ("layers.3.hc_attn_fn", "model.layers.3.attn_hc.fn"),
            ("layers.3.ffn.gate.bias", "model.layers.3.mlp.gate.e_score_correction_bias"),
            ("layers.3.ffn.experts.17.w2.weight", "model.layers.3.mlp.experts.17.down_proj.weight"),
            ("layers.3.ffn.shared_experts.w3", "model.layers.3.mlp.shared_experts.up_proj"),
            ("layers.1.engram.wkv.weight", "model.layers.1.engram.wkv.weight"),
            ("layers.1.engram.embed.weight", nil),
            ("layers.3.ffn.gate.bias_vl", nil),
            ("mtp.0.attn_norm.weight", nil),
            ("model.layers.0.mlp.gate.weight", "model.layers.0.mlp.gate.weight"),
        ]
        for (name, expected) in cases {
            #expect(DeepseekV41Checkpoint.rename(name) == expected, "\(name)")
        }
    }

    @Test("Memory-mapped Engram rows match an in-memory table")
    func mappedEngramGather() throws {
        let (rows, dim) = (300, 64)
        let values = MLXRandom.normal([rows, dim], key: MLXRandom.key(2)) * 8
        let weight = toFP8(values)
        let scale = MLXRandom.randInt(low: 120, high: 134, [rows, dim / 32], key: MLXRandom.key(3))
            .asType(.uint8)
        let url = FileManager.default.temporaryDirectory
            .appending(component: "deepseek_v41_engram_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeSafetensors(
            url,
            [
                ("pad", "F32", [3], [UInt8](repeating: 0, count: 12)),
                ("table.weight", "F8_E4M3", [rows, dim], weight.asArray(UInt8.self)),
                ("table.scale", "F8_E8M0", [rows, dim / 32], scale.asArray(UInt8.self)),
            ])
        let locations = try DeepseekV41TensorLocation.read(url)
        var files = [URL: DeepseekV41MappedFile]()
        let mapped = try DeepseekV41MappedEngramTable(
            weight: locations["table.weight"]!, scales: locations["table.scale"]!, biases: nil,
            format: .fp8(block: 32), files: &files)
        let resident = DeepseekV41ArrayEngramTable(
            weight: weight, scales: scale, format: .fp8(block: 32))
        let ids = [0, 299, 17, 17, 150, 1, 298] + (0 ..< 600).map { ($0 * 37) % rows }
        let a = mapped.rows(ids)
        let b = resident.rows(ids)
        #expect(a.shape == [ids.count, dim])
        #expect(abs(a - b).max().item(Float.self) == 0)
        // And both equal a direct dequantization of the selected rows.
        let expected =
            fromFP8(weight, dtype: .float32).reshaped(rows, 2, 32)
            * pow(MLXArray(Float(2)), scale.asType(.float32) - 127).expandedDimensions(axis: -1)
        let direct = expected.reshaped(rows, dim).take(MLXArray(ids.map(Int32.init)), axis: 0)
        #expect(abs(a - direct).max().item(Float.self) == 0)
    }

    @Test("The fused Sinkhorn kernel matches the unfused loop")
    func sinkhornKernel() {
        let comb = softmax(MLXRandom.normal([3, 5, 4, 4], key: MLXRandom.key(4)), axis: -1) + 1e-6
        let fused = deepseekV41Sinkhorn(comb, iters: 20, eps: 1e-6)
        let loop = deepseekV41Sinkhorn(comb, iters: 20, eps: 1e-6, fused: false)
        #expect(abs(fused - loop).max().item(Float.self) < 1e-6)
        // Doubly stochastic up to eps.
        #expect(abs(loop.sum(axis: -1) - 1).max().item(Float.self) < 1e-3)
        #expect(abs(loop.sum(axis: -2) - 1).max().item(Float.self) < 1e-3)
    }

    @Test("Engram buckets of the published config sum to its table sizes")
    func publishedBucketLayout() throws {
        let json = """
            {"model_type": "deepseek_v41", "text_config": {"engram_layer_ids": [1, 14],
             "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4,
             "engram_vocab_size": 16000000, "engram_n_heads": 8, "engram_head_dim": 256}}
            """
        let configuration = try JSONDecoder().decode(
            DeepseekV41Configuration.self, from: Data(json.utf8))
        let primes = DeepseekV41EngramHasher.bucketPrimes(configuration)
        #expect(primes.map { $0.reduce(0, +) } == [384_006_168, 384_016_682])
        #expect(primes.map(\.count) == [24, 24])
    }
}

/// Writes a safetensors file from raw little-endian bytes.
func writeSafetensors(_ url: URL, _ tensors: [(String, String, [Int], [UInt8])]) throws {
    var header = [String: Any]()
    var body = [UInt8]()
    for (name, dtype, shape, bytes) in tensors {
        header[name] = [
            "dtype": dtype, "shape": shape, "data_offsets": [body.count, body.count + bytes.count],
        ]
        body += bytes
    }
    var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    while json.count % 8 != 0 {
        json.append(0x20)
    }
    var data = Data()
    withUnsafeBytes(of: UInt64(json.count).littleEndian) { data.append(contentsOf: $0) }
    data.append(json)
    data.append(contentsOf: body)
    try data.write(to: url)
}

/// The DSpark drafter (`mtp.*`) against the reference in `scripts/parity/deepseek_v41.py`,
/// and speculative decoding with the deepseek_v41 target.
struct DeepseekV41DSparkTests {
    private struct Loaded {
        let fixture: ParityFixture
        let model: DeepseekV41Model
        let drafter: DeepseekV41DSparkDraftModel
        let expected: [String: MLXArray]
    }

    private func load() async throws -> Loaded {
        let fixture = try ParityFixture("deepseek_v41")
        let model = try #require(try await fixture.loadLanguageModel() as? DeepseekV41Model)
        let drafter = try #require(
            try await DeepseekV41DSparkDraftModel.load(directory: draftDirectory(fixture)))
        let expected = try loadArrays(
            url: draftDirectory(fixture).appending(component: "expected.safetensors"))
        return Loaded(fixture: fixture, model: model, drafter: drafter, expected: expected)
    }

    /// The `mtp.*` tensors live in a sibling fixture, which keeps each under 1 MB.
    private func draftDirectory(_ fixture: ParityFixture) -> URL {
        fixture.directory.deletingLastPathComponent().appending(component: "deepseek_v41_dspark")
    }

    /// Draft logits after seeding the windows with the prompt in two chunks.
    private func draft(_ loaded: Loaded, drafter: DeepseekV41DSparkDraftModel) throws -> (
        MLXArray, MTPDraft
    ) {
        var request = LMOutput.State()
        request[mtpEmitFlagKey] = true
        request[mtpHiddenLayerIDsKey] = drafter.targetHiddenLayerIDs
        let output = loaded.model(
            LMInput.Text(tokens: loaded.fixture.inputIDs), cache: nil, state: request)
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        var state = drafter.makeState(parameters: nil)
        for (range, isFinal) in [(0 ..< 10, false), (10 ..< hidden.dim(1), true)] {
            drafter.prepareDrafterChunk(
                target: loaded.model, shiftedTokens: loaded.fixture.inputIDs[0..., range],
                targetHidden: hidden[0..., range], isFinal: isFinal, state: &state,
                sampler: ArgMaxSampler())
        }
        let bonus = try #require(loaded.expected["bonus"])
        let draft = drafter.draftBlock(
            target: loaded.model, lastToken: bonus, lastHidden: hidden[0..., (-1)...],
            sharedKV: [:], positionDeltas: nil, queryOffset: hidden.dim(1), blockSize: 6,
            state: &state, sampler: ArgMaxSampler())
        #expect(state.nextPosition == hidden.dim(1))
        return (hidden, draft)
    }

    @Test("Target layer captures and DSpark draft logits match the reference")
    func draftMatchesReference() async throws {
        let loaded = try await load()
        let (hidden, draft) = try draft(loaded, drafter: loaded.drafter)
        let reference = try #require(loaded.expected["hidden"])
        #expect(abs(hidden - reference).max().item(Float.self) < 1e-4)
        let logits = try #require(loaded.expected["draft_logits"])
        let difference = abs(draft.logits - logits).max().item(Float.self)
        print("deepseek_v41 DSpark draft logits max |diff| = \(difference)")
        #expect(difference < 1e-4)
        #expect(
            draft.tokens.asArray(Int32.self) == loaded.expected["draft_tokens"]!.asArray(Int32.self)
        )
        #expect(loaded.drafter.isCompatible(with: loaded.model))
    }

    @Test("The drafter registry loads DSpark from mtp.* tensors")
    func registryLoadsDrafter() async throws {
        let loaded = try await load()
        await DeepseekV41DSparkRegistration.register()
        let directory = draftDirectory(loaded.fixture)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let model = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: data, modelType: "deepseek_v41")
        try await loadWeights(modelDirectory: directory, model: model)
        let drafter = try #require(model as? DeepseekV41DSparkDraftModel)
        let (_, draft) = try draft(loaded, drafter: drafter)
        let logits = try #require(loaded.expected["draft_logits"])
        #expect(abs(draft.logits - logits).max().item(Float.self) < 1e-4)
    }

    @Test("Greedy DSpark speculative decoding reproduces greedy decoding", arguments: [2, 4, 6])
    func speculativeMatchesGreedy(blockSize: Int) async throws {
        let loaded = try await load()
        let input = LMInput(tokens: loaded.fixture.inputIDs[0])
        let parameters = GenerateParameters(maxTokens: 20, temperature: 0)
        let expected = try greedy(loaded.model, input: input, parameters: parameters)

        var speculative = try MTPSpeculativeTokenIterator(
            input: input, mainModel: loaded.model, drafter: loaded.drafter,
            parameters: parameters, blockSize: blockSize)
        var tokens = [Int]()
        while let token = speculative.next() { tokens.append(token) }

        #expect(expected.count == 20)
        #expect(tokens == expected)
        #expect(speculative.passthroughReason == nil)
        #expect(speculative.proposedCount > 0)
        let state = try #require(speculative.drafterState)
        // The windows hold every committed position, except that a final single-token step
        // without room for a draft runs the target alone.
        #expect(speculative.mainCache[0].offset - state.nextPosition <= 1)
    }

    @Test(
        "Partially accepted drafts rewind the target's window, compressor and Engram state",
        arguments: [2, 3, 6])
    func partialAcceptanceMatchesGreedy(blockSize: Int) async throws {
        let loaded = try await load()
        let input = LMInput(tokens: loaded.fixture.inputIDs[0])
        let parameters = GenerateParameters(maxTokens: 20, temperature: 0)
        let expected = try greedy(loaded.model, input: input, parameters: parameters)

        let drafter = ScriptedContinuationDrafter(
            promptLength: input.text.tokens.size, continuation: expected,
            vocabularySize: loaded.model.vocabularySize)
        var speculative = try MTPSpeculativeTokenIterator(
            input: input, mainModel: loaded.model, drafter: drafter,
            parameters: parameters, blockSize: blockSize)
        var tokens = [Int]()
        while let token = speculative.next() { tokens.append(token) }

        #expect(tokens == expected)
        #expect(speculative.acceptedCount > 0)
        #expect(speculative.acceptedCount < speculative.proposedCount)
    }

    private func greedy(
        _ model: DeepseekV41Model, input: LMInput, parameters: GenerateParameters
    ) throws -> [Int] {
        var iterator = try TokenIterator(input: input, model: model, parameters: parameters)
        var tokens = [Int]()
        while let token = iterator.next() { tokens.append(token) }
        return tokens
    }
}

/// Proposes the known greedy continuation, corrupting every fourth draft.
final class ScriptedContinuationDrafter: Module, MTPDrafterModel {
    let promptLength: Int
    let continuation: [Int]
    let vocabularySize: Int
    let requiresSharedTargetKV = false

    init(promptLength: Int, continuation: [Int], vocabularySize: Int) {
        self.promptLength = promptLength
        self.continuation = continuation
        self.vocabularySize = vocabularySize
        super.init()
    }

    func isCompatible(with target: any LanguageModel) -> Bool { true }

    func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
        queryOffset: Int, blockSize: Int, sampler: any LogitSampler
    ) -> MTPDraft {
        let start = queryOffset - promptLength + 1
        let tokens = (0 ..< (blockSize - 1)).map { i -> Int in
            let index = start + i
            let token = index < continuation.count ? continuation[index] : 0
            return index % 4 == 3 ? (token + 1) % vocabularySize : token
        }
        let logits = MLXArray.zeros([1, tokens.count, vocabularySize])
        return MTPDraft(tokens: MLXArray(tokens.map(Int32.init))[.newAxis], logits: logits)
    }
}
