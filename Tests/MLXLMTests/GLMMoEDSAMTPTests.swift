import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The GLM-MoE-DSA MTP head on the tiny fixture from `scripts/parity/glm_moe_dsa.py`.
struct GLMMoEDSAMTPTests {
    private struct Loaded {
        let fixture: ParityFixture
        let model: GLMMoEDSAModel
        let drafter: GLMMoEDSAMTPDraftModel
        let expected: [String: MLXArray]
    }

    private func load() async throws -> Loaded {
        let fixture = try ParityFixture("glm_moe_dsa")
        let model = try #require(try await fixture.loadLanguageModel() as? GLMMoEDSAModel)
        let drafter = try #require(
            try await GLMMoEDSAMTPDraftModel.load(directory: fixture.directory))
        let expected = try loadArrays(
            url: fixture.directory.appending(component: "expected.safetensors"))
        return Loaded(fixture: fixture, model: model, drafter: drafter, expected: expected)
    }

    @Test("MTP head matches transformers' MtpLayer on the target's hidden states")
    func mtpMatchesReference() async throws {
        let loaded = try await load()
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        let output = loaded.model(
            LMInput.Text(tokens: loaded.fixture.inputIDs), cache: nil, state: state)
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        let referenceHidden = try #require(loaded.expected["hidden"])
        #expect(abs(hidden - referenceHidden).max().item(Float.self) < 1e-4)

        let ids = try #require(loaded.expected["mtp_input_ids"])
        let mtpHidden = loaded.drafter(
            inputEmbeddings: loaded.model.mtpEmbedding(ids), hidden: hidden,
            cache: KVCacheSimple())
        let logits = loaded.model.lmHead(mtpHidden)
        let reference = try #require(loaded.expected["mtp_logits"])
        #expect(abs(logits - reference).max().item(Float.self) < 1e-4)
        #expect(loaded.drafter.isCompatible(with: loaded.model))
    }

    @Test("The drafter registry loads only the MTP layer from the target checkpoint")
    func registryLoadsHeadFromTargetCheckpoint() async throws {
        let loaded = try await load()
        await GLMMoEDSAMTPRegistration.register()
        let data = try Data(
            contentsOf: loaded.fixture.directory.appending(component: "config.json"))
        let drafter = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: data, modelType: "glm_moe_dsa")
        try await loadWeights(modelDirectory: loaded.fixture.directory, model: drafter)
        let ids = try #require(loaded.expected["mtp_input_ids"])
        let hidden = try #require(loaded.expected["hidden"])
        let head = try #require(drafter as? GLMMoEDSAMTPDraftModel)
        let logits = loaded.model.lmHead(
            head(
                inputEmbeddings: loaded.model.mtpEmbedding(ids), hidden: hidden,
                cache: KVCacheSimple()))
        let reference = try #require(loaded.expected["mtp_logits"])
        #expect(abs(logits - reference).max().item(Float.self) < 1e-4)
    }

    @Test("Greedy MTP speculative decoding reproduces greedy decoding", arguments: [2, 3, 4])
    func speculativeMatchesGreedy(blockSize: Int) async throws {
        let loaded = try await load()
        let input = LMInput(tokens: loaded.fixture.inputIDs[0])
        let parameters = GenerateParameters(maxTokens: 24, temperature: 0)

        var baseline = try TokenIterator(
            input: input, model: loaded.model, parameters: parameters)
        var expected = [Int]()
        while let token = baseline.next() { expected.append(token) }

        var speculative = try MTPSpeculativeTokenIterator(
            input: input, mainModel: loaded.model, drafter: loaded.drafter,
            parameters: parameters, blockSize: blockSize)
        var tokens = [Int]()
        while let token = speculative.next() { tokens.append(token) }

        #expect(expected.count == 24)
        #expect(tokens == expected)
        #expect(speculative.blockSize == blockSize)
        #expect(speculative.passthroughReason == nil)
        #expect(speculative.proposedCount > 0)
    }

    @Test(
        "Partially accepted drafts keep the sparse-attention target cache exact",
        arguments: [2, 3, 4, 6])
    func partialAcceptanceMatchesGreedy(blockSize: Int) async throws {
        let loaded = try await load()
        let input = LMInput(tokens: loaded.fixture.inputIDs[0])
        let parameters = GenerateParameters(maxTokens: 24, temperature: 0)

        var baseline = try TokenIterator(
            input: input, model: loaded.model, parameters: parameters)
        var expected = [Int]()
        while let token = baseline.next() { expected.append(token) }

        let drafter = ScriptedDrafter(
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

    @Test("An MTP head that agrees with the target is accepted and stays aligned")
    func agreeingHeadIsAccepted() async throws {
        let loaded = try await load()
        makeBigram(loaded.model, drafter: loaded.drafter)
        let input = LMInput(tokens: loaded.fixture.inputIDs[0])
        let parameters = GenerateParameters(maxTokens: 23, temperature: 0)

        var baseline = try TokenIterator(
            input: input, model: loaded.model, parameters: parameters)
        var expected = [Int]()
        while let token = baseline.next() { expected.append(token) }

        var speculative = try MTPSpeculativeTokenIterator(
            input: input, mainModel: loaded.model, drafter: loaded.drafter,
            parameters: parameters, blockSize: 4)
        var tokens = [Int]()
        while let token = speculative.next() { tokens.append(token) }

        #expect(tokens == expected)
        #expect(speculative.acceptedCount * 10 >= speculative.proposedCount * 9)
        let state = try #require(speculative.drafterState)
        #expect(state.nextPosition == state.cache[0].offset)
        #expect(state.cache[0].offset == speculative.mainCache[0].offset)
    }

    /// Zeroes every layer's output projections so the target predicts from the current token
    /// only, then turns the MTP head into the same function of its input token.
    private func makeBigram(_ model: GLMMoEDSAModel, drafter: GLMMoEDSAMTPDraftModel) {
        func zeroOutputs(_ module: Module) {
            let zeroed = module.parameters().flattened().compactMap { key, value in
                key.hasSuffix("o_proj.weight") || key.hasSuffix("down_proj.weight")
                    ? (key, MLXArray.zeros(like: value)) : nil
            }
            module.update(parameters: ModuleParameters.unflattened(zeroed))
        }
        zeroOutputs(model)
        zeroOutputs(drafter)
        let hidden = model.configuration.hiddenSize
        drafter.update(
            parameters: ModuleParameters.unflattened([
                "eh_proj.weight": concatenated(
                    [MLXArray.eye(hidden), MLXArray.zeros([hidden, hidden])], axis: 1),
                "enorm.weight": MLXArray.ones([hidden]),
                "shared_head.norm.weight": model.model.norm.weight,
            ]))
        eval(model, drafter)
    }
}

/// Proposes the known greedy continuation, corrupting every fourth draft.
private final class ScriptedDrafter: Module, MTPDrafterModel {
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
        // `queryOffset` is the bonus token's position; continuation[0] sits at `promptLength`.
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
