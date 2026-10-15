// Copyright © 2026 Faux Fox, Inc.
// Port of the GLiNER2 span extractor's classification path (fastino-ai/GLiNER2, Apache-2.0).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `config.json` of a `gliner2` checkpoint such as fastino/GLiNER2.5-Decide.
public struct GLiNER2Configuration: Codable, Sendable {
    public let modelType: String
    public let architecture: String?
    public let tokenPooling: String?
    public let encoder: DeBERTaV2Configuration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architecture
        case tokenPooling = "token_pooling"
        case encoder = "encoder_config"
    }

    func validate() throws {
        guard modelType == "gliner2", (architecture ?? "span") == "span",
            (tokenPooling ?? "first") == "first"
        else {
            throw DecisionError.invalidConfiguration(
                "only GLiNER2 span checkpoints with first-token pooling are supported")
        }
        try encoder.validate()
    }
}

/// A task's label scores: a softmax for single-label tasks, sigmoids for multi-label ones.
public struct GLiNER2Classification: Sendable, Hashable {
    public let task: String
    public let labels: [String]
    public let scores: [Double]
    public let logits: [Double]
    public let multiLabel: Bool

    public init(
        task: String, labels: [String], scores: [Double], logits: [Double], multiLabel: Bool
    ) {
        self.task = task
        self.labels = labels
        self.scores = scores
        self.logits = logits
        self.multiLabel = multiLabel
    }
}

final class GLiNER2Network: Module {
    @ModuleInfo var encoder: DeBERTaV2Model
    @ModuleInfo var classifier: [UnaryLayer]

    init(_ configuration: GLiNER2Configuration) {
        let d = configuration.encoder.hiddenSize
        _encoder.wrappedValue = DeBERTaV2Model(configuration.encoder)
        _classifier.wrappedValue = [Linear(d, 2 * d), ReLU(), Linear(2 * d, 1)]
    }

    /// Classifier logits at every task's label markers, one array per sequence.
    func logits(_ encodings: [GLiNER2Encoding]) -> [[[Float]]] {
        let longest = encodings.map(\.ids.count).max() ?? 0
        let ids = MLXArray(
            encodings.flatMap { e in
                (e.ids + Array(repeating: 0, count: longest - e.ids.count)).map(Int32.init)
            },
            [encodings.count, longest])
        let mask =
            MLXArray(Int32(0) ..< Int32(longest))[.newAxis]
            .< MLXArray(encodings.map { Int32($0.ids.count) })[0..., .newAxis]
        let hidden = encoder(ids, mask: mask)
        var gathered = [MLXArray]()
        var shape = [[Int]]()
        for (row, encoding) in encodings.enumerated() {
            let labels = encoding.markerPositions.flatMap { $0.dropFirst() }
            shape.append(encoding.markerPositions.map { $0.count - 1 })
            gathered.append(hidden[row][MLXArray(labels.map(Int32.init))])
        }
        var x = concatenated(gathered, axis: 0)
        for layer in classifier { x = layer(x) }
        let values = x.squeezed(axis: -1).asType(.float32)
        eval(values)
        var flat = values.asArray(Float.self)[...]
        return shape.map { counts in
            counts.map { count in
                defer { flat = flat.dropFirst(count) }
                return Array(flat.prefix(count))
            }
        }
    }

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (original, value) in weights {
            guard original.hasPrefix("encoder.") || original.hasPrefix("classifier.") else {
                continue
            }
            if original.hasSuffix("position_ids") { continue }
            let key =
                original
                .replacingOccurrences(of: "encoder.encoder.layer.", with: "encoder.encoder.layers.")
                .replacingOccurrences(of: ".attention.self.", with: ".attention.self_attn.")
                .replacingOccurrences(of: ".LayerNorm.", with: ".layer_norm.")
            result[key] = value
        }
        return result
    }
}

/// A loaded GLiNER2 classifier. Text only; every task of a request reads one joint sequence.
public final class GLiNER2Model: DecisionModel {
    public let artifact: DecisionArtifact
    public let configuration: GLiNER2Configuration
    public let weightBytes: Int64
    public let modalities: Set<DecisionModality> = [.text]
    /// Subword limit of one sequence; text words past it are dropped.
    public let maximumTokens: Int
    /// Padded positions per forward batch.
    public let tokenBudget: Int

    private let state: SerialAccessContainer<GLiNER2Runtime>

    fileprivate init(
        _ runtime: consuming GLiNER2Runtime, configuration: GLiNER2Configuration,
        artifact: DecisionArtifact, bytes: Int64, maximumTokens: Int, tokenBudget: Int
    ) {
        self.configuration = configuration
        self.artifact = artifact
        weightBytes = bytes
        self.maximumTokens = maximumTokens
        self.tokenBudget = tokenBudget
        state = SerialAccessContainer(consume runtime)
    }

    /// Scores every task over `text` in one forward pass, as `classify_text` does.
    public func classify(_ text: String, tasks: [GLiNER2Task]) async throws
        -> [GLiNER2Classification]
    {
        let limit = maximumTokens
        let budget = tokenBudget
        return try await state.read { runtime in
            try runtime.classify([(text, tasks)], maximumTokens: limit, tokenBudget: budget)[0].0
        }
    }

    public func decide(_ requests: [DecisionRequest]) async throws -> [DecisionResult] {
        for request in requests {
            guard request.images.isEmpty, request.audio == nil else {
                throw DecisionError.unsupportedInput("GLiNER2 reads text only")
            }
            guard !request.questions.isEmpty else {
                throw DecisionError.invalidInput("a request needs at least one question")
            }
            for question in request.questions { try question.validate() }
            guard Set(request.questions.map(\.id)).count == request.questions.count else {
                throw DecisionError.invalidInput("GLiNER2 task names must be unique")
            }
        }
        let limit = maximumTokens
        let budget = tokenBudget
        let artifact = artifact
        let inputs = requests.map { ($0.state ?? "", $0.questions.map(GLiNER2Task.init)) }
        let classified = try await state.read { runtime in
            try runtime.classify(inputs, maximumTokens: limit, tokenBudget: budget)
        }
        return zip(requests, classified).map { request, entry in
            let distributions = zip(request.questions, entry.0).map { question, result in
                DecisionDistribution(
                    questionID: question.id, outcomes: question.outcomes,
                    probabilities: result.scores, logits: result.logits, temperature: 1)
            }
            return DecisionResult(
                distributions: distributions, inputTokens: entry.1, artifact: artifact)
        }
    }
}

public enum GLiNER2Loader {
    /// Loads a local `gliner2` snapshot (PyTorch layout, as Fastino publishes it).
    public static func load(
        from directory: URL, artifact: DecisionArtifact, precision: DecisionPrecision = .checkpoint,
        maximumTokens: Int = 2048, tokenBudget: Int = 16384
    ) async throws -> GLiNER2Model {
        guard !artifact.identifier.isEmpty, !artifact.revision.isEmpty, maximumTokens >= 16,
            tokenBudget >= maximumTokens
        else {
            throw DecisionError.invalidConfiguration(
                "provide an artifact identity and valid limits")
        }
        let configuration = try JSONDecoder().decode(
            GLiNER2Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try configuration.validate()
        let tokenizer = try await NativeDecisionTokenizer.load(from: directory)
        for token in ["[P]", "[L]", "[SEP_STRUCT]", "[SEP_TEXT]", "[DESCRIPTION]"] {
            guard let id = tokenizer.id(of: token), try tokenizer.tokenize(token) == [token],
                id < configuration.encoder.vocabularySize
            else {
                throw DecisionError.invalidConfiguration("the tokenizer does not keep \(token)")
            }
        }
        let files = try DecisionWeights.files(in: directory)
        let network = GLiNER2Network(configuration)
        let weights = GLiNER2Network.sanitize(try DecisionWeights.load(files))
        try DecisionWeights.apply(
            weights, to: network, groupSize: nil, bits: nil, mode: nil, precision: precision)
        return GLiNER2Model(
            GLiNER2Runtime(network: network, tokenizer: tokenizer), configuration: configuration,
            artifact: artifact, bytes: DecisionWeights.byteCount(files),
            maximumTokens: maximumTokens,
            tokenBudget: tokenBudget)
    }
}

final class GLiNER2Runtime {
    let network: GLiNER2Network
    let tokenizer: any DecisionTokenizer

    init(network: GLiNER2Network, tokenizer: any DecisionTokenizer) {
        self.network = network
        self.tokenizer = tokenizer
    }

    func classify(
        _ inputs: [(String, [GLiNER2Task])], maximumTokens: Int, tokenBudget: Int
    ) throws -> [([GLiNER2Classification], Int)] {
        var encodings = [GLiNER2Encoding]()
        for (text, tasks) in inputs {
            guard !tasks.isEmpty else {
                throw DecisionError.invalidInput("provide at least one task")
            }
            for task in tasks {
                guard !task.name.isEmpty, task.labels.count >= 1,
                    Set(task.labels).count == task.labels.count
                else {
                    throw DecisionError.invalidInput("task \(task.name) needs unique labels")
                }
            }
            encodings.append(
                try GLiNER2Schema.encode(
                    text: text, tasks: tasks, tokenizer: tokenizer, maximumTokens: maximumTokens))
        }
        var logits = [[[Float]]](repeating: [], count: encodings.count)
        for batch in decisionBatches(lengths: encodings.map(\.ids.count), budget: tokenBudget) {
            try Task.checkCancellation()
            for (index, values) in zip(batch, network.logits(batch.map { encodings[$0] })) {
                logits[index] = values
            }
        }
        var results = [([GLiNER2Classification], Int)]()
        for ((_, tasks), (encoding, values)) in zip(inputs, zip(encodings, logits)) {
            var classifications = [GLiNER2Classification]()
            for (task, z) in zip(tasks, values) {
                let logits = z.map(Double.init)
                guard logits.allSatisfy(\.isFinite) else { throw DecisionError.nonFiniteOutput }
                let scores =
                    task.multiLabel ? logits.map { 1 / (1 + Foundation.exp(-$0)) } : softmax(logits)
                classifications.append(
                    GLiNER2Classification(
                        task: task.name, labels: task.labels, scores: scores, logits: logits,
                        multiLabel: task.multiLabel))
            }
            results.append((classifications, encoding.ids.count))
        }
        return results
    }
}
