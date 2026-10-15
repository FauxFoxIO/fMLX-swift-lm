// Copyright © 2026 Faux Fox, Inc.

import Foundation

/// One named option of a choice question.
public struct DecisionOption: Sendable, Hashable, Codable {
    public let name: String
    /// Optional text that tells the model what the option means.
    public let description: String?

    public init(_ name: String, description: String? = nil) {
        self.name = name
        self.description = description
    }
}

public enum DecisionQuestionKind: Sendable, Hashable {
    /// Pick one of two or more named options.
    case choice([DecisionOption])
    /// Rate on 2 to 10 ordered levels, lowest first. Each level is its own description.
    case score(levels: [String])
    /// A statement that holds or does not. The descriptions are optional.
    case yesNo(whenTrue: String? = nil, whenFalse: String? = nil)
}

/// A typed question answered with a probability for every outcome, in one forward pass.
public struct DecisionQuestion: Sendable, Hashable {
    /// Names the question. GLiNER2 reads it as the task name, so a short descriptive word works best.
    public var id: String
    public var instructions: String
    public var kind: DecisionQuestionKind

    public init(id: String, instructions: String, kind: DecisionQuestionKind) {
        self.id = id
        self.instructions = instructions
        self.kind = kind
    }

    /// Outcome keys in distribution order: option names, score levels, or `yes` then `no`.
    public var outcomes: [String] {
        switch kind {
        case .choice(let options): options.map(\.name)
        case .score(let levels): levels
        case .yesNo: ["yes", "no"]
        }
    }

    public func validate() throws {
        switch kind {
        case .choice(let options):
            guard options.count >= 2 else {
                throw DecisionError.invalidQuestion(id, "a choice needs at least two options")
            }
        case .score(let levels):
            guard (2 ... 10).contains(levels.count) else {
                throw DecisionError.invalidQuestion(id, "a score needs 2 to 10 levels")
            }
        case .yesNo:
            break
        }
        let outcomes = outcomes
        guard Set(outcomes).count == outcomes.count, !outcomes.contains(where: \.isEmpty) else {
            throw DecisionError.invalidQuestion(id, "outcome names must be unique and not empty")
        }
    }
}

/// One question's distribution over its outcomes.
public struct DecisionDistribution: Sendable, Hashable {
    public let questionID: String
    /// Same order as ``DecisionQuestion/outcomes``.
    public let outcomes: [String]
    /// Sums to 1.
    public let probabilities: [Double]
    /// The model's raw scores in outcome order, before calibration.
    public let logits: [Double]
    /// The temperature that divided the logits before the softmax.
    public let temperature: Double

    public init(
        questionID: String, outcomes: [String], probabilities: [Double], logits: [Double],
        temperature: Double
    ) {
        self.questionID = questionID
        self.outcomes = outcomes
        self.probabilities = probabilities
        self.logits = logits
        self.temperature = temperature
    }

    public var top: String { outcomes[topIndex] }
    public var confidence: Double { probabilities[topIndex] }

    public func probability(of outcome: String) -> Double? {
        outcomes.firstIndex(of: outcome).map { probabilities[$0] }
    }

    private var topIndex: Int {
        probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
    }
}

public enum DecisionImage: Sendable {
    /// A local image file; EXIF orientation is applied.
    case file(URL)
    /// An encoded image (PNG, JPEG, HEIC, ...); EXIF orientation is applied.
    case encoded(Data)
    /// Packed RGB, row-major, eight bits per channel. Read as given.
    case rgb(bytes: [UInt8], width: Int, height: Int)
}

public enum DecisionAudio: Sendable {
    /// A local audio file, mixed to mono and resampled to 16 kHz.
    case file(URL)
    /// Interleaved float samples in [-1, 1].
    case pcm(samples: [Float], sampleRate: Int, channels: Int = 1)
    /// Interleaved 16-bit PCM, scaled by 1/32768.
    case pcm16(samples: [Int16], sampleRate: Int, channels: Int = 1)
}

/// A state and the questions asked about it. A request carries images or audio, not both.
public struct DecisionRequest: Sendable {
    /// Text or serialized JSON. `nil` when the media are the whole state.
    public var state: String?
    public var questions: [DecisionQuestion]
    public var images: [DecisionImage]
    public var audio: DecisionAudio?

    public init(
        state: String?, questions: [DecisionQuestion], images: [DecisionImage] = [],
        audio: DecisionAudio? = nil
    ) {
        self.state = state
        self.questions = questions
        self.images = images
        self.audio = audio
    }
}

public struct DecisionResult: Sendable {
    /// One per question, in request order.
    public let distributions: [DecisionDistribution]
    /// Every position the encoder read, media included.
    public let inputTokens: Int
    public let artifact: DecisionArtifact
}

/// The checkpoint identity is supplied by the installer; loaders do not download files.
public struct DecisionArtifact: Codable, Sendable, Hashable {
    public let identifier: String
    public let revision: String

    public init(identifier: String, revision: String) {
        self.identifier = identifier
        self.revision = revision
    }
}

public enum DecisionModality: String, Codable, Sendable, Hashable, CaseIterable {
    case text, image, audio
}

/// The dtype weights run in. `.checkpoint` keeps the stored dtype; `.float32` upcasts float tensors.
public enum DecisionPrecision: String, Codable, Sendable {
    case checkpoint, float32
}

/// A loaded decision model. Calls are serialized; each answers every question of a request.
public protocol DecisionModel: AnyObject, Sendable {
    var artifact: DecisionArtifact { get }
    var modalities: Set<DecisionModality> { get }
    func decide(_ request: DecisionRequest) async throws -> DecisionResult
    func decide(_ requests: [DecisionRequest]) async throws -> [DecisionResult]
}

extension DecisionModel {
    public func decide(_ request: DecisionRequest) async throws -> DecisionResult {
        guard let result = try await decide([request]).first else {
            throw DecisionError.invalidInput("the model returned no result")
        }
        return result
    }
}

public enum DecisionError: Error, LocalizedError, Sendable {
    case invalidQuestion(String, String)
    case invalidInput(String)
    case unsupportedInput(String)
    case contextOverflow(String)
    case invalidConfiguration(String)
    case invalidWeights(String)
    case nonFiniteOutput

    public var errorDescription: String? {
        switch self {
        case .invalidQuestion(let id, let reason): "Invalid decision question \(id): \(reason)"
        case .invalidInput(let reason): "Invalid decision input: \(reason)"
        case .unsupportedInput(let reason): "Unsupported decision input: \(reason)"
        case .contextOverflow(let reason): "Decision input does not fit: \(reason)"
        case .invalidConfiguration(let reason): "Invalid decision model configuration: \(reason)"
        case .invalidWeights(let reason): "Invalid decision model weights: \(reason)"
        case .nonFiniteOutput: "The decision model produced a non-finite score."
        }
    }
}

func softmax(_ values: [Double]) -> [Double] {
    guard let peak = values.max() else { return [] }
    let exps = values.map { Foundation.exp($0 - peak) }
    let total = exps.reduce(0, +)
    return exps.map { $0 / total }
}
