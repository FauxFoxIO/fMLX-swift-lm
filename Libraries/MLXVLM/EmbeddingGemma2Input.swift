// Copyright © 2026 Faux Fox, Inc.

import Foundation

/// Ordered content for one EmbeddingGemma 2 embedding, independent of chat or generation.
public enum EmbeddingGemma2Content: Sendable {
    case text(String)
    case image(EmbeddingGemma2Image)
    case audio(EmbeddingGemma2Audio)
    case video(EmbeddingGemma2Video)
}

public enum EmbeddingGemma2Image: Sendable {
    case file(URL)
    /// Packed RGB, row-major, eight bits per channel.
    case rgb(bytes: [UInt8], width: Int, height: Int)
}

public enum EmbeddingGemma2Audio: Sendable {
    case file(URL)
    /// Interleaved, finite PCM samples. Channels are mixed to mono before resampling.
    case pcm(samples: [Float], sampleRate: Int, channels: Int = 1)
}

public enum EmbeddingGemma2Video: Sendable {
    case file(URL)
    /// Contiguous source frames with their presentation times in seconds.
    case frames([EmbeddingGemma2VideoFrame])
    /// Already selected frames, processed once in order without reference 1 Hz resampling.
    case keyframes([EmbeddingGemma2VideoFrame], softTokenBudget: Int)
}

public struct EmbeddingGemma2VideoFrame: Sendable {
    public let image: EmbeddingGemma2Image
    public let time: Double

    public init(image: EmbeddingGemma2Image, time: Double) {
        self.image = image
        self.time = time
    }
}

public enum EmbeddingGemma2Purpose: Sendable {
    case query(task: String = "search result")
    case document(title: String = "none")
}

public enum EmbeddingGemma2Dimension: Int, Codable, Sendable, CaseIterable {
    case d768 = 768
    case d512 = 512
    case d256 = 256
    case d128 = 128
}

/// The checkpoint identity is supplied by the installer; the loader does not download files.
public struct EmbeddingGemma2Artifact: Codable, Sendable, Equatable {
    public static let referenceRevision = "914f7f89142e33e77833254d9c9b90c3cef7303b"
    public static let pythonReferenceRevision = "3d87e88402f307efbf68e568971aa887ee7d9ed0"
    public let identifier: String
    public let revision: String

    public init(identifier: String, revision: String) {
        self.identifier = identifier
        self.revision = revision
    }
}

public struct EmbeddingGemma2Result: Sendable {
    public let values: [Float]
    public let dimension: EmbeddingGemma2Dimension
    public let artifact: EmbeddingGemma2Artifact
    public let referenceRevision: String
    public let tokenCount: Int
}

public enum EmbeddingGemma2Modality: String, Codable, Hashable, Sendable, CaseIterable {
    case text, vision, audio
}

public enum EmbeddingGemma2Error: Error, LocalizedError, Sendable {
    case invalidInput(String)
    case invalidConfiguration(String)
    case invalidWeights(String)
    case tokenLimit(count: Int, limit: Int)
    case featureCount(modality: String, expected: Int, actual: Int)
    case nonFiniteEmbedding

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let reason): return "Invalid EmbeddingGemma 2 input: \(reason)"
        case .invalidConfiguration(let reason):
            return "Invalid EmbeddingGemma 2 configuration: \(reason)"
        case .invalidWeights(let reason): return "Invalid EmbeddingGemma 2 weights: \(reason)"
        case .tokenLimit(let count, let limit):
            return
                "EmbeddingGemma 2 input has \(count) tokens; the limit is \(limit). Divide it into contiguous units."
        case .featureCount(let modality, let expected, let actual):
            return
                "EmbeddingGemma 2 \(modality) has \(actual) features for \(expected) placeholders."
        case .nonFiniteEmbedding: return "EmbeddingGemma 2 produced a non-finite or zero embedding."
        }
    }
}
