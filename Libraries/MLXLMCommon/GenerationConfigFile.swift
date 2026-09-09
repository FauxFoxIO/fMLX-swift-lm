// Copyright © 2024 Apple Inc.

import Foundation

/// JSON wrapper for `generation_config.json` file.
///
/// This file can override values from `config.json`, particularly `eos_token_id`.
/// Following mlx-lm Python behavior, if `generation_config.json` exists and contains
/// `eos_token_id`, it takes precedence over the value in `config.json`.
public struct GenerationConfigFile: Codable, Sendable {
    public var eosTokenIds: IntOrIntArray?
    public var stopStrings: Set<String>
    public var doSample: Bool?
    public var temperature: Float?
    public var topP: Float?
    public var topK: Int?

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
        case stopStrings = "stop_strings"
        case stop
        case doSample = "do_sample"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
    }

    public init(
        eosTokenIds: IntOrIntArray? = nil,
        stopStrings: Set<String> = [],
        doSample: Bool? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil
    ) {
        self.eosTokenIds = eosTokenIds
        self.stopStrings = stopStrings
        self.doSample = doSample
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        eosTokenIds = try container.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenIds)
        doSample = try container.decodeIfPresent(Bool.self, forKey: .doSample)
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)

        stopStrings = []
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stopStrings))
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stop))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(eosTokenIds, forKey: .eosTokenIds)
        try container.encodeIfPresent(doSample, forKey: .doSample)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(topK, forKey: .topK)
        if !stopStrings.isEmpty {
            try container.encode(stopStrings.sorted(), forKey: .stopStrings)
        }
    }

    private static func decodeStringSet(
        from container: KeyedDecodingContainer<CodingKeys>,
        forKey key: CodingKeys
    ) -> Set<String> {
        if let values = try? container.decode([String].self, forKey: key) {
            return Set(values)
        }
        if let value = try? container.decode(String.self, forKey: key) {
            return [value]
        }
        return []
    }
}
