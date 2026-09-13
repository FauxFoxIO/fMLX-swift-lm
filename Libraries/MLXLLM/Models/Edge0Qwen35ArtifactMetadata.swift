// Copyright © 2026 Faux Fox.

import Foundation

enum Edge0Qwen35ArtifactKind: String {
    case lora
    case prerouter
}

struct Edge0Qwen35ArtifactMetadata: Decodable, Equatable {
    let model: String
    let kind: String
    let topK: String
    let rank: String
    let alpha: String
    let source: String
    let sourceMD5: String
    let converted: String
    let formatVersion: String
    let owners: String?

    enum CodingKeys: String, CodingKey {
        case model, kind, source, converted, owners
        case topK = "K"
        case rank = "r"
        case alpha
        case sourceMD5 = "source_md5"
        case formatVersion = "format_version"
    }
}

enum Edge0Qwen35ArtifactMetadataError: Error, Equatable, LocalizedError {
    case missingNestedMetadata
    case malformedNestedMetadata
    case incompatibleMetadata

    var errorDescription: String? {
        switch self {
        case .missingNestedMetadata:
            "The Edge0 sidecar is missing its nested metadata"
        case .malformedNestedMetadata:
            "The Edge0 sidecar has malformed nested metadata"
        case .incompatibleMetadata:
            "The Edge0 sidecar metadata does not match the pinned Qwen35 profile"
        }
    }
}

func validateEdge0Qwen35ArtifactMetadata(
    _ metadata: [String: String],
    kind: Edge0Qwen35ArtifactKind,
    owners: [Int]? = nil
) throws {
    guard let nested = metadata["__metadata__"] else {
        throw Edge0Qwen35ArtifactMetadataError.missingNestedMetadata
    }
    guard let data = nested.data(using: .utf8),
        let value = try? JSONDecoder().decode(Edge0Qwen35ArtifactMetadata.self, from: data)
    else {
        throw Edge0Qwen35ArtifactMetadataError.malformedNestedMetadata
    }

    let expectedSourceMD5: String
    switch kind {
    case .lora:
        expectedSourceMD5 = "dbdef1af692986ad1937562c0d2aab7f"
    case .prerouter:
        expectedSourceMD5 = "df15dff55499f6dc0343d54379b03e32"
    }
    let expectedOwners = owners.map { String(describing: $0) }
    guard value.model == "edge0-35b", value.kind == kind.rawValue,
        value.topK == "4", value.rank == "16", value.alpha == "32",
        value.formatVersion == "1", value.sourceMD5 == expectedSourceMD5,
        !value.source.isEmpty, !value.converted.isEmpty,
        value.owners == expectedOwners
    else {
        throw Edge0Qwen35ArtifactMetadataError.incompatibleMetadata
    }
}
