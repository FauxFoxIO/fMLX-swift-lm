// Copyright © 2026 Faux Fox.

import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct FMLXRepositoryFile: Codable, Hashable, Sendable {
    public let path: String
    public let sizeBytes: Int64
    public let sha256: String?

    public init(path: String, sizeBytes: Int64, sha256: String?) {
        self.path = path
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }
}

public struct FMLXRepositorySnapshot: Codable, Hashable, Sendable {
    public let repositoryID: String
    public let revision: String
    public let files: [FMLXRepositoryFile]
    public let modelType: String?

    public init(
        repositoryID: String, revision: String, files: [FMLXRepositoryFile], modelType: String?
    ) {
        self.repositoryID = repositoryID
        self.revision = revision
        self.files = files
        self.modelType = modelType
    }
}

/// Provider client for public Hugging Face model metadata and immutable snapshot files.
public struct HuggingFaceModelHub: @unchecked Sendable {
    public let baseURL: URL
    private let session: URLSession
    let transferConfiguration: URLSessionConfiguration

    public init(
        baseURL: URL = URL(string: "https://huggingface.co")!, session: URLSession = .shared,
        transferConfiguration: URLSessionConfiguration = .ephemeral
    ) {
        self.baseURL = baseURL
        self.session = session
        self.transferConfiguration = transferConfiguration
    }

    public func search(_ query: String, limit: Int = 50) async throws -> [FMLXModelSearchResult] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return [] }
        var components = URLComponents(
            url: baseURL.appendingPathComponent("api/models"), resolvingAgainstBaseURL: false)
        components?.queryItems = [
            URLQueryItem(name: "search", value: query),
            URLQueryItem(name: "filter", value: "mlx"),
            URLQueryItem(name: "sort", value: "trendingScore"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: String(min(max(limit, 1), 100))),
            URLQueryItem(name: "full", value: "true"),
            URLQueryItem(name: "config", value: "true"),
        ]
        guard let url = components?.url else {
            throw FMLXModelManagementError.invalidRepositoryResponse
        }
        let records: [HubModel] = try await request(url)
        return records.compactMap { record in
            guard let id = record.identifier, Self.isValidRepositoryID(id) else { return nil }
            let modelType = record.config?.modelType
            if let modelType, !Self.supportedModelTypes.contains(modelType) { return nil }
            return FMLXModelSearchResult(
                repositoryID: id, displayName: id.split(separator: "/").last.map(String.init) ?? id,
                revision: record.sha, sizeBytes: record.usedStorage ?? 0,
                parameterCount: record.safetensors?.parameterCount,
                downloads: record.downloads ?? 0, likes: record.likes ?? 0,
                modelType: modelType)
        }
    }

    public func repository(_ id: String, revision: String = "main") async throws
        -> FMLXRepositorySnapshot
    {
        guard Self.isValidRepositoryID(id), Self.isValidRevision(revision) else {
            throw FMLXModelManagementError.invalidRepositoryID
        }
        var components = URLComponents(
            url: baseURL.appendingPathComponent("api/models/\(id)/revision/\(revision)"),
            resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "blobs", value: "true")]
        guard let url = components?.url else {
            throw FMLXModelManagementError.invalidRepositoryResponse
        }
        let record: HubModel = try await request(url)
        guard let commit = record.sha, !commit.isEmpty else {
            throw FMLXModelManagementError.invalidRepositoryResponse
        }
        let files = try (record.siblings ?? []).compactMap { sibling -> FMLXRepositoryFile? in
            guard Self.shouldInstall(sibling.filename) else { return nil }
            guard Self.isSafeRelativePath(sibling.filename) else {
                throw FMLXModelManagementError.unsafePath(sibling.filename)
            }
            let size = sibling.lfs?.size ?? sibling.size ?? 0
            guard size >= 0 else { throw FMLXModelManagementError.invalidRepositoryResponse }
            return FMLXRepositoryFile(
                path: sibling.filename, sizeBytes: size, sha256: sibling.lfs?.sha256)
        }
        guard files.contains(where: { $0.path == "config.json" }),
            files.contains(where: { $0.path.hasSuffix(".safetensors") })
        else {
            throw FMLXModelManagementError.incompleteCheckpoint(
                "config.json and safetensors weights are required")
        }
        if let type = record.config?.modelType, !Self.supportedModelTypes.contains(type) {
            throw FMLXModelManagementError.unsupportedModelType(type)
        }
        return FMLXRepositorySnapshot(
            repositoryID: id, revision: commit, files: files, modelType: record.config?.modelType)
    }

    public func fileURL(repositoryID: String, revision: String, path: String) throws -> URL {
        guard Self.isValidRepositoryID(repositoryID), Self.isValidRevision(revision),
            Self.isSafeRelativePath(path)
        else { throw FMLXModelManagementError.invalidRepositoryID }
        return baseURL.appendingPathComponent(repositoryID)
            .appendingPathComponent("resolve").appendingPathComponent(revision)
            .appendingPathComponent(path)
    }

    public static let supportedModelTypes: Set<String> = [
        "llama", "mistral", "qwen3", "qwen3_5", "qwen3_5_text", "qwen3_5_moe",
        "qwen3_5_mtp",
    ]

    public static func isValidRepositoryID(_ id: String) -> Bool {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2
            && parts.allSatisfy { part in
                !part.isEmpty && part != "." && part != ".."
                    && part.allSatisfy { $0.isLetter || $0.isNumber || "-_.".contains($0) }
            }
    }

    static func isValidRevision(_ revision: String) -> Bool {
        !revision.isEmpty && revision != "." && revision != ".."
            && !revision.contains("\\") && !revision.contains("//")
    }

    static func isSafeRelativePath(_ path: String) -> Bool {
        !path.isEmpty && !path.hasPrefix("/") && !path.contains("\\")
            && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
                !$0.isEmpty && $0 != "." && $0 != ".."
            }
    }

    private static func shouldInstall(_ path: String) -> Bool {
        guard isSafeRelativePath(path), !path.hasPrefix(".") else { return false }
        let name = (path as NSString).lastPathComponent.lowercased()
        let ext = (name as NSString).pathExtension
        return ["json", "safetensors", "jinja", "txt", "model"].contains(ext)
            || name == "license" || name.hasPrefix("license.") || name == "readme.md"
    }

    private func request<Value: Decodable>(_ url: URL) async throws -> Value {
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        request.setValue("fMLX-swift-lm", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else {
            throw FMLXModelManagementError.invalidRepositoryResponse
        }
        guard (200 ..< 300).contains(response.statusCode) else {
            throw FMLXModelManagementError.downloadFailed(statusCode: response.statusCode)
        }
        return try JSONDecoder().decode(Value.self, from: data)
    }
}

private struct HubModel: Decodable {
    struct Configuration: Decodable {
        let modelType: String?
        enum CodingKeys: String, CodingKey { case modelType = "model_type" }
    }
    struct Safetensors: Decodable {
        let total: Int64?
        let parameters: [String: Int64]?
        var parameterCount: Int64? {
            guard let parameters else { return total }
            return parameters.values.reduce(0) { partial, value in
                let sum = partial.addingReportingOverflow(value)
                return sum.overflow ? Int64.max : sum.partialValue
            }
        }
    }
    struct Sibling: Decodable {
        struct LFS: Decodable {
            let sha256: String?
            let size: Int64?
        }
        let filename: String
        let size: Int64?
        let lfs: LFS?
        enum CodingKeys: String, CodingKey {
            case filename = "rfilename"
            case size
            case lfs
        }
    }
    let id: String?
    let modelID: String?
    let sha: String?
    let downloads: Int?
    let likes: Int?
    let usedStorage: Int64?
    let config: Configuration?
    let safetensors: Safetensors?
    let siblings: [Sibling]?
    var identifier: String? { id ?? modelID }
}
