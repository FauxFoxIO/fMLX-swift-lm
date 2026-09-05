// Copyright © 2026 Faux Fox.

import Foundation

public enum FMLXModelManagementError: LocalizedError, Equatable, Sendable {
    case invalidRepositoryID
    case invalidRepositoryResponse
    case unsupportedModelType(String)
    case unsafePath(String)
    case downloadFailed(statusCode: Int)
    case sizeMismatch(file: String, expected: Int64, actual: Int64)
    case checksumMismatch(file: String)
    case incompleteCheckpoint(String)
    case modelAlreadyInstalled(String)
    case modelNotInstalled(String)
    case downloadNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID: "The model repository identifier is invalid."
        case .invalidRepositoryResponse: "The model repository returned invalid metadata."
        case .unsupportedModelType(let type): "fMLX does not support the model type '\(type)'."
        case .unsafePath(let path): "The model repository contains an unsafe path: \(path)"
        case .downloadFailed(let status): "The model download failed with HTTP status \(status)."
        case .sizeMismatch(let file, let expected, let actual):
            "\(file) is incomplete (expected \(expected) bytes, found \(actual))."
        case .checksumMismatch(let file): "\(file) failed SHA-256 validation."
        case .incompleteCheckpoint(let reason): "The checkpoint is incomplete: \(reason)"
        case .modelAlreadyInstalled(let id): "The model '\(id)' is already installed."
        case .modelNotInstalled(let id): "The model '\(id)' is not installed."
        case .downloadNotFound(let id): "The model download '\(id)' was not found."
        }
    }
}

public struct FMLXInstalledModel: Codable, Hashable, Identifiable, Sendable {
    public let id: String
    public let repositoryID: String
    public let displayName: String
    public let directory: URL
    public let sizeBytes: Int64
    public let modelType: String
    public let revision: String?
    public let supportsEmbeddedMTP: Bool

    public init(
        id: String, repositoryID: String, displayName: String, directory: URL,
        sizeBytes: Int64, modelType: String, revision: String?, supportsEmbeddedMTP: Bool
    ) {
        self.id = id
        self.repositoryID = repositoryID
        self.displayName = displayName
        self.directory = directory
        self.sizeBytes = sizeBytes
        self.modelType = modelType
        self.revision = revision
        self.supportsEmbeddedMTP = supportsEmbeddedMTP
    }
}

public struct FMLXModelSearchResult: Codable, Hashable, Identifiable, Sendable {
    public var id: String { repositoryID }
    public let repositoryID: String
    public let displayName: String
    public let revision: String?
    public let sizeBytes: Int64
    public let parameterCount: Int64?
    public let downloads: Int
    public let likes: Int
    public let modelType: String?

    public init(
        repositoryID: String, displayName: String, revision: String?, sizeBytes: Int64,
        parameterCount: Int64?, downloads: Int, likes: Int, modelType: String?
    ) {
        self.repositoryID = repositoryID
        self.displayName = displayName
        self.revision = revision
        self.sizeBytes = sizeBytes
        self.parameterCount = parameterCount
        self.downloads = downloads
        self.likes = likes
        self.modelType = modelType
    }
}

public enum FMLXModelDownloadStatus: String, Codable, Hashable, Sendable {
    case pending
    case downloading
    case validating
    case completed
    case failed
    case cancelled
}

public struct FMLXModelDownload: Codable, Hashable, Identifiable, Sendable {
    public let id: UUID
    public let repositoryID: String
    public let revision: String?
    public let status: FMLXModelDownloadStatus
    public let downloadedBytes: Int64
    public let totalBytes: Int64
    public let currentFile: String?
    public let errorMessage: String?
    public let canCancel: Bool
    public let canRetry: Bool
    public let updatedAt: Date

    public var progress: Double {
        guard totalBytes > 0 else { return 0 }
        return min(max(Double(downloadedBytes) / Double(totalBytes), 0), 1)
    }

    public init(
        id: UUID, repositoryID: String, revision: String?, status: FMLXModelDownloadStatus,
        downloadedBytes: Int64, totalBytes: Int64, currentFile: String? = nil,
        errorMessage: String? = nil, canCancel: Bool = false, canRetry: Bool = false,
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.repositoryID = repositoryID
        self.revision = revision
        self.status = status
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
        self.currentFile = currentFile
        self.errorMessage = errorMessage
        self.canCancel = canCancel
        self.canRetry = canRetry
        self.updatedAt = updatedAt
    }
}

public struct FMLXModelCacheInfo: Codable, Hashable, Sendable {
    public let sizeBytes: Int64
    public let capacityBytes: Int64

    public init(sizeBytes: Int64, capacityBytes: Int64) {
        self.sizeBytes = sizeBytes
        self.capacityBytes = capacityBytes
    }
}

public struct FMLXModelStoreSnapshot: Codable, Hashable, Sendable {
    public let installedModels: [FMLXInstalledModel]
    public let downloads: [FMLXModelDownload]
    public let cache: FMLXModelCacheInfo

    public init(
        installedModels: [FMLXInstalledModel], downloads: [FMLXModelDownload],
        cache: FMLXModelCacheInfo
    ) {
        self.installedModels = installedModels
        self.downloads = downloads
        self.cache = cache
    }
}
