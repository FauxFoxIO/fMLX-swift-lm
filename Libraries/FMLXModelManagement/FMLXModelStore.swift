// Copyright © 2026 Faux Fox.

import CryptoKit
import Foundation

/// Durable local model catalog and download service. Model publication is atomic at directory level.
public actor FMLXModelStore {
    public struct Configuration: Sendable {
        public let modelsDirectory: URL
        public let cacheDirectory: URL
        public let cacheCapacityBytes: Int64

        public init(
            modelsDirectory: URL, cacheDirectory: URL, cacheCapacityBytes: Int64
        ) {
            self.modelsDirectory = modelsDirectory
            self.cacheDirectory = cacheDirectory
            self.cacheCapacityBytes = cacheCapacityBytes
        }
    }

    private struct InstalledMetadata: Codable {
        let repositoryID: String
        let revision: String
    }

    private struct Transaction: Codable {
        var download: FMLXModelDownload
        let snapshot: FMLXRepositorySnapshot
    }

    private let configuration: Configuration
    private let hub: HuggingFaceModelHub
    private let transactionsDirectory: URL
    private var transactions: [UUID: Transaction] = [:]
    private var tasks: [UUID: Task<Void, Never>] = [:]
    private var transfers: [UUID: ModelFileTransfer] = [:]
    private var persistedProgress: [UUID: Int64] = [:]

    public init(configuration: Configuration, hub: HuggingFaceModelHub = .init()) throws {
        guard configuration.modelsDirectory.isFileURL, configuration.cacheDirectory.isFileURL,
            configuration.cacheCapacityBytes > 0
        else { throw FMLXModelManagementError.unsafePath("Model store configuration") }
        self.configuration = configuration
        self.hub = hub
        transactionsDirectory = configuration.modelsDirectory.deletingLastPathComponent()
            .appendingPathComponent("downloads", isDirectory: true)
        try Self.ensureDirectory(configuration.modelsDirectory)
        try Self.ensureDirectory(configuration.cacheDirectory)
        try Self.ensureDirectory(transactionsDirectory)
        transactions = try Self.readTransactions(from: transactionsDirectory)
    }

    deinit {
        for task in tasks.values { task.cancel() }
        for transfer in transfers.values { transfer.cancel() }
    }

    /// Resumes transactions interrupted while pending, downloading, or validating.
    public func recoverDownloads() {
        for (id, transaction) in transactions
        where [
            .pending, .downloading, .validating,
        ].contains(transaction.download.status) {
            begin(id: id)
        }
    }

    public func snapshot() throws -> FMLXModelStoreSnapshot {
        try FMLXModelStoreSnapshot(
            installedModels: installedModels(), downloads: downloads(), cache: cacheInfo())
    }

    public func installedModels() throws -> [FMLXInstalledModel] {
        let manager = FileManager.default
        let owners = try manager.contentsOfDirectory(
            at: configuration.modelsDirectory,
            includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        var result: [FMLXInstalledModel] = []
        for owner in owners.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            guard try Self.isPlainDirectory(owner) else { continue }
            let models = try manager.contentsOfDirectory(
                at: owner, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            for directory in models.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                guard try Self.isPlainDirectory(directory),
                    let model = try? inspectModel(
                        directory: directory, owner: owner.lastPathComponent)
                else { continue }
                result.append(model)
            }
        }
        return result.sorted {
            $0.repositoryID.localizedStandardCompare($1.repositoryID) == .orderedAscending
        }
    }

    public func downloads() -> [FMLXModelDownload] {
        transactions.values.map(\.download).sorted { $0.updatedAt > $1.updatedAt }
    }

    public func activeDownloads() -> [FMLXModelDownload] {
        downloads().filter { [.pending, .downloading, .validating].contains($0.status) }
    }

    public func search(query: String) async throws -> [FMLXModelSearchResult] {
        try await hub.search(query)
    }

    @discardableResult
    public func download(repositoryID: String, revision: String = "main") async throws
        -> FMLXModelDownload
    {
        if let existing = transactions.values.first(where: {
            $0.download.repositoryID == repositoryID
                && [.pending, .downloading, .validating].contains($0.download.status)
        }) {
            return existing.download
        }
        if let existing = transactions.values
            .filter({
                $0.download.repositoryID == repositoryID
                    && [.failed, .cancelled].contains($0.download.status)
            })
            .max(by: { $0.download.updatedAt < $1.download.updatedAt })
        {
            return try retryDownload(id: existing.download.id)
        }
        if try installedModels().contains(where: { $0.repositoryID == repositoryID }) {
            throw FMLXModelManagementError.modelAlreadyInstalled(repositoryID)
        }
        let snapshot = try await hub.repository(repositoryID, revision: revision)
        let total = try snapshot.files.reduce(Int64(0)) { partial, file in
            let sum = partial.addingReportingOverflow(file.sizeBytes)
            guard !sum.overflow else { throw FMLXModelManagementError.invalidRepositoryResponse }
            return sum.partialValue
        }
        let id = UUID()
        let transaction = Transaction(
            download: FMLXModelDownload(
                id: id, repositoryID: repositoryID, revision: snapshot.revision,
                status: .pending, downloadedBytes: 0, totalBytes: total,
                canCancel: true),
            snapshot: snapshot)
        transactions[id] = transaction
        try persist(transaction)
        persistedProgress[id] = 0
        begin(id: id)
        return transaction.download
    }

    public func retryDownload(id: UUID) throws -> FMLXModelDownload {
        guard var transaction = transactions[id] else {
            throw FMLXModelManagementError.downloadNotFound(id.uuidString)
        }
        guard [.failed, .cancelled].contains(transaction.download.status) else {
            return transaction.download
        }
        transaction.download = replacing(
            transaction.download, status: .pending, errorMessage: nil,
            canCancel: true, canRetry: false)
        transactions[id] = transaction
        try persist(transaction)
        persistedProgress[id] = transaction.download.downloadedBytes
        begin(id: id)
        return transaction.download
    }

    public func cancelDownload(id: UUID) throws -> FMLXModelDownload {
        guard var transaction = transactions[id] else {
            throw FMLXModelManagementError.downloadNotFound(id.uuidString)
        }
        guard [.pending, .downloading, .validating].contains(transaction.download.status) else {
            return transaction.download
        }
        transfers.removeValue(forKey: id)?.cancel()
        tasks.removeValue(forKey: id)?.cancel()
        transaction.download = replacing(
            transaction.download, status: .cancelled, canCancel: false, canRetry: true)
        transactions[id] = transaction
        try persist(transaction)
        persistedProgress[id] = transaction.download.downloadedBytes
        return transaction.download
    }

    public func deleteModel(repositoryID: String) throws {
        let isSimpleID =
            !repositoryID.isEmpty && repositoryID != "." && repositoryID != ".."
            && !repositoryID.contains("/") && !repositoryID.contains("\\")
        guard HuggingFaceModelHub.isValidRepositoryID(repositoryID) || isSimpleID else {
            throw FMLXModelManagementError.invalidRepositoryID
        }
        guard
            let model = try installedModels().first(where: {
                $0.repositoryID == repositoryID || $0.id == repositoryID
            })
        else { throw FMLXModelManagementError.modelNotInstalled(repositoryID) }
        try requireDescendant(model.directory, of: configuration.modelsDirectory)
        try FileManager.default.removeItem(at: model.directory)
        let owner = model.directory.deletingLastPathComponent()
        if (try? FileManager.default.contentsOfDirectory(atPath: owner.path).isEmpty) == true {
            try? FileManager.default.removeItem(at: owner)
        }
    }

    public func cacheInfo() throws -> FMLXModelCacheInfo {
        FMLXModelCacheInfo(
            sizeBytes: try Self.directorySize(configuration.cacheDirectory),
            capacityBytes: configuration.cacheCapacityBytes)
    }

    public func clearCache() throws {
        try requireExactDirectory(configuration.cacheDirectory)
        for item in try FileManager.default.contentsOfDirectory(
            at: configuration.cacheDirectory, includingPropertiesForKeys: nil)
        {
            try FileManager.default.removeItem(at: item)
        }
    }

    public func shutdown() {
        let runningTasks = tasks.values
        tasks.removeAll()
        let runningTransfers = transfers.values
        transfers.removeAll()
        for transfer in runningTransfers { transfer.cancel() }
        for task in runningTasks { task.cancel() }
    }

    private func begin(id: UUID) {
        guard tasks[id] == nil, transactions[id] != nil else { return }
        tasks[id] = Task { [weak self] in await self?.performDownload(id: id) }
    }

    private func performDownload(id: UUID) async {
        do {
            guard var transaction = transactions[id] else { return }
            let destination = try destinationDirectory(for: transaction.snapshot.repositoryID)
            if FileManager.default.fileExists(atPath: destination.path) {
                _ = try inspectModel(
                    directory: destination,
                    owner: transaction.snapshot.repositoryID.split(separator: "/")[0].description)
                transaction.download = replacing(
                    transaction.download, status: .completed,
                    downloadedBytes: transaction.download.totalBytes,
                    canCancel: false, canRetry: false)
                transactions[id] = transaction
                try persist(transaction)
                tasks[id] = nil
                return
            }
            let staging = transactionDirectory(id).appendingPathComponent("staging/model")
            try Self.ensureDirectory(staging)
            for file in transaction.snapshot.files {
                try Task.checkCancellation()
                let output = staging.appendingPathComponent(file.path)
                try requireDescendant(output, of: staging)
                try Self.ensureDirectory(output.deletingLastPathComponent())
                var existing = Self.fileSize(output)
                if file.sizeBytes > 0 && existing > file.sizeBytes {
                    try FileManager.default.removeItem(at: output)
                    existing = 0
                }
                if file.sizeBytes == 0 || existing != file.sizeBytes {
                    transaction.download = replacing(
                        transaction.download, status: .downloading,
                        downloadedBytes: try stagedBytes(transaction.snapshot, at: staging),
                        currentFile: file.path, canCancel: true)
                    transactions[id] = transaction
                    try persist(transaction)
                    let transfer = ModelFileTransfer(
                        destination: output, existingBytes: existing,
                        expectedBytes: file.sizeBytes,
                        sessionConfiguration: hub.transferConfiguration
                    ) { [weak self] bytes in
                        Task { await self?.recordProgress(id: id, file: file, bytes: bytes) }
                    }
                    transfers[id] = transfer
                    let received = try await transfer.start(
                        url: hub.fileURL(
                            repositoryID: transaction.snapshot.repositoryID,
                            revision: transaction.snapshot.revision, path: file.path))
                    transfers[id] = nil
                    if file.sizeBytes > 0, received != file.sizeBytes {
                        throw FMLXModelManagementError.sizeMismatch(
                            file: file.path, expected: file.sizeBytes, actual: received)
                    }
                }
                if let checksum = file.sha256,
                    try Self.sha256(output) != checksum.lowercased()
                {
                    throw FMLXModelManagementError.checksumMismatch(file: file.path)
                }
            }
            transaction.download = replacing(
                transaction.download, status: .validating,
                downloadedBytes: transaction.download.totalBytes,
                currentFile: nil, canCancel: false)
            transactions[id] = transaction
            try persist(transaction)
            _ = try inspectModel(
                directory: staging,
                owner: transaction.snapshot.repositoryID.split(separator: "/")[0].description,
                repositoryID: transaction.snapshot.repositoryID,
                revision: transaction.snapshot.revision)
            let metadata = InstalledMetadata(
                repositoryID: transaction.snapshot.repositoryID,
                revision: transaction.snapshot.revision)
            try Self.encoder.encode(metadata).write(
                to: staging.appendingPathComponent(".fmlx-model.json"), options: .atomic)
            try Self.ensureDirectory(destination.deletingLastPathComponent())
            guard !FileManager.default.fileExists(atPath: destination.path) else {
                throw FMLXModelManagementError.modelAlreadyInstalled(
                    transaction.snapshot.repositoryID)
            }
            try FileManager.default.moveItem(at: staging, to: destination)
            transaction.download = replacing(
                transaction.download, status: .completed,
                downloadedBytes: transaction.download.totalBytes,
                canCancel: false, canRetry: false)
            transactions[id] = transaction
            try persist(transaction)
        } catch is CancellationError {
            if var transaction = transactions[id], transaction.download.status != .cancelled {
                transaction.download = replacing(
                    transaction.download, status: .cancelled,
                    canCancel: false, canRetry: true)
                transactions[id] = transaction
                try? persist(transaction)
            }
        } catch {
            if var transaction = transactions[id] {
                transaction.download = replacing(
                    transaction.download, status: .failed,
                    errorMessage: error.localizedDescription,
                    canCancel: false, canRetry: true)
                transactions[id] = transaction
                try? persist(transaction)
            }
        }
        transfers[id] = nil
        tasks[id] = nil
    }

    private func recordProgress(id: UUID, file: FMLXRepositoryFile, bytes: Int64) {
        guard var transaction = transactions[id], transaction.download.status == .downloading else {
            return
        }
        let completed = transaction.snapshot.files.prefix { $0.path != file.path }
            .reduce(Int64(0)) { $0 + $1.sizeBytes }
        transaction.download = replacing(
            transaction.download, status: .downloading,
            downloadedBytes: min(completed + bytes, transaction.download.totalBytes),
            currentFile: file.path, canCancel: true)
        transactions[id] = transaction
        let last = persistedProgress[id] ?? 0
        if transaction.download.downloadedBytes == transaction.download.totalBytes
            || transaction.download.downloadedBytes - last >= 1_024 * 1_024
        {
            if (try? persist(transaction)) != nil {
                persistedProgress[id] = transaction.download.downloadedBytes
            }
        }
    }

    private func inspectModel(
        directory: URL, owner: String, repositoryID: String? = nil, revision: String? = nil
    ) throws -> FMLXInstalledModel {
        let configURL = directory.appendingPathComponent("config.json")
        guard
            let object = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL))
                as? [String: Any]
        else { throw FMLXModelManagementError.incompleteCheckpoint("config.json is invalid") }
        let text = object["text_config"] as? [String: Any]
        guard let modelType = object["model_type"] as? String else {
            throw FMLXModelManagementError.incompleteCheckpoint("model_type is missing")
        }
        guard HuggingFaceModelHub.supportedModelTypes.contains(modelType) else {
            throw FMLXModelManagementError.unsupportedModelType(modelType)
        }
        guard
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("tokenizer.json").path)
        else { throw FMLXModelManagementError.incompleteCheckpoint("tokenizer.json is missing") }
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            guard
                let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL))
                    as? [String: Any], let map = index["weight_map"] as? [String: String],
                !map.isEmpty
            else { throw FMLXModelManagementError.incompleteCheckpoint("weight index is invalid") }
            for file in Set(map.values) {
                guard HuggingFaceModelHub.isSafeRelativePath(file),
                    Self.fileSize(directory.appendingPathComponent(file)) > 0
                else { throw FMLXModelManagementError.incompleteCheckpoint("\(file) is missing") }
            }
        } else {
            let weights = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey]
            )
            .filter { $0.pathExtension == "safetensors" && Self.fileSize($0) > 0 }
            guard !weights.isEmpty else {
                throw FMLXModelManagementError.incompleteCheckpoint("model weights are missing")
            }
        }
        let stored = try? JSONDecoder().decode(
            InstalledMetadata.self,
            from: Data(contentsOf: directory.appendingPathComponent(".fmlx-model.json")))
        let resolvedRepositoryID =
            repositoryID ?? stored?.repositoryID
            ?? "\(owner)/\(directory.lastPathComponent)"
        let resolvedRevision: String
        if let revision {
            resolvedRevision = revision
        } else if let storedRevision = stored?.revision {
            resolvedRevision = storedRevision
        } else {
            resolvedRevision = try Self.localRevision(directory)
        }
        return FMLXInstalledModel(
            id: directory.lastPathComponent, repositoryID: resolvedRepositoryID,
            displayName: directory.lastPathComponent, directory: directory,
            sizeBytes: try Self.directorySize(directory), modelType: modelType,
            revision: resolvedRevision,
            supportsEmbeddedMTP: ((text?["mtp_num_hidden_layers"] as? Int) ?? 0) > 0)
    }

    private func stagedBytes(_ snapshot: FMLXRepositorySnapshot, at directory: URL) throws -> Int64
    {
        snapshot.files.reduce(0) { partial, file in
            min(
                Int64.max,
                partial
                    + min(
                        Self.fileSize(directory.appendingPathComponent(file.path)), file.sizeBytes))
        }
    }

    private func replacing(
        _ value: FMLXModelDownload, status: FMLXModelDownloadStatus,
        downloadedBytes: Int64? = nil, currentFile: String? = nil,
        errorMessage: String? = nil, canCancel: Bool, canRetry: Bool = false
    ) -> FMLXModelDownload {
        FMLXModelDownload(
            id: value.id, repositoryID: value.repositoryID, revision: value.revision,
            status: status, downloadedBytes: downloadedBytes ?? value.downloadedBytes,
            totalBytes: value.totalBytes, currentFile: currentFile,
            errorMessage: errorMessage, canCancel: canCancel, canRetry: canRetry)
    }

    private func persist(_ transaction: Transaction) throws {
        try Self.ensureDirectory(transactionDirectory(transaction.download.id))
        try Self.encoder.encode(transaction).write(
            to: transactionDirectory(transaction.download.id).appendingPathComponent("state.json"),
            options: .atomic)
    }

    private func transactionDirectory(_ id: UUID) -> URL {
        transactionsDirectory.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    private func destinationDirectory(for repositoryID: String) throws -> URL {
        guard HuggingFaceModelHub.isValidRepositoryID(repositoryID) else {
            throw FMLXModelManagementError.invalidRepositoryID
        }
        let parts = repositoryID.split(separator: "/").map(String.init)
        return configuration.modelsDirectory.appendingPathComponent(parts[0], isDirectory: true)
            .appendingPathComponent(parts[1], isDirectory: true)
    }

    private func requireDescendant(_ url: URL, of root: URL) throws {
        let rootPath = root.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath + "/") else {
            throw FMLXModelManagementError.unsafePath(path)
        }
    }

    private func requireExactDirectory(_ url: URL) throws {
        guard try Self.isPlainDirectory(url) else {
            throw FMLXModelManagementError.unsafePath(url.path)
        }
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static func readTransactions(from root: URL) throws -> [UUID: Transaction] {
        var result: [UUID: Transaction] = [:]
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        for directory in try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        {
            guard try isPlainDirectory(directory),
                let id = UUID(uuidString: directory.lastPathComponent),
                let transaction = try? decoder.decode(
                    Transaction.self,
                    from: Data(contentsOf: directory.appendingPathComponent("state.json")))
            else { continue }
            result[id] = transaction
        }
        return result
    }

    private static func ensureDirectory(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private static func isPlainDirectory(_ url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        return values.isDirectory == true && values.isSymbolicLink != true
    }

    private static func fileSize(_ url: URL) -> Int64 {
        let value = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard value?.isRegularFile == true else { return 0 }
        return Int64(value?.fileSize ?? 0)
    }

    private static func directorySize(_ root: URL) throws -> Int64 {
        guard
            let enumerator = FileManager.default.enumerator(
                at: root,
                includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey],
                options: [.skipsHiddenFiles])
        else { return 0 }
        var result: Int64 = 0
        for case let url as URL in enumerator {
            let values = try url.resourceValues(
                forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
            if values.isSymbolicLink == true {
                enumerator.skipDescendants()
                continue
            }
            guard values.isRegularFile == true else { continue }
            let sum = result.addingReportingOverflow(Int64(values.fileSize ?? 0))
            result = sum.overflow ? Int64.max : sum.partialValue
        }
        return result
    }

    private static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 4 * 1_024 * 1_024), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func localRevision(_ directory: URL) throws -> String {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]
        )
        .filter {
            $0.lastPathComponent == "config.json"
                || $0.lastPathComponent == "model.safetensors.index.json"
                || $0.pathExtension == "safetensors"
        }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var hasher = SHA256()
        for file in files {
            let values = try file.resourceValues(
                forKeys: [.fileSizeKey, .contentModificationDateKey])
            hasher.update(
                data: Data(
                    "\(file.lastPathComponent):\(values.fileSize ?? 0):\(values.contentModificationDate?.timeIntervalSince1970 ?? 0)\n"
                        .utf8))
        }
        return "local-" + hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
