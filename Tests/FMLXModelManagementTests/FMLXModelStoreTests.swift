// Copyright © 2026 Faux Fox.

import Foundation
import Testing

@testable import FMLXModelManagement

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

@Test func discoversExistingModelWithoutInstallationMetadata() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let model = fixture.models.appendingPathComponent("mlx-community/Qwen", isDirectory: true)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try Data(#"{"model_type":"qwen3","vocab_size":8}"#.utf8)
        .write(to: model.appendingPathComponent("config.json"))
    try Data(#"{}"#.utf8).write(to: model.appendingPathComponent("tokenizer.json"))
    try Data(repeating: 1, count: 32)
        .write(to: model.appendingPathComponent("model.safetensors"))

    let store = try fixture.store()
    let installed = try await store.installedModels()

    #expect(installed.count == 1)
    #expect(installed.first?.id == "Qwen")
    #expect(installed.first?.repositoryID == "mlx-community/Qwen")
    #expect(installed.first?.modelType == "qwen3")
}

@Test func reportsEmbeddedMTPFromTextConfiguration() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let model = fixture.models.appendingPathComponent("OsaurusAI/Qwen-MTP", isDirectory: true)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try Data(#"{"model_type":"qwen3_5_moe","text_config":{"mtp_num_hidden_layers":1}}"#.utf8)
        .write(to: model.appendingPathComponent("config.json"))
    try Data(#"{}"#.utf8).write(to: model.appendingPathComponent("tokenizer.json"))
    try Data(repeating: 1, count: 32)
        .write(to: model.appendingPathComponent("model.safetensors"))

    let installed = try await fixture.store().installedModels()

    #expect(installed.first?.supportsEmbeddedMTP == true)
}

@Test func incompleteCheckpointIsNotPublishedInCatalog() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let model = fixture.models.appendingPathComponent("owner/incomplete", isDirectory: true)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try Data(#"{"model_type":"qwen3"}"#.utf8)
        .write(to: model.appendingPathComponent("config.json"))

    #expect(try await fixture.store().installedModels().isEmpty)
}

@Test func cacheInspectionAndClearingStayInsideCacheRoot() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let cacheFile = fixture.cache.appendingPathComponent("prefix/entry.bin")
    try FileManager.default.createDirectory(
        at: cacheFile.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(repeating: 7, count: 128).write(to: cacheFile)
    let store = try fixture.store()

    #expect(try await store.cacheInfo().sizeBytes == 128)
    try await store.clearCache()
    #expect(try await store.cacheInfo().sizeBytes == 0)
    #expect(FileManager.default.fileExists(atPath: fixture.models.path))
}

@Test func repositoryAndFilePathValidationRejectTraversal() throws {
    #expect(HuggingFaceModelHub.isValidRepositoryID("owner/model"))
    #expect(!HuggingFaceModelHub.isValidRepositoryID("owner/model/extra"))
    #expect(!HuggingFaceModelHub.isSafeRelativePath("../secret"))
    #expect(!HuggingFaceModelHub.isSafeRelativePath("nested/../secret"))
    #expect(HuggingFaceModelHub.isSafeRelativePath("optiq/mtp.safetensors"))
}

@Test func fileTransferResumesWithHTTPRange() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let destination = fixture.root.appendingPathComponent("partial.bin")
    try Data("native ".utf8).write(to: destination)
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [RangeProtocol.self]
    let transfer = ModelFileTransfer(
        destination: destination, existingBytes: 7,
        expectedBytes: Int64(RangeProtocol.content.count),
        sessionConfiguration: configuration
    ) { _ in }

    let received = try await transfer.start(url: URL(string: "https://unit.test/model")!)

    #expect(received == Int64(RangeProtocol.content.count))
    #expect(try Data(contentsOf: destination) == RangeProtocol.content)
}

@Test func installedCatalogProbe() async throws {
    guard let root = ProcessInfo.processInfo.environment["FMLX_MODEL_STORE_PROBE_ROOT"] else {
        return
    }
    let models = URL(fileURLWithPath: root, isDirectory: true)
    let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(
        "fmlx-probe-cache-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: scratch) }
    let store = try FMLXModelStore(
        configuration: .init(
            modelsDirectory: models, cacheDirectory: scratch, cacheCapacityBytes: 1_024))

    let installed = try await store.installedModels()

    #expect(installed.contains { $0.repositoryID == "mlx-community/Qwen3.8-27B-4bit" })
}

private struct Fixture {
    let root: URL
    let models: URL
    let cache: URL

    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "fmlx-model-store-\(UUID().uuidString)", isDirectory: true)
        models = root.appendingPathComponent("models", isDirectory: true)
        cache = root.appendingPathComponent("cache", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    func store() throws -> FMLXModelStore {
        try FMLXModelStore(
            configuration: .init(
                modelsDirectory: models, cacheDirectory: cache, cacheCapacityBytes: 1_024))
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}

private final class RangeProtocol: URLProtocol, @unchecked Sendable {
    static let content = Data("native fMLX model bytes".utf8)

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let start =
            request.value(forHTTPHeaderField: "Range")
            .flatMap { $0.dropFirst("bytes=".count).split(separator: "-").first }
            .flatMap { Int($0) } ?? 0
        let body = Self.content.dropFirst(start)
        let response = HTTPURLResponse(
            url: request.url!, statusCode: start > 0 ? 206 : 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Length": String(body.count)])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
