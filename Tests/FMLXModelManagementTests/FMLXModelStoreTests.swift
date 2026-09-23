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

@Test func modelRepositoryIDsAndPageURLsCanonicalizeSafely() {
    #expect((try? HuggingFaceModelHub.canonicalRepositoryID("owner/model")) == "owner/model")
    #expect(
        (try? HuggingFaceModelHub.canonicalRepositoryID(
            " https://huggingface.co/owner/model/ ")) == "owner/model")
    #expect(
        (try? HuggingFaceModelHub.canonicalRepositoryID(
            "https://huggingface.co/owner/model?tab=files")) == "owner/model")
    #expect(
        (try? HuggingFaceModelHub.canonicalRepositoryID(
            "http://huggingface.co/owner/model")) == nil)
    #expect(
        (try? HuggingFaceModelHub.canonicalRepositoryID(
            "https://huggingface.co.evil.test/owner/model")) == nil)
    #expect(
        (try? HuggingFaceModelHub.canonicalRepositoryID(
            "https://huggingface.co/owner/model/tree/main")) == nil)
    #expect(
        (try? HuggingFaceModelHub.canonicalRepositoryID(
            "https://huggingface.co/owner%2Fother/model")) == nil)
}

@Test func modelHubUsesEphemeralBearerCredentials() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [RepositoryProtocol.self]
    RepositoryProtocol.requests.reset()
    let hub = HuggingFaceModelHub(
        session: URLSession(configuration: configuration),
        credentialProvider: TestTokenProvider()
    )

    let snapshot = try await hub.repository("https://huggingface.co/owner/model")

    #expect(snapshot.repositoryID == "owner/model")
    #expect(
        RepositoryProtocol.requests.snapshot().last?
            .value(forHTTPHeaderField: "Authorization") == "Bearer test-token")
}

@Test func nativeDeviceAuthorizationRequestsGatedRepositoryScope() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [DeviceAuthorizationProtocol.self]
    DeviceAuthorizationProtocol.requests.reset()
    let client = HuggingFaceDeviceOAuthClient(
        clientID: "flow-public-client",
        credentialStore: InMemoryHuggingFaceOAuthCredentialStore(),
        session: URLSession(configuration: configuration)
    )

    let challenge = try await client.beginDeviceLogin()
    let request = try #require(DeviceAuthorizationProtocol.requests.snapshot().last)
    var components = URLComponents()
    components.percentEncodedQuery = String(decoding: try #require(request.httpBody), as: UTF8.self)

    #expect(
        components.queryItems?.first(where: { $0.name == "scope" })?.value
            == "openid profile gated-repos")
    #expect(challenge.userCode == "ABCD-EFGH")
    #expect(!challenge.description.contains("private-device-code"))
    await client.cancelDeviceLogin(id: challenge.id)
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
        repositoryID: "owner/model", authorizationToken: "test-token",
        sessionConfiguration: configuration
    ) { _ in }

    RangeProtocol.requests.reset()
    let received = try await transfer.start(url: URL(string: "https://unit.test/model")!)

    #expect(received == Int64(RangeProtocol.content.count))
    #expect(try Data(contentsOf: destination) == RangeProtocol.content)
    #expect(
        RangeProtocol.requests.snapshot().last?
            .value(forHTTPHeaderField: "Range") == "bytes=7-")
    #expect(
        RangeProtocol.requests.snapshot().last?
            .value(forHTTPHeaderField: "Authorization") == "Bearer test-token")
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
    static let requests = RequestRecorder()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.record(request)
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

private final class RepositoryProtocol: URLProtocol, @unchecked Sendable {
    static let requests = RequestRecorder()
    private static let responseData = Data(
        #"{"id":"owner/model","sha":"commit","siblings":[{"rfilename":"config.json","size":8},{"rfilename":"model.safetensors","size":16}],"config":{"model_type":"qwen3"}}"#
            .utf8
    )

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.record(request)
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200,
            httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.responseData)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private final class DeviceAuthorizationProtocol: URLProtocol, @unchecked Sendable {
    static let requests = RequestRecorder()
    private static let responseData = Data(
        #"{"device_code":"private-device-code","user_code":"ABCD-EFGH","verification_uri":"https://huggingface.co/oauth/authorize","expires_in":600,"interval":5}"#
            .utf8
    )

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.record(request)
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200,
            httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.responseData)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private final class RequestRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var requests: [URLRequest] = []

    func record(_ request: URLRequest) {
        lock.lock()
        requests.append(request)
        lock.unlock()
    }

    func snapshot() -> [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return requests
    }

    func reset() {
        lock.lock()
        requests.removeAll()
        lock.unlock()
    }
}

private actor TestTokenProvider: HuggingFaceAccessTokenProvider {
    func accessToken() -> String? { "test-token" }
    func invalidateAccessToken() {}
}
