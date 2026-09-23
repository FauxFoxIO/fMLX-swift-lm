// Copyright © 2026 Faux Fox.

import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(Security)
@preconcurrency import Security
#endif

public enum HuggingFaceAuthenticationState: Equatable, Sendable {
    case setupRequired
    case signedOut
    case signedIn(username: String)
    case reauthenticationRequired
}

public struct HuggingFaceDeviceLoginChallenge: Hashable, Sendable,
    CustomStringConvertible, CustomDebugStringConvertible
{
    public let id: UUID
    public let verificationURL: URL
    public let userCode: String
    public let expiresAt: Date

    fileprivate init(id: UUID, verificationURL: URL, userCode: String, expiresAt: Date) {
        self.id = id
        self.verificationURL = verificationURL
        self.userCode = userCode
        self.expiresAt = expiresAt
    }

    public var description: String { "<redacted-huggingface-device-login>" }
    public var debugDescription: String { description }
}

public struct HuggingFaceOAuthCredential: Sendable, CustomStringConvertible,
    CustomDebugStringConvertible
{
    public let username: String
    public let expiresAt: Date?
    fileprivate let accessToken: String
    fileprivate let refreshToken: String?

    fileprivate init(
        username: String, accessToken: String, refreshToken: String?, expiresAt: Date?
    ) {
        self.username = username
        self.accessToken = accessToken
        self.refreshToken = refreshToken
        self.expiresAt = expiresAt
    }

    public var description: String { "<redacted-huggingface-credential>" }
    public var debugDescription: String { description }
}

public protocol HuggingFaceOAuthCredentialStore: Sendable {
    func load() async throws -> HuggingFaceOAuthCredential?
    func save(_ credential: HuggingFaceOAuthCredential) async throws
    func delete() async throws
}

public protocol HuggingFaceAccessTokenProvider: Sendable {
    func accessToken() async throws -> String?
    func invalidateAccessToken() async
}

public actor InMemoryHuggingFaceOAuthCredentialStore: HuggingFaceOAuthCredentialStore {
    private var credential: HuggingFaceOAuthCredential?

    public init() {}

    public func load() -> HuggingFaceOAuthCredential? { credential }

    public func save(_ credential: HuggingFaceOAuthCredential) {
        self.credential = credential
    }

    public func delete() {
        credential = nil
    }
}

#if canImport(Security)
public actor KeychainHuggingFaceOAuthCredentialStore: HuggingFaceOAuthCredentialStore {
    private struct StoredCredential: Codable {
        let username: String
        let accessToken: String
        let refreshToken: String?
        let expiresAt: Date?
    }

    private let service: String
    private let account = "active"

    public init(service: String) throws {
        guard !service.isEmpty, !service.contains("\n"), !service.contains("\r") else {
            throw FMLXModelManagementError.credentialStoreUnavailable
        }
        self.service = service
    }

    public func load() throws -> HuggingFaceOAuthCredential? {
        var result: CFTypeRef?
        let status = SecItemCopyMatching(readQuery() as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = result as? Data,
            let stored = try? JSONDecoder().decode(StoredCredential.self, from: data)
        else {
            throw FMLXModelManagementError.credentialStoreUnavailable
        }
        return HuggingFaceOAuthCredential(
            username: stored.username,
            accessToken: stored.accessToken,
            refreshToken: stored.refreshToken,
            expiresAt: stored.expiresAt
        )
    }

    public func save(_ credential: HuggingFaceOAuthCredential) throws {
        let stored = StoredCredential(
            username: credential.username,
            accessToken: credential.accessToken,
            refreshToken: credential.refreshToken,
            expiresAt: credential.expiresAt
        )
        guard let data = try? JSONEncoder().encode(stored) else {
            throw FMLXModelManagementError.credentialStoreUnavailable
        }
        let query = baseQuery()
        let attributes = [kSecValueData as String: data]
        let update = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if update == errSecSuccess { return }
        guard update == errSecItemNotFound else {
            throw FMLXModelManagementError.credentialStoreUnavailable
        }
        var insertion = query
        insertion[kSecValueData as String] = data
        guard SecItemAdd(insertion as CFDictionary, nil) == errSecSuccess else {
            throw FMLXModelManagementError.credentialStoreUnavailable
        }
    }

    public func delete() throws {
        let status = SecItemDelete(baseQuery() as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw FMLXModelManagementError.credentialStoreUnavailable
        }
    }

    private func baseQuery() -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
            kSecUseDataProtectionKeychain as String: true,
        ]
    }

    private func readQuery() -> [String: Any] {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        query[kSecUseAuthenticationUI as String] = kSecUseAuthenticationUISkip
        return query
    }
}
#endif

/// Implements Hugging Face's public OAuth device-code flow for native apps.
public actor HuggingFaceDeviceOAuthClient: HuggingFaceAccessTokenProvider {
    private static let issuer = URL(string: "https://huggingface.co")!
    private static let issuerHost = "huggingface.co"
    private struct PendingLogin: Sendable {
        let deviceCode: String
        let challenge: HuggingFaceDeviceLoginChallenge
        var intervalSeconds: Int
    }

    private struct DeviceResponse: Decodable {
        let deviceCode: String?
        let userCode: String?
        let verificationURI: URL?
        let verificationURIComplete: URL?
        let expiresIn: Int?
        let interval: Int?
        let error: String?

        enum CodingKeys: String, CodingKey {
            case deviceCode = "device_code"
            case userCode = "user_code"
            case verificationURI = "verification_uri"
            case verificationURIComplete = "verification_uri_complete"
            case expiresIn = "expires_in"
            case interval
            case error
        }
    }

    private struct TokenResponse: Decodable {
        let accessToken: String?
        let refreshToken: String?
        let expiresIn: Int?
        let error: String?

        enum CodingKeys: String, CodingKey {
            case accessToken = "access_token"
            case refreshToken = "refresh_token"
            case expiresIn = "expires_in"
            case error
        }
    }

    private struct UserInfo: Decodable {
        let preferredUsername: String?
        let name: String?
        let sub: String?

        enum CodingKeys: String, CodingKey {
            case preferredUsername = "preferred_username"
            case name
            case sub
        }
    }

    private let clientID: String?
    private let credentialStore: any HuggingFaceOAuthCredentialStore
    private let session: URLSession
    private var pendingLogins: [UUID: PendingLogin] = [:]

    public init(
        clientID: String?,
        credentialStore: any HuggingFaceOAuthCredentialStore,
        session: URLSession = .shared
    ) {
        self.clientID = Self.normalizedClientID(clientID)
        self.credentialStore = credentialStore
        self.session = session
    }

    public var isConfigured: Bool { clientID != nil }

    public func authenticationState() async throws -> HuggingFaceAuthenticationState {
        guard let stored = try await credentialStore.load() else {
            return isConfigured ? .signedOut : .setupRequired
        }
        do {
            guard let credential = try await usableCredential(from: stored) else {
                return .reauthenticationRequired
            }
            return .signedIn(username: credential.username)
        } catch FMLXModelManagementError.authenticationRequired {
            return .reauthenticationRequired
        }
    }

    public func beginDeviceLogin() async throws -> HuggingFaceDeviceLoginChallenge {
        guard let clientID else { throw FMLXModelManagementError.oauthClientNotConfigured }
        let data = try Self.formData([
            ("client_id", clientID),
            ("scope", "openid profile gated-repos"),
        ])
        var request = URLRequest(url: endpoint("oauth/device"))
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = data
        let (body, response) = try await session.data(for: request)
        let result = try Self.decode(DeviceResponse.self, from: body, response: response)
        guard let deviceCode = result.deviceCode, !deviceCode.isEmpty,
            let userCode = result.userCode, !userCode.isEmpty,
            let verificationURL = result.verificationURIComplete ?? result.verificationURI,
            Self.isSafeVerificationURL(verificationURL),
            let expiresIn = result.expiresIn, expiresIn > 0
        else {
            if result.error != nil { throw Self.oauthError(result.error) }
            throw FMLXModelManagementError.invalidRepositoryResponse
        }
        let id = UUID()
        let challenge = HuggingFaceDeviceLoginChallenge(
            id: id,
            verificationURL: verificationURL,
            userCode: userCode,
            expiresAt: Date().addingTimeInterval(TimeInterval(expiresIn))
        )
        pendingLogins[id] = PendingLogin(
            deviceCode: deviceCode,
            challenge: challenge,
            intervalSeconds: max(result.interval ?? 5, 1)
        )
        return challenge
    }

    /// Polls until the user authorizes, denies, cancels, or lets the code expire.
    public func completeDeviceLogin(id: UUID) async throws -> String {
        while true {
            guard var pending = pendingLogins[id] else {
                throw FMLXModelManagementError.deviceLoginExpired
            }
            guard Date() < pending.challenge.expiresAt else {
                pendingLogins[id] = nil
                throw FMLXModelManagementError.deviceLoginExpired
            }
            try await Task.sleep(for: .seconds(pending.intervalSeconds))
            try Task.checkCancellation()

            guard let clientID else {
                pendingLogins[id] = nil
                throw FMLXModelManagementError.oauthClientNotConfigured
            }
            let fields = [
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("device_code", pending.deviceCode),
                ("client_id", clientID),
            ]
            let result: TokenResponse
            do {
                result = try await tokenRequest(fields: fields, devicePoll: true)
            } catch let error as OAuthPollingError {
                switch error {
                case .authorizationPending:
                    continue
                case .slowDown:
                    pending.intervalSeconds += 5
                    pendingLogins[id] = pending
                    continue
                case .rejected(let oauthError):
                    pendingLogins[id] = nil
                    throw Self.oauthError(oauthError)
                }
            }
            guard let accessToken = result.accessToken, !accessToken.isEmpty else {
                pendingLogins[id] = nil
                throw Self.oauthError(result.error)
            }
            guard pendingLogins[id] != nil else {
                throw CancellationError()
            }
            let username = try await userName(accessToken: accessToken)
            guard pendingLogins[id] != nil, !Task.isCancelled else {
                throw CancellationError()
            }
            let credential = Self.makeCredential(
                username: username,
                accessToken: accessToken,
                refreshToken: result.refreshToken,
                expiresIn: result.expiresIn
            )
            try await credentialStore.save(credential)
            pendingLogins[id] = nil
            return username
        }
    }

    public func cancelDeviceLogin(id: UUID) {
        pendingLogins[id] = nil
    }

    public func signOut() async throws {
        pendingLogins.removeAll()
        try await credentialStore.delete()
    }

    public func accessToken() async throws -> String? {
        guard let stored = try await credentialStore.load() else { return nil }
        guard let credential = try await usableCredential(from: stored) else { return nil }
        return credential.accessToken
    }

    public func invalidateAccessToken() async {
        guard let credential = try? await credentialStore.load() else { return }
        guard let refreshToken = credential.refreshToken else {
            try? await credentialStore.delete()
            return
        }
        let expired = HuggingFaceOAuthCredential(
            username: credential.username,
            accessToken: credential.accessToken,
            refreshToken: refreshToken,
            expiresAt: .distantPast
        )
        try? await credentialStore.save(expired)
    }

    private func usableCredential(
        from stored: HuggingFaceOAuthCredential
    ) async throws -> HuggingFaceOAuthCredential? {
        guard let expiresAt = stored.expiresAt, expiresAt <= Date() else { return stored }
        guard let refreshToken = stored.refreshToken, let clientID else {
            try? await credentialStore.delete()
            throw FMLXModelManagementError.authenticationRequired
        }
        do {
            let response = try await tokenRequest(fields: [
                ("grant_type", "refresh_token"),
                ("refresh_token", refreshToken),
                ("client_id", clientID),
            ])
            guard let accessToken = response.accessToken, !accessToken.isEmpty else {
                try? await credentialStore.delete()
                throw FMLXModelManagementError.authenticationRequired
            }
            let credential = Self.makeCredential(
                username: stored.username,
                accessToken: accessToken,
                refreshToken: response.refreshToken ?? refreshToken,
                expiresIn: response.expiresIn
            )
            try await credentialStore.save(credential)
            return credential
        } catch FMLXModelManagementError.authenticationRequired {
            try? await credentialStore.delete()
            throw FMLXModelManagementError.authenticationRequired
        } catch {
            throw error
        }
    }

    private func tokenRequest(
        fields: [(String, String)], devicePoll: Bool = false
    ) async throws -> TokenResponse {
        var request = URLRequest(url: endpoint("oauth/token"))
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = try Self.formData(fields)
        let (body, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw FMLXModelManagementError.invalidRepositoryResponse
        }
        let result = try? JSONDecoder().decode(TokenResponse.self, from: body)
        guard (200 ..< 300).contains(httpResponse.statusCode) else {
            if let error = result?.error {
                if devicePoll {
                    if error == "authorization_pending" {
                        throw OAuthPollingError.authorizationPending
                    }
                    if error == "slow_down" { throw OAuthPollingError.slowDown }
                    throw OAuthPollingError.rejected(error)
                }
                throw Self.oauthError(error)
            }
            throw FMLXModelManagementError.downloadFailed(statusCode: httpResponse.statusCode)
        }
        guard let result else { throw FMLXModelManagementError.invalidRepositoryResponse }
        return result
    }

    private func userName(accessToken: String) async throws -> String {
        var request = URLRequest(url: endpoint("oauth/userinfo"))
        request.timeoutInterval = 30
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (body, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
            (200 ..< 300).contains(httpResponse.statusCode),
            let info = try? JSONDecoder().decode(UserInfo.self, from: body),
            let username = [info.preferredUsername, info.name, info.sub]
                .compactMap({ $0?.trimmingCharacters(in: .whitespacesAndNewlines) })
                .first(where: { !$0.isEmpty })
        else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            if status == 401 { throw FMLXModelManagementError.authenticationRequired }
            throw FMLXModelManagementError.invalidRepositoryResponse
        }
        return username
    }

    private func endpoint(_ path: String) -> URL {
        path.split(separator: "/").reduce(baseURL) { url, component in
            url.appendingPathComponent(String(component))
        }
    }

    private var baseURL: URL { Self.issuer }

    private static func formData(_ fields: [(String, String)]) throws -> Data {
        var components = URLComponents()
        components.queryItems = fields.map { URLQueryItem(name: $0.0, value: $0.1) }
        guard let query = components.percentEncodedQuery,
            let data = query.data(using: .utf8)
        else {
            throw FMLXModelManagementError.invalidRepositoryResponse
        }
        return data
    }

    private static func decode<Value: Decodable>(
        _ type: Value.Type, from data: Data, response: URLResponse
    ) throws -> Value {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw FMLXModelManagementError.invalidRepositoryResponse
        }
        if (200 ..< 300).contains(httpResponse.statusCode) {
            guard let value = try? JSONDecoder().decode(type, from: data) else {
                throw FMLXModelManagementError.invalidRepositoryResponse
            }
            return value
        }
        let error = (try? JSONDecoder().decode(DeviceResponse.self, from: data))?.error
        if let error { throw oauthError(error) }
        throw FMLXModelManagementError.downloadFailed(statusCode: httpResponse.statusCode)
    }

    private static func oauthError(_ error: String?) -> FMLXModelManagementError {
        switch error {
        case "invalid_client": .oauthClientInvalid
        case "invalid_scope": .oauthScopeUnavailable
        case "access_denied": .deviceLoginDenied
        case "expired_token": .deviceLoginExpired
        case "invalid_grant": .authenticationRequired
        default: .invalidRepositoryResponse
        }
    }

    private static func makeCredential(
        username: String, accessToken: String, refreshToken: String?, expiresIn: Int?
    ) -> HuggingFaceOAuthCredential {
        let expiration = expiresIn.map { Date().addingTimeInterval(TimeInterval(max($0, 1))) }
        return HuggingFaceOAuthCredential(
            username: username,
            accessToken: accessToken,
            refreshToken: refreshToken,
            expiresAt: expiration
        )
    }

    private static func normalizedClientID(_ value: String?) -> String? {
        guard let value else { return nil }
        let clientID = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clientID.isEmpty, clientID.count <= 2_048,
            clientID.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) })
        else { return nil }
        return clientID
    }

    private static func isSafeVerificationURL(_ url: URL) -> Bool {
        guard url.scheme?.lowercased() == "https",
            url.user == nil, url.password == nil,
            let host = url.host?.lowercased(), host == issuerHost,
            url.port == nil || url.port == 443
        else {
            return false
        }
        return true
    }

    private enum OAuthPollingError: Error {
        case authorizationPending
        case slowDown
        case rejected(String)
    }
}
