// Copyright © 2024 Apple Inc.

import Foundation

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?

    /// Creates independent decoding state for one generated sequence.
    /// Override to supply a tokenizer-specific incremental implementation.
    func makeStreamingDetokenizer() -> any StreamingDetokenizer

    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

extension Tokenizer {
    public func makeStreamingDetokenizer() -> any StreamingDetokenizer {
        NaiveStreamingDetokenizer(tokenizer: self)
    }

    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokenIds: [Int]) -> String {
        decode(tokenIds: tokenIds, skipSpecialTokens: false)
    }

    public var eosTokenId: Int? {
        guard let eosToken else { return nil }
        return convertTokenToId(eosToken)
    }

    public var unknownTokenId: Int? {
        guard let unknownToken else { return nil }
        return convertTokenToId(unknownToken)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]]
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: tools, additionalContext: nil)
    }
}

public enum TokenizerError: LocalizedError {
    case missingChatTemplate

    public var errorDescription: String? {
        switch self {
        case .missingChatTemplate:
            "This tokenizer does not have a chat template."
        }
    }
}

/// Decodes one generated sequence. Preserve special tokens for downstream protocol parsing.
/// `next()` returns available text, or nil until more tokens arrive. Custom implementations
/// should emit stable text, holding incomplete bytes or context-dependent suffixes as needed.
public protocol StreamingDetokenizer: IteratorProtocol<String> {
    mutating func append(token: Int)

    /// Ends the sequence and returns all remaining text, including any unread output.
    /// Call after the last token and before finishing downstream parsers. Repeated calls
    /// return nil. Create a new decoder before appending another sequence.
    mutating func finish() -> String?
}

extension StreamingDetokenizer {
    /// Preserves the legacy behavior for decoders without a final flush operation.
    public mutating func finish() -> String? { nil }
}

public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: any Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public mutating func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokenIds: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        next(isFinal: false)
    }

    /// Flushes remaining text, including replacement characters for incomplete bytes.
    /// Repeated calls return nil. Create a new detokenizer for the next sequence.
    public mutating func finish() -> String? {
        guard !segmentTokens.isEmpty else { return nil }
        defer {
            segmentTokens.removeAll()
            segment = ""
        }
        guard let text = next(isFinal: true), !text.isEmpty else { return nil }
        return text
    }

    private mutating func next(isFinal: Bool) -> String? {
        let newSegment = tokenizer.decode(tokenIds: segmentTokens)

        // SentencePiece may rewrite trailing whitespace or extend a grapheme.
        // Compare scalars so only newly decoded content is emitted.
        let common = zip(newSegment.unicodeScalars, segment.unicodeScalars)
            .prefix { $0 == $1 }.count
        let new = String(newSegment.unicodeScalars.dropFirst(common))

        // if the new segment ends with REPLACEMENT CHARACTER this means
        // that the token didn't produce a complete unicode character
        if !isFinal, new.last == "\u{fffd}" {
            return nil
        }

        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return new
    }
}
