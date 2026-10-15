// Copyright © 2026 Faux Fox, Inc.

import Foundation
import Tokenizers

/// The tokenizer operations the decision models use. Errors propagate; nothing is guessed.
protocol DecisionTokenizer: Sendable {
    /// IDs without special tokens, as Hugging Face `tokenizer(text, add_special_tokens=False)`.
    func encode(_ text: String) throws -> [Int]
    /// Subword strings, as Hugging Face `tokenizer.tokenize(text)`.
    func tokenize(_ text: String) throws -> [String]
    func id(of token: String) -> Int?
    var unknownToken: String? { get }
}

struct NativeDecisionTokenizer: DecisionTokenizer {
    let tokenizer: any Tokenizers.Tokenizer

    static func load(from directory: URL) async throws -> Self {
        guard directory.isFileURL else {
            throw DecisionError.invalidConfiguration("the tokenizer needs a local directory")
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DecisionError.invalidConfiguration("tokenizer.json must be a JSON object")
        }
        for key in ["truncation", "padding"] where object[key] != nil && !(object[key] is NSNull) {
            throw DecisionError.invalidConfiguration(
                "tokenizer.json enables \(key); decision inputs must not change silently")
        }
        return Self(tokenizer: try await AutoTokenizer.from(directory: directory))
    }

    func encode(_ text: String) throws -> [Int] {
        try tokenizer.encode(text: text, textPair: nil, addSpecialTokens: false)
    }

    func tokenize(_ text: String) throws -> [String] {
        try tokenizer.tokenize(text: text)
    }

    func id(of token: String) -> Int? { tokenizer.convertTokenToId(token) }

    var unknownToken: String? { tokenizer.unknownToken }
}
