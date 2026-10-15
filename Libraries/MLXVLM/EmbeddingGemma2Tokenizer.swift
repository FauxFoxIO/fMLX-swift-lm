// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLXLMCommon
import Tokenizers

/// The native embedding path propagates tokenizer errors instead of suppressing them.
struct EmbeddingGemma2Tokenizer: Sendable {
    private enum Backend: Sendable {
        case native(any Tokenizers.Tokenizer)
        case supplied(any MLXLMCommon.Tokenizer)
    }
    private let backend: Backend

    init(_ tokenizer: any MLXLMCommon.Tokenizer) { backend = .supplied(tokenizer) }
    private init(native: any Tokenizers.Tokenizer) { backend = .native(native) }

    static func load(from directory: URL) async throws -> Self {
        let assets = try checkedAssets(directory)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        guard try checkedAssets(directory) == assets else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "tokenizer assets changed during loading")
        }
        return Self(native: tokenizer)
    }

    static func checkedAssets(_ directory: URL) throws -> [String: Data] {
        guard directory.isFileURL else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "tokenizer requires a local checkpoint directory")
        }
        var result = [String: Data]()
        for name in [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "added_tokens.json",
        ] {
            let url = directory.appendingPathComponent(name)
            if name == "tokenizer.json" || FileManager.default.fileExists(atPath: url.path) {
                let data = try Data(contentsOf: url)
                guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                else {
                    throw EmbeddingGemma2Error.invalidConfiguration("\(name) must be a JSON object")
                }
                if name == "tokenizer.json" {
                    for key in ["truncation", "padding"]
                    where object[key] != nil && !(object[key] is NSNull) {
                        throw EmbeddingGemma2Error.invalidConfiguration(
                            "disable tokenizer \(key); embedding input cannot be silently changed")
                    }
                }
                result[name] = data
            }
        }
        return result
    }

    func encode(text: String, addSpecialTokens: Bool) throws -> [Int] {
        switch backend {
        case .native(let tokenizer):
            try tokenizer.encode(text: text, textPair: nil, addSpecialTokens: addSpecialTokens)
        case .supplied(let tokenizer):
            tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        }
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch backend {
        case .native(let tokenizer): tokenizer.convertTokenToId(token)
        case .supplied(let tokenizer): tokenizer.convertTokenToId(token)
        }
    }

    /// The `<bos>` and `<eos>` IDs that wrap every embedded sequence.
    func sequenceBoundaries() throws -> (begin: Int, end: Int) {
        let (bos, eos): (String?, String?) =
            switch backend {
            case .native(let tokenizer): (tokenizer.bosToken, tokenizer.eosToken)
            case .supplied(let tokenizer): (tokenizer.bosToken, tokenizer.eosToken)
            }
        guard let begin = convertTokenToId(bos ?? "<bos>"),
            let end = convertTokenToId(eos ?? "<eos>")
        else {
            throw EmbeddingGemma2Error.invalidConfiguration("tokenizer has no <bos> or <eos>")
        }
        return (begin, end)
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch backend {
        case .native(let tokenizer): tokenizer.convertIdToToken(id)
        case .supplied(let tokenizer): tokenizer.convertIdToToken(id)
        }
    }
}
