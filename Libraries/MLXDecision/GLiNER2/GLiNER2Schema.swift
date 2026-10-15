// Copyright © 2026 Faux Fox, Inc.
// Port of the GLiNER2 schema transformer (fastino-ai/GLiNER2 `processor.py` and
// `processing/word_splitter.py`, Apache-2.0) for classification schemas.

import Foundation

/// One classification head over a text: a task name, its labels, and an optional prompt.
public struct GLiNER2Task: Sendable, Hashable {
    public var name: String
    public var labels: [String]
    public var prompt: String?
    /// Label descriptions, in label order. Labels without one are omitted.
    public var descriptions: [String: String]
    /// Multi-label tasks score each label with a sigmoid; single-label tasks share a softmax.
    public var multiLabel: Bool

    public init(
        name: String, labels: [String], prompt: String? = nil, descriptions: [String: String] = [:],
        multiLabel: Bool = false
    ) {
        self.name = name
        self.labels = labels
        self.prompt = prompt
        self.descriptions = descriptions
        self.multiLabel = multiLabel
    }

    /// The task a generic decision question becomes: options, levels, or `yes`/`no` as labels.
    public init(_ question: DecisionQuestion) {
        name = question.id
        prompt = question.instructions.isEmpty ? nil : question.instructions
        multiLabel = false
        var descriptions = [String: String]()
        switch question.kind {
        case .choice(let options):
            labels = options.map(\.name)
            for option in options {
                if let description = option.description, !description.isEmpty {
                    descriptions[option.name] = description
                }
            }
        case .score(let levels):
            labels = levels
        case .yesNo(let whenTrue, let whenFalse):
            labels = ["yes", "no"]
            if let whenTrue, !whenTrue.isEmpty { descriptions["yes"] = whenTrue }
            if let whenFalse, !whenFalse.isEmpty { descriptions["no"] = whenFalse }
        }
        self.descriptions = descriptions
    }
}

struct GLiNER2Encoding: Sendable {
    let ids: [Int]
    let schemaTokens: [[String]]
    let textTokens: [String]
    /// Per task: the `[P]` position, then one `[L]` position per label.
    let markerPositions: [[Int]]
    let textWordPositions: [Int]
}

enum GLiNER2Schema {
    static let special = (
        sepStruct: "[SEP_STRUCT]", sepText: "[SEP_TEXT]", prompt: "[P]", label: "[L]",
        description: "[DESCRIPTION]",
        all: [
            "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]", "[L]", "[EXAMPLE]",
            "[OUTPUT]",
            "[DESCRIPTION]",
        ]
    )

    private static let word = "[\\p{L}\\p{N}\\p{M}_\\u200c\\u200d]"
    private static let splitter = try! NSRegularExpression(
        pattern: "(?:https?://[^\\s]+|www\\.[^\\s]+)"
            + "|[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}"
            + "|@[a-z0-9_]+"
            + "|\(word)+(?:[-_]\(word)+)*"
            + "|\\S",
        options: [.caseInsensitive])

    /// Text ends with sentence punctuation, as collation expects.
    static func normalize(_ text: String) -> String {
        guard !text.isEmpty else { return "." }
        return text.hasSuffix(".") || text.hasSuffix("!") || text.hasSuffix("?") ? text : text + "."
    }

    /// The whitespace word splitter: URLs, emails, handles, words with inner `-`/`_`, else one scalar.
    static func words(_ text: String) -> [String] {
        let range = NSRange(text.startIndex ..< text.endIndex, in: text)
        return splitter.matches(in: text, range: range).compactMap { match in
            Range(match.range, in: text).map { text[$0].lowercased() }
        }
    }

    static func schemaTokens(_ task: GLiNER2Task) -> [String] {
        var prompt = task.name
        if let text = task.prompt, !text.isEmpty { prompt = "\(task.name): \(text)" }
        for label in task.labels {
            if let description = task.descriptions[label] {
                prompt += " \(special.description) \(label): \(description)"
            }
        }
        var tokens = ["(", special.prompt, prompt, "("]
        for label in task.labels { tokens += [special.label, label] }
        return tokens + [")", ")"]
    }

    /// Schemas, `[SEP_TEXT]`, then the text's words, each tokenized on its own.
    /// Words past `maximumTokens` are dropped, keeping the schema whole.
    static func encode(
        text: String, tasks: [GLiNER2Task], tokenizer: some DecisionTokenizer, maximumTokens: Int
    ) throws -> GLiNER2Encoding {
        for task in tasks {
            for value in [task.name, task.prompt ?? ""] + task.labels
                + Array(task.descriptions.values)
            where special.all.contains(where: value.contains) {
                throw DecisionError.invalidInput(
                    "GLiNER2 task \(task.name) uses a reserved token such as [L] or [SEP_STRUCT]")
            }
        }
        let schemas = tasks.map(schemaTokens)

        var cache = [String: [String]]()
        func pieces(_ token: String) throws -> [String] {
            if let cached = cache[token] { return cached }
            let result = try tokenizer.tokenize(token)
            cache[token] = result
            return result
        }

        // Markers sit at fixed slots of each schema: `[P]` at 1, then every `[L]` from 4.
        var subwords = [String]()
        var markerPositions = [[Int]](repeating: [], count: schemas.count)
        for (schemaIndex, schema) in schemas.enumerated() {
            if schemaIndex > 0 { subwords += try pieces(special.sepStruct) }
            for (index, token) in schema.enumerated() {
                if index == 1 || (index >= 4 && index < schema.count - 2 && index % 2 == 0) {
                    markerPositions[schemaIndex].append(subwords.count)
                }
                subwords += try pieces(token)
            }
        }
        subwords += try pieces(special.sepText)
        guard subwords.count < maximumTokens else {
            throw DecisionError.contextOverflow(
                "the GLiNER2 schema takes \(subwords.count) of \(maximumTokens) tokens")
        }
        var textTokens = [String]()
        var textWordPositions = [Int]()
        for word in words(normalize(text)) {
            let parts = try pieces(word)
            if subwords.count + parts.count > maximumTokens { break }
            textTokens.append(word)
            textWordPositions.append(subwords.count)
            subwords += parts
        }
        let unknown = tokenizer.unknownToken.flatMap { tokenizer.id(of: $0) }
        let ids = try subwords.map { piece -> Int in
            guard let id = tokenizer.id(of: piece) ?? unknown else {
                throw DecisionError.invalidConfiguration("the tokenizer has no ID for \(piece)")
            }
            return id
        }
        return GLiNER2Encoding(
            ids: ids, schemaTokens: schemas, textTokens: textTokens,
            markerPositions: markerPositions,
            textWordPositions: textWordPositions)
    }
}
