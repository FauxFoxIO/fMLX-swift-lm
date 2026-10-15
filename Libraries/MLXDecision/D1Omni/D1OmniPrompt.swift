// Copyright © 2026 Faux Fox, Inc.
// Port of the d1-omni reference `prompt.py` (LiquidAI/d1-omni-600M) and MLX-VLM `d1_omni/prompt.py`.

import Foundation

/// How the media in front of the text shaped training. Options and defaults follow it.
enum D1OmniMediaMode: Sendable {
    case text, image, audio
}

/// A question in the reference's Decision Index form.
struct D1OmniQuestion: Sendable, Hashable {
    enum Kind: Int, Sendable {
        case choice = 0
        case score = 1
        case noul = 2

        var name: String {
            switch self {
            case .choice: "choice"
            case .score: "score"
            case .noul: "noul"
            }
        }
    }

    struct Choice: Sendable, Hashable {
        let name: String
        let description: String?
    }

    let kind: Kind
    let instructions: String
    var choices: [Choice] = []
    var levels: [String] = []
    /// `nil` when the question gives no criteria.
    var noulTrue: String? = nil
    var noulFalse: String? = nil
    var hasNoulCriteria = false

    var optionCount: Int {
        switch kind {
        case .choice: choices.count
        case .score: levels.count
        case .noul: 2
        }
    }

    init(_ question: DecisionQuestion) {
        instructions = question.instructions
        switch question.kind {
        case .choice(let options):
            kind = .choice
            choices = options.map { Choice(name: $0.name, description: $0.description) }
        case .score(let levels):
            kind = .score
            self.levels = levels
        case .yesNo(let whenTrue, let whenFalse):
            kind = .noul
            noulTrue = whenTrue
            noulFalse = whenFalse
            hasNoulCriteria = whenTrue != nil || whenFalse != nil
        }
    }

    init(
        kind: Kind, instructions: String, choices: [Choice] = [], levels: [String] = [],
        noulTrue: String? = nil, noulFalse: String? = nil, hasNoulCriteria: Bool = false
    ) {
        self.kind = kind
        self.instructions = instructions
        self.choices = choices
        self.levels = levels
        self.noulTrue = noulTrue
        self.noulFalse = noulFalse
        self.hasNoulCriteria = hasNoulCriteria
    }

    /// The reference temperature key, such as `choice:3-5`.
    var temperatureKey: String {
        let k = optionCount
        let bucket = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"
        return "\(kind.name):\(bucket)"
    }
}

struct D1OmniSpecialTokens: Sendable, Hashable {
    static let names = (
        state: "<|reserved_7|>", question: "<|reserved_8|>", option: "<|reserved_9|>",
        optionEnd: "<|reserved_10|>", decide: "<|reserved_11|>", marker: "<|mask|>"
    )

    let bos: Int
    let state: Int
    let question: Int
    let option: Int
    let optionEnd: Int
    let decide: Int
    let marker: Int

    init(
        bos: Int, state: Int, question: Int, option: Int, optionEnd: Int, decide: Int,
        marker: Int
    ) {
        self.bos = bos
        self.state = state
        self.question = question
        self.option = option
        self.optionEnd = optionEnd
        self.decide = decide
        self.marker = marker
    }

    init(tokenizer: some DecisionTokenizer, bos: Int) throws {
        func id(_ token: String) throws -> Int {
            guard let id = tokenizer.id(of: token), try tokenizer.encode(token) == [id] else {
                throw DecisionError.invalidConfiguration("the tokenizer does not keep \(token)")
            }
            return id
        }
        let names = Self.names
        self.init(
            bos: bos, state: try id(names.state), question: try id(names.question),
            option: try id(names.option), optionEnd: try id(names.optionEnd),
            decide: try id(names.decide), marker: try id(names.marker))
    }
}

enum D1OmniPrompt {
    private static let special = try! NSRegularExpression(pattern: "<\\|([A-Za-z0-9_]+)\\|>")
    static let perOption = 24

    /// `<|name|>` becomes `<¦name¦>`, so caller text cannot emit a delimiter or marker.
    static func escape(_ text: String) -> String {
        let range = NSRange(text.startIndex ..< text.endIndex, in: text)
        return special.stringByReplacingMatches(
            in: text, range: range, withTemplate: "<\u{00A6}$1\u{00A6}>")
    }

    static func options(_ question: D1OmniQuestion, mode: D1OmniMediaMode) -> [String] {
        func nonEmpty(_ value: String?) -> String? {
            guard let value, !value.isEmpty else { return nil }
            return value
        }
        switch question.kind {
        case .choice:
            if mode == .audio {
                return question.choices.enumerated().map { index, choice in
                    "option_\(String(format: "%03d", index)): \(nonEmpty(choice.description) ?? choice.name)"
                }
            }
            return question.choices.map { choice in
                nonEmpty(choice.description).map { "\(choice.name): \($0)" } ?? choice.name
            }
        case .score:
            return question.levels.enumerated().map { "level \($0.offset): \($0.element)" }
        case .noul:
            if mode == .audio { return ["false: no", "true: yes"] }
            var whenFalse = question.noulFalse
            var whenTrue = question.noulTrue
            if !question.hasNoulCriteria, mode == .image {
                whenFalse = "no"
                whenTrue = "yes"
            }
            return [
                "false: " + (nonEmpty(whenFalse) ?? "no, the statement does not hold"),
                "true: " + (nonEmpty(whenTrue) ?? "yes, the statement holds"),
            ]
        }
    }

    /// Token IDs of one question over one state, and the position of each option's marker.
    ///
    /// The option block gets `max(96, min(24k + 32, maxLength / 2))` tokens, shared evenly;
    /// the state is cut on the right to the room that is left.
    static func encode(
        tokenizer: some DecisionTokenizer, special: D1OmniSpecialTokens, state: String,
        question: D1OmniQuestion, maxLength: Int, mode: D1OmniMediaMode
    ) throws -> (ids: [Int], markers: [Int]) {
        func encoded(_ text: String) throws -> [Int] { try tokenizer.encode(escape(text)) }
        let rendered = options(question, mode: mode)
        let budget = max(96, min(rendered.count * perOption + 32, maxLength / 2))
        let per = max(2, (budget - 3 * rendered.count) / rendered.count)
        var body = Array(
            ([special.question] + (try encoded(question.instructions))).prefix(max(16, budget)))
        var markers = [Int]()
        for text in rendered {
            markers.append(body.count + 1)
            body += [special.option, special.marker]
            body += try encoded(" " + text).prefix(per)
            body.append(special.optionEnd)
        }
        body.append(special.decide)
        let room = max(0, maxLength - body.count - 2)
        let stateIDs = [special.state] + (try encoded(state)).prefix(room)
        let ids = Array(([special.bos] + stateIDs + body).prefix(maxLength))
        markers = markers.map { $0 + 1 + stateIDs.count }
        guard let last = markers.last, last < maxLength else {
            throw DecisionError.contextOverflow("the options do not fit in the context")
        }
        return (ids, markers)
    }
}
