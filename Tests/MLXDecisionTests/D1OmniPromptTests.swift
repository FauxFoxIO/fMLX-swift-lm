// Copyright © 2026 Faux Fox, Inc.

import Foundation
import Testing

@testable import MLXDecision

/// A generator `normalized_question` as a public question.
func decisionQuestion(_ json: [String: Any]) -> DecisionQuestion {
    let id = json["id"] as! String
    let instructions = json["instructions"] as! String
    switch json["kind"] as! String {
    case "choice":
        let choices = (json["choices"] as! [[Any]]).map {
            DecisionOption($0[0] as! String, description: $0[1] as? String)
        }
        return DecisionQuestion(id: id, instructions: instructions, kind: .choice(choices))
    case "score":
        return DecisionQuestion(
            id: id, instructions: instructions, kind: .score(levels: json["levels"] as! [String]))
    default:
        return DecisionQuestion(
            id: id, instructions: instructions,
            kind: .yesNo(
                whenTrue: json["when_true"] as? String, whenFalse: json["when_false"] as? String))
    }
}

let fakeD1Special = D1OmniSpecialTokens(
    bos: 1, state: 3, question: 4, option: 5, optionEnd: 6, decide: 7, marker: 2)

struct D1OmniPromptTests {
    @Test("Prompt IDs and marker positions match the reference encoder")
    func referenceEncodings() throws {
        let fixture = try Fixtures.json("d1-prompts.json") as! [String: Any]
        let cases = fixture["cases"] as! [[String: Any]]
        #expect(cases.count > 100)
        for entry in cases {
            let question = D1OmniQuestion(decisionQuestion(entry["question"] as! [String: Any]))
            let mode: D1OmniMediaMode =
                switch entry["media"] as! String {
                case "audio": .audio
                case "image": .image
                default: .text
                }
            let state = entry["state"] as! String
            let maxLength = entry["max_length"] as! Int
            if let error = entry["error"] as? String {
                #expect(throws: DecisionError.self, "\(error)") {
                    try D1OmniPrompt.encode(
                        tokenizer: FakeD1Tokenizer(), special: fakeD1Special, state: state,
                        question: question, maxLength: maxLength, mode: mode)
                }
                continue
            }
            let (ids, markers) = try D1OmniPrompt.encode(
                tokenizer: FakeD1Tokenizer(), special: fakeD1Special, state: state,
                question: question,
                maxLength: maxLength, mode: mode)
            #expect(ids == entry["ids"] as! [Int], "\(question.kind) \(mode) \(maxLength)")
            #expect(markers == entry["markers"] as! [Int])
            for marker in markers { #expect(ids[marker] == fakeD1Special.marker) }
        }
    }

    @Test("Caller text cannot forge delimiters")
    func escaping() {
        #expect(D1OmniPrompt.escape("a <|mask|> b <|reserved_7|>") == "a <¦mask¦> b <¦reserved_7¦>")
        #expect(D1OmniPrompt.escape("<|not closed") == "<|not closed")
        #expect(D1OmniPrompt.escape("<|with space|>") == "<|with space|>")
    }

    @Test("Options follow how each modality was trained")
    func options() {
        let choice = D1OmniQuestion(
            DecisionQuestion(
                id: "c", instructions: "",
                kind: .choice([DecisionOption("a"), DecisionOption("b", description: "bee")])))
        #expect(D1OmniPrompt.options(choice, mode: .text) == ["a", "b: bee"])
        #expect(D1OmniPrompt.options(choice, mode: .audio) == ["option_000: a", "option_001: bee"])
        let yesNo = D1OmniQuestion(DecisionQuestion(id: "y", instructions: "", kind: .yesNo()))
        #expect(
            D1OmniPrompt.options(yesNo, mode: .text) == [
                "false: no, the statement does not hold", "true: yes, the statement holds",
            ])
        #expect(D1OmniPrompt.options(yesNo, mode: .image) == ["false: no", "true: yes"])
        #expect(D1OmniPrompt.options(yesNo, mode: .audio) == ["false: no", "true: yes"])
        let described = D1OmniQuestion(
            DecisionQuestion(id: "y", instructions: "", kind: .yesNo(whenTrue: "it holds")))
        #expect(
            D1OmniPrompt.options(described, mode: .image) == [
                "false: no, the statement does not hold", "true: it holds",
            ])
        let score = D1OmniQuestion(
            DecisionQuestion(id: "s", instructions: "", kind: .score(levels: ["low", "high"])))
        #expect(D1OmniPrompt.options(score, mode: .text) == ["level 0: low", "level 1: high"])
    }

    @Test("Temperature keys bucket by option count")
    func temperatureKeys() {
        func key(_ count: Int) -> String {
            D1OmniQuestion(
                DecisionQuestion(
                    id: "q", instructions: "",
                    kind: .choice((0 ..< count).map { DecisionOption("o\($0)") }))
            ).temperatureKey
        }
        #expect(key(2) == "choice:2")
        #expect(key(5) == "choice:3-5")
        #expect(key(6) == "choice:6-10")
        #expect(key(11) == "choice:11+")
        #expect(
            D1OmniQuestion(DecisionQuestion(id: "q", instructions: "", kind: .yesNo()))
                .temperatureKey
                == "noul:2")
    }
}
