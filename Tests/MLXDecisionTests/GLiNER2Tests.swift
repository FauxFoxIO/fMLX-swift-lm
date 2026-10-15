// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXDecision

func gliner2Tasks(_ json: [[String: Any]]) -> [GLiNER2Task] {
    json.map { task in
        GLiNER2Task(
            name: task["task"] as! String, labels: task["labels"] as! [String],
            prompt: task["prompt"] as? String,
            descriptions: task["label_descriptions"] as? [String: String] ?? [:])
    }
}

@Suite(.serialized)
struct GLiNER2Tests {
    @Test("Schema tokens, word splits, IDs, and marker positions match the reference processor")
    func schemas() throws {
        let fixture = try Fixtures.json("gliner2-schemas.json") as! [String: Any]
        for entry in fixture["cases"] as! [[String: Any]] {
            let text = entry["text"] as! String
            let tasks = gliner2Tasks(entry["tasks"] as! [[String: Any]])
            #expect(GLiNER2Schema.normalize(text) == entry["normalized"] as! String)
            let encoding = try GLiNER2Schema.encode(
                text: text, tasks: tasks, tokenizer: FakeGLiNERTokenizer(), maximumTokens: 100_000)
            #expect(encoding.schemaTokens == entry["schema_tokens"] as! [[String]])
            #expect(encoding.textTokens == entry["text_tokens"] as! [String], "\(text)")
            #expect(encoding.ids == entry["input_ids"] as! [Int])
            #expect(encoding.textWordPositions == entry["text_word_first_positions"] as! [Int])
            #expect(encoding.markerPositions == entry["schema_special_positions"] as! [[Int]])
        }
    }

    @Test("Words past the token limit are dropped; an oversized schema throws")
    func limits() throws {
        let task = GLiNER2Task(name: "intent", labels: ["a", "b"])
        let full = try GLiNER2Schema.encode(
            text: "one two three four five", tasks: [task], tokenizer: FakeGLiNERTokenizer(),
            maximumTokens: 1000)
        let cut = try GLiNER2Schema.encode(
            text: "one two three four five", tasks: [task], tokenizer: FakeGLiNERTokenizer(),
            maximumTokens: full.ids.count - 3)
        #expect(cut.textTokens == ["one", "two", "three"])
        #expect(cut.ids.count <= full.ids.count - 3)
        #expect(throws: DecisionError.self) {
            try GLiNER2Schema.encode(
                text: "x", tasks: [task], tokenizer: FakeGLiNERTokenizer(), maximumTokens: 5)
        }
    }

    @Test("Reserved GLiNER2 tokens in task strings are rejected, not interpreted")
    func reservedTokens() {
        for task in [
            GLiNER2Task(name: "intent", labels: ["a", "[SEP_STRUCT]"]),
            GLiNER2Task(name: "[L]", labels: ["a", "b"]),
            GLiNER2Task(name: "intent", labels: ["a", "b"], prompt: "pick [P] one"),
            GLiNER2Task(name: "intent", labels: ["a", "b"], descriptions: ["a": "x [SEP_TEXT]"]),
        ] {
            #expect(throws: DecisionError.self) {
                try GLiNER2Schema.encode(
                    text: "text", tasks: [task], tokenizer: FakeGLiNERTokenizer(),
                    maximumTokens: 1000)
            }
        }
        // Text cannot inject them: the word splitter breaks brackets apart.
        #expect(GLiNER2Schema.words("[SEP_TEXT] x") == ["[", "sep_text", "]", "x"])
    }

    @Test("Decision questions become GLiNER2 tasks")
    func questionTasks() {
        let choice = GLiNER2Task(
            DecisionQuestion(
                id: "team", instructions: "Who handles it?",
                kind: .choice([
                    DecisionOption("billing", description: "Charges"), DecisionOption("tech"),
                ])))
        #expect(choice.labels == ["billing", "tech"])
        #expect(choice.prompt == "Who handles it?")
        #expect(choice.descriptions == ["billing": "Charges"])
        #expect(
            GLiNER2Schema.schemaTokens(choice) == [
                "(", "[P]", "team: Who handles it? [DESCRIPTION] billing: Charges", "(", "[L]",
                "billing", "[L]", "tech", ")", ")",
            ])
        let yesNo = GLiNER2Task(DecisionQuestion(id: "done", instructions: "", kind: .yesNo()))
        #expect(yesNo.labels == ["yes", "no"])
        #expect(yesNo.prompt == nil)
        let score = GLiNER2Task(
            DecisionQuestion(
                id: "urgency", instructions: "How urgent?", kind: .score(levels: ["low", "high"])))
        #expect(score.labels == ["low", "high"])
    }

    @Test("Relative positions use DeBERTa's log buckets")
    func relativePositions() throws {
        let data = try Data(
            contentsOf: Fixtures.directory.appendingPathComponent("tiny-deberta/config.json"))
        let configuration = try JSONDecoder().decode(GLiNER2Configuration.self, from: data).encoder
        #expect(configuration.relativeSpan == 32)
        let positions = configuration.relativePositions(length: 40).asArray(Int32.self)
        #expect(positions[0 * 40 + 1] == -1)
        #expect(positions[1 * 40 + 0] == 1)
        #expect(positions[0 * 40 + 4] == -4)
        // |r| > bucket / 2 grows logarithmically and saturates near the maximum.
        #expect(positions[39 * 40 + 0] > 4 && positions[39 * 40 + 0] < 39)
        #expect(positions[0 * 40 + 39] == -positions[39 * 40 + 0])
    }

    @Test("The tiny DeBERTa-v2 encoder and classifier match Transformers")
    func tinyEncoder() throws {
        let directory = Fixtures.directory.appendingPathComponent("tiny-deberta")
        let configuration = try JSONDecoder().decode(
            GLiNER2Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try configuration.validate()
        let network = GLiNER2Network(configuration)
        let weights = GLiNER2Network.sanitize(
            try DecisionWeights.load([directory.appendingPathComponent("model.safetensors")]))
        #expect(weights["count_pred.0.weight"] == nil)
        try DecisionWeights.apply(
            weights, to: network, groupSize: nil, bits: nil, mode: nil, precision: .checkpoint)
        let reference = try Fixtures.json("tiny-deberta/reference.json") as! [String: Any]
        let sequences = reference["sequences"] as! [[Int]]
        let longest = sequences.map(\.count).max()!
        let ids = MLXArray(
            sequences.flatMap {
                ($0 + Array(repeating: 0, count: longest - $0.count)).map(Int32.init)
            },
            [sequences.count, longest])
        let mask =
            MLXArray(Int32(0) ..< Int32(longest))[.newAxis]
            .< MLXArray(sequences.map { Int32($0.count) })[0..., .newAxis]
        let hidden = network.encoder(ids, mask: mask)
        let summaries = reference["hidden"] as! [[String: Any]]
        let firsts = reference["first_hidden"] as! [[Double]]
        let classifier = reference["classifier"] as! [[Double]]
        for (row, sequence) in sequences.enumerated() {
            let states = hidden[row, ..<sequence.count]
            expectSummary(states, summaries[row], tolerance: 1e-3)
            expectClose(
                hidden[row, 0, ..<8].asArray(Float.self).map(Double.init), firsts[row],
                tolerance: 1e-3)
            var x = states
            for layer in network.classifier { x = layer(x) }
            expectClose(
                x.squeezed(axis: -1).asArray(Float.self).map(Double.init), classifier[row],
                tolerance: 1e-3)
        }
    }

    @Test("Classification batches requests, keeps order, and reads text only")
    func runtime() async throws {
        struct SmallVocabulary: DecisionTokenizer {
            let base = FakeGLiNERTokenizer()
            func encode(_ text: String) throws -> [Int] { try base.encode(text).map { $0 % 300 } }
            func tokenize(_ text: String) throws -> [String] { try base.tokenize(text) }
            func id(of token: String) -> Int? { base.id(of: token).map { $0 % 300 } }
            var unknownToken: String? { nil }
        }
        let directory = Fixtures.directory.appendingPathComponent("tiny-deberta")
        let configuration = try JSONDecoder().decode(
            GLiNER2Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let network = GLiNER2Network(configuration)
        try DecisionWeights.apply(
            GLiNER2Network.sanitize(
                try DecisionWeights.load([directory.appendingPathComponent("model.safetensors")])),
            to: network, groupSize: nil, bits: nil, mode: nil, precision: .checkpoint)
        let runtime = GLiNER2Runtime(network: network, tokenizer: SmallVocabulary())
        let tasks = [
            GLiNER2Task(name: "intent", labels: ["refund", "other", "login"]),
            GLiNER2Task(name: "topics", labels: ["billing", "app"], multiLabel: true),
        ]
        let texts = [
            "Please refund the double charge", "short", String(repeating: "word ", count: 30),
        ]
        let together = try runtime.classify(
            texts.map { ($0, tasks) }, maximumTokens: 2048, tokenBudget: 16384)
        let tight = try runtime.classify(
            texts.map { ($0, tasks) }, maximumTokens: 2048, tokenBudget: 64)
        for (index, text) in texts.enumerated() {
            let alone = try runtime.classify(
                [(text, tasks)], maximumTokens: 2048, tokenBudget: 16384)
            for (a, b) in paired(together[index].0, alone[0].0) {
                expectClose(a.scores, b.scores, tolerance: 1e-4)
            }
            for (a, b) in paired(tight[index].0, alone[0].0) {
                expectClose(a.scores, b.scores, tolerance: 1e-4)
            }
            #expect(abs(together[index].0[0].scores.reduce(0, +) - 1) < 1e-9)
            #expect(together[index].0[1].scores.allSatisfy { $0 > 0 && $0 < 1 })
            #expect(together[index].0[1].multiLabel)
        }
        #expect(throws: DecisionError.self) {
            try runtime.classify(
                [("x", [GLiNER2Task(name: "dup", labels: ["a", "a"])])], maximumTokens: 2048,
                tokenBudget: 16384)
        }
    }
}
