// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import Testing

@testable import MLXDecision

/// Published checkpoints against MLX-VLM (d1-omni) and the `gliner2` package (GLiNER2.5-Decide).
/// Each suite runs only when its environment variable names a local snapshot.
enum Checkpoints {
    static func directory(_ variable: String) -> URL? {
        guard let path = ProcessInfo.processInfo.environment[variable], !path.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
}

func checkD1Omni(snapshot: URL, fixture: String, tolerance: Double) async throws {
    let reference = try Fixtures.json(fixture) as! [String: Any]
    let model = try await D1OmniLoader.load(
        from: snapshot,
        artifact: DecisionArtifact(
            identifier: reference["model"] as! String, revision: reference["revision"] as! String))
    #expect(model.weightBytes > 500_000_000)
    let tokenizer = try await NativeDecisionTokenizer.load(from: snapshot)
    let special = try D1OmniSpecialTokens(tokenizer: tokenizer, bos: model.configuration.bosTokenID)

    for entry in reference["text"] as! [[String: Any]] {
        let state = entry["state"] as! String
        let questions = (entry["questions"] as! [[String: Any]]).map(decisionQuestion)
        for (index, question) in questions.enumerated() {
            let (ids, markers) = try D1OmniPrompt.encode(
                tokenizer: tokenizer, special: special, state: state,
                question: D1OmniQuestion(question), maxLength: model.configuration.maxLength,
                mode: .text)
            #expect(ids == (entry["ids"] as! [[Int]])[index])
            #expect(markers == (entry["markers"] as! [[Int]])[index])
        }
        let clock = ContinuousClock()
        let start = clock.now
        let result = try await model.decide(DecisionRequest(state: state, questions: questions))
        print("\(fixture) text decision: \(clock.now - start)")
        #expect(result.inputTokens == entry["input_tokens"] as! Int)
        for (distribution, expected) in zip(
            result.distributions, entry["probabilities"] as! [[Double]])
        {
            expectClose(distribution.probabilities, expected, tolerance: tolerance)
        }
    }
    if let float32 = reference["text_float32"] as? [[[Double]]] {
        // Upcast weights isolate the port from float16 rounding.
        let upcast = try await D1OmniLoader.load(
            from: snapshot, artifact: model.artifact, precision: .float32)
        for (entry, expected) in zip(reference["text"] as! [[String: Any]], float32) {
            let questions = (entry["questions"] as! [[String: Any]]).map(decisionQuestion)
            let result = try await upcast.decide(
                DecisionRequest(state: entry["state"] as? String, questions: questions))
            for (distribution, probabilities) in zip(result.distributions, expected) {
                expectClose(distribution.probabilities, probabilities, tolerance: 5e-4)
            }
        }
    }
    for entry in reference["images"] as! [[String: Any]] {
        let (width, height) = (entry["width"] as! Int, entry["height"] as! Int)
        let questions = (entry["questions"] as! [[String: Any]]).map(decisionQuestion)
        let clock = ContinuousClock()
        let start = clock.now
        let result = try await model.decide(
            DecisionRequest(
                state: nil, questions: questions,
                images: [
                    .rgb(
                        bytes: Fixtures.rgb(width: width, height: height), width: width,
                        height: height)
                ]))
        print("\(fixture) \(width)x\(height) image decision: \(clock.now - start)")
        #expect(result.inputTokens == entry["input_tokens"] as! Int)
        for (distribution, expected) in zip(
            result.distributions, entry["probabilities"] as! [[Double]])
        {
            expectClose(distribution.probabilities, expected, tolerance: tolerance * 2)
        }
    }
    for entry in reference["audio"] as! [[String: Any]] {
        let questions = (entry["questions"] as! [[String: Any]]).map(decisionQuestion)
        let clock = ContinuousClock()
        let start = clock.now
        defer { print("\(fixture) audio decision: \(clock.now - start)") }
        let result = try await model.decide(
            DecisionRequest(
                state: nil, questions: questions,
                audio: .pcm16(
                    samples: Fixtures.pcm(count: entry["samples"] as! Int), sampleRate: 16000)))
        #expect(result.inputTokens == entry["input_tokens"] as! Int)
        for (distribution, expected) in zip(
            result.distributions, entry["probabilities"] as! [[Double]])
        {
            expectClose(distribution.probabilities, expected, tolerance: tolerance * 2)
        }
    }
}

@Suite(.serialized)
struct CheckpointTests {
    @Test(
        "d1-omni-600M MLX fp16 matches MLX-VLM",
        .enabled(if: Checkpoints.directory("MLX_DECISION_D1_OMNI_FP16") != nil))
    func d1OmniFloat16() async throws {
        try await checkD1Omni(
            snapshot: Checkpoints.directory("MLX_DECISION_D1_OMNI_FP16")!,
            // Float16 rounding alone moves probabilities by up to about 0.007 (the export's card).
            fixture: "d1-omni-mlx-fp16.json", tolerance: 1e-2)
    }

    @Test(
        "d1-omni-600M MLX 8-bit matches MLX-VLM",
        .enabled(if: Checkpoints.directory("MLX_DECISION_D1_OMNI_8BIT") != nil))
    func d1Omni8Bit() async throws {
        try await checkD1Omni(
            snapshot: Checkpoints.directory("MLX_DECISION_D1_OMNI_8BIT")!,
            fixture: "d1-omni-mlx-8bit.json", tolerance: 1e-2)
    }

    @Test(
        "GLiNER2.5-Decide matches the gliner2 package",
        .enabled(if: Checkpoints.directory("MLX_DECISION_GLINER2_DECIDE") != nil))
    func gliner2Decide() async throws {
        let snapshot = Checkpoints.directory("MLX_DECISION_GLINER2_DECIDE")!
        let reference = try Fixtures.json("gliner2.5-decide.json") as! [String: Any]
        let model = try await GLiNER2Loader.load(
            from: snapshot,
            artifact: DecisionArtifact(
                identifier: reference["model"] as! String,
                revision: reference["revision"] as! String))
        let tokenizer = try await NativeDecisionTokenizer.load(from: snapshot)
        for entry in reference["cases"] as! [[String: Any]] {
            let text = entry["text"] as! String
            let tasks = gliner2Tasks(entry["tasks"] as! [[String: Any]])
            let encoding = try GLiNER2Schema.encode(
                text: text, tasks: tasks, tokenizer: tokenizer, maximumTokens: 100_000)
            #expect(encoding.ids == entry["input_ids"] as! [Int])
            let clock = ContinuousClock()
            let start = clock.now
            let results = try await model.classify(text, tasks: tasks)
            print("GLiNER2.5-Decide classification: \(clock.now - start)")
            let expected = entry["probabilities"] as! [String: [Double]]
            for result in results {
                expectClose(result.scores, expected[result.task]!, tolerance: 2e-3)
            }
        }
        let question = DecisionQuestion(
            id: "handoff", instructions: "Should a person take over?", kind: .yesNo())
        let decided = try await model.decide(
            DecisionRequest(state: "Stop the bot and get me a person.", questions: [question]))
        #expect(decided.distributions[0].outcomes == ["yes", "no"])
        await #expect(throws: DecisionError.self) {
            try await model.decide(
                DecisionRequest(
                    state: nil, questions: [question],
                    images: [.rgb(bytes: [0, 0, 0], width: 1, height: 1)]))
        }
    }
}
