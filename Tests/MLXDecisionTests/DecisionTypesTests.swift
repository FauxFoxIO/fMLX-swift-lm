// Copyright © 2026 Faux Fox, Inc.

import Foundation
import Testing

@testable import MLXDecision

struct DecisionTypesTests {
    @Test("Questions validate option counts and unique outcomes")
    func validation() throws {
        try DecisionQuestion(
            id: "q", instructions: "",
            kind: .choice([DecisionOption("a"), DecisionOption("b")])
        ).validate()
        try DecisionQuestion(id: "q", instructions: "", kind: .yesNo()).validate()
        let invalid: [DecisionQuestionKind] = [
            .choice([DecisionOption("a")]),
            .choice([DecisionOption("a"), DecisionOption("a")]),
            .choice([DecisionOption("a"), DecisionOption("")]),
            .score(levels: ["only"]),
            .score(levels: (0 ..< 11).map(String.init)),
        ]
        for kind in invalid {
            #expect(throws: DecisionError.self) {
                try DecisionQuestion(id: "q", instructions: "", kind: kind).validate()
            }
        }
    }

    @Test("Outcomes follow option, level, and yes-then-no order")
    func outcomes() {
        #expect(
            DecisionQuestion(id: "q", instructions: "", kind: .yesNo()).outcomes == ["yes", "no"])
        #expect(
            DecisionQuestion(id: "q", instructions: "", kind: .score(levels: ["low", "high"]))
                .outcomes
                == ["low", "high"])
    }

    @Test("Distributions report their top outcome")
    func distribution() {
        let distribution = DecisionDistribution(
            questionID: "q", outcomes: ["a", "b", "c"], probabilities: softmax([0.1, 2.0, -1]),
            logits: [0.1, 2.0, -1], temperature: 1)
        #expect(distribution.top == "b")
        #expect(abs(distribution.probabilities.reduce(0, +) - 1) < 1e-12)
        #expect(distribution.confidence == distribution.probability(of: "b"))
        #expect(distribution.probability(of: "z") == nil)
        let stable = softmax([1000, 1000])
        #expect(stable == [0.5, 0.5])
    }

    @Test("PCM inputs are mixed, scaled, and cut to the clip limit")
    func audioInputs() async throws {
        let stereo = try await DecisionMedia.waveform(
            .pcm(samples: [0.5, -0.5, 1, 0], sampleRate: 16000, channels: 2), maximumSeconds: 30)
        #expect(stereo == [0, 0.5])
        let pcm16 = try await DecisionMedia.waveform(
            .pcm16(samples: [16384, -32768], sampleRate: 16000), maximumSeconds: 30)
        #expect(pcm16 == [0.5, -1])
        let long = try await DecisionMedia.waveform(
            .pcm(samples: [Float](repeating: 0.1, count: 40_000), sampleRate: 16000),
            maximumSeconds: 2)
        #expect(long.count == 32_000)
        let resampled = try await DecisionMedia.waveform(
            .pcm(samples: (0 ..< 48_000).map { Float(sin(Double($0) * 0.01)) }, sampleRate: 48_000),
            maximumSeconds: 30)
        #expect(abs(resampled.count - 16_000) <= 32)
        await #expect(throws: DecisionError.self) {
            try await DecisionMedia.waveform(
                .pcm(samples: [0, 1, 2], sampleRate: 16000, channels: 2), maximumSeconds: 30)
        }
        await #expect(throws: DecisionError.self) {
            try await DecisionMedia.waveform(
                .pcm(samples: [.nan], sampleRate: 16000), maximumSeconds: 30)
        }
        let clip = D1OmniMelFrontend.clip([1, 2])
        #expect(clip.count == 8000 && clip[0] == 1 && clip[2] == 0)
    }
}
