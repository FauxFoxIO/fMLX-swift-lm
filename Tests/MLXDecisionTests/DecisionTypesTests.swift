// Copyright © 2026 Faux Fox, Inc.

import Foundation
import Testing

@testable import MLXDecision

#if canImport(AVFoundation)
import AVFoundation
#endif

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

    @Test("JSON states serialize as Python's json.dumps(ensure_ascii=False)")
    func jsonStates() {
        typealias M = DecisionJSON.Member
        let page = DecisionJSON.object([
            M("page", .string("Release notes")), M("selection", .string("Ship the canvas build")),
            M("speaker", .string("user")),
        ])
        #expect(
            page.serialized
                == #"{"page": "Release notes", "selection": "Ship the canvas build", "speaker": "user"}"#
        )
        let mixed = DecisionJSON.object([
            M("n", .integer(1)), M("f", .number(1)), M("e", .number(1e-05)),
            M("big", .number(1e16)),
            M("neg", .number(-0.5)), M("t", .bool(true)), M("x", .null),
            M("list", .array([.integer(1), .string("two"), .array([.number(3.25)])])),
            M("nested", .object([M("a", .object([]))])), M("empty", .array([])),
        ])
        #expect(
            mixed.serialized
                == #"{"n": 1, "f": 1.0, "e": 1e-05, "big": 1e+16, "neg": -0.5, "t": true, "x": null, "list": [1, "two", [3.25]], "nested": {"a": {}}, "empty": []}"#
        )
        let escaped = DecisionJSON.object([
            M(
                "esc",
                .string("quote \" back \\ nl \n tab \t cr \r bell \u{07} uni é 🎉 \u{2028} slash /"))
        ])
        #expect(
            escaped.serialized
                == "{\"esc\": \"quote \\\" back \\\\ nl \\n tab \\t cr \\r bell \\u0007 uni é 🎉 \u{2028} slash /\"}"
        )
        #expect(DecisionJSON.string("plain string").serialized == #""plain string""#)
        #expect(
            DecisionJSON.array([
                .number(1.5e300), .number(0.1), .number(100), .number(123_456_789_012_345_678),
                .number(-0.0),
            ]).serialized == "[1.5e+300, 0.1, 100.0, 1.2345678901234568e+17, -0.0]")
    }

    #if canImport(AVFoundation)
    @Test("Audio files decode to 16 kHz mono and stop at the clip limit")
    func audioFiles() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("decision-\(UUID().uuidString).caf")
        defer { try? FileManager.default.removeItem(at: url) }
        let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 2)!
        let frames = AVAudioFrameCount(44_100 * 3)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for channel in 0 ..< 2 {
            for i in 0 ..< Int(frames) {
                buffer.floatChannelData![channel][i] = Float(sin(Double(i) * 0.05)) * 0.5
            }
        }
        do {
            var settings = format.settings
            settings[AVLinearPCMIsNonInterleaved] = false
            let file = try AVAudioFile(forWriting: url, settings: settings)
            try file.write(from: buffer)
        }
        let full = try await DecisionMedia.waveform(.file(url), maximumSeconds: 30)
        #expect(abs(full.count - 48_000) <= 64)
        #expect(full.map(abs).max()! > 0.4 && full.map(abs).max()! < 0.6)
        let cut = try await DecisionMedia.waveform(.file(url), maximumSeconds: 1)
        #expect(cut.count == 16_000)
    }
    #endif
}
