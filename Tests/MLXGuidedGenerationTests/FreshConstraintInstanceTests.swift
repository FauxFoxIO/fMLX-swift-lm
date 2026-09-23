// Copyright © 2026 Faux Fox.

import Foundation
import Testing

@testable import MLXGuidedGeneration

@Suite("Fresh constraint instances")
struct FreshConstraintInstanceTests {
    @Test("reuses a compiled grammar with independent matchers at the initial state")
    func independentMatchers() throws {
        let vocab = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        let tokenizer = try GrammarTokenizer(
            vocab: vocab, vocabType: .byteFallback, eosTokenId: 255)
        var template: GrammarConstraint? = try GrammarConstraint(
            tokenizer: tokenizer, jsonSchema: #"{"type":"integer"}"#)
        let initialMask = try #require(template).computeMask()
        let first = try #require(template).freshInstance()
        let second = try #require(template).freshInstance()
        template = nil

        #expect(try first.computeMask().mask == initialMask.mask)
        #expect(try second.computeMask().mask == initialMask.mask)

        _ = try first.commitToken(52)
        #expect(try second.computeMask().mask == initialMask.mask)
        let restarted = try first.freshInstance()
        #expect(try restarted.computeMask().mask == initialMask.mask)
    }

    @Test(
        "Compiled grammar reuse timing",
        .enabled(if: ProcessInfo.processInfo.environment["FMLX_BENCHMARK_GRAMMAR_REUSE"] == "1"))
    func compileVersusFreshInstance() throws {
        let vocab = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        let tokenizer = try GrammarTokenizer(
            vocab: vocab, vocabType: .byteFallback, eosTokenId: 255)
        let schema =
            #"{"type":"object","properties":{"name":{"type":"string"},"count":{"type":"integer"}},"required":["name","count"]}"#
        let template = try GrammarConstraint(tokenizer: tokenizer, jsonSchema: schema)
        let trials = 32

        let freshStart = ContinuousClock.now
        for _ in 0 ..< trials {
            _ = try template.freshInstance().computeMask()
        }
        let freshDuration = freshStart.duration(to: .now)

        let compileStart = ContinuousClock.now
        for _ in 0 ..< trials {
            _ = try GrammarConstraint(tokenizer: tokenizer, jsonSchema: schema).computeMask()
        }
        let compileDuration = compileStart.duration(to: .now)

        print("GRAMMAR_REUSE trials=\(trials) fresh=\(freshDuration) compile=\(compileDuration)")
    }
}
