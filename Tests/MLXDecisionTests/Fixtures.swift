// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import Testing

@testable import MLXDecision

enum Fixtures {
    static var directory: URL {
        Bundle.module.url(forResource: "Resources", withExtension: nil)!
    }

    static func json(_ name: String) throws -> Any {
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent(name)),
            options: [.fragmentsAllowed])
    }

    /// The generator's `synthetic_rgb`.
    static func rgb(width: Int, height: Int) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                for c in 0 ..< 3 {
                    bytes[(y * width + x) * 3 + c] = UInt8(
                        (x * 3 + y * 5 + c * 70 + (x * y) % 37) % 256)
                }
            }
        }
        return bytes
    }

    /// The generator's `synthetic_pcm`.
    static func pcm(count: Int) -> [Int16] {
        var state: Int64 = 12345
        return (0 ..< count).map { n in
            let t = Double(n) / 16000
            state = (1_103_515_245 * state + 12345) % 2_147_483_648
            let noise = (Double(state) / 2_147_483_648 - 0.5) * 800
            let value =
                6000 * sin(2 * Double.pi * 220 * t)
                + 3000 * sin(2 * Double.pi * (300 + 400 * t) * t) + noise
            return Int16(max(-32768, min(32767, value.rounded(.toNearestOrEven))))
        }
    }

    static func checksum<T: BinaryInteger>(_ values: [T]) -> Int64 {
        let modulus: Int64 = (1 << 61) - 1
        var total: Int64 = 0
        for (i, v) in values.enumerated() {
            total = (total + Int64(v) * Int64(i % 9973 + 1)) % modulus
        }
        return (total + modulus) % modulus
    }
}

/// Compares an array with the generator's `summary`.
func expectSummary(
    _ array: MLXArray, _ summary: [String: Any], tolerance: Double,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let values = array.asType(.float32).asArray(Float.self).map(Double.init)
    let count = summary["count"] as! Int
    #expect(values.count == count, sourceLocation: sourceLocation)
    guard values.count == count else { return }
    let sum = summary["sum"] as! Double
    let absSum = summary["abs_sum"] as! Double
    let scale = max(absSum / Double(max(count, 1)), 1e-6)
    #expect(
        abs(values.reduce(0, +) - sum) <= tolerance * scale * Double(count).squareRoot() * 4,
        "sum \(values.reduce(0, +)) vs \(sum)", sourceLocation: sourceLocation)
    #expect(
        abs(values.reduce(0) { $0 + abs($1) } - absSum) <= tolerance * absSum,
        sourceLocation: sourceLocation)
    let indices = summary["indices"] as! [Int]
    let expected = summary["values"] as! [Double]
    var worst = 0.0
    for (i, value) in zip(indices, expected) {
        worst = max(worst, abs(values[i] - value))
    }
    #expect(
        worst <= tolerance * max(scale, 1), "sample error \(worst)", sourceLocation: sourceLocation)
}

func expectClose(
    _ actual: [Double], _ expected: [Double], tolerance: Double,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(actual.count == expected.count, sourceLocation: sourceLocation)
    let worst = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    #expect(worst <= tolerance, "\(actual) vs \(expected)", sourceLocation: sourceLocation)
}

/// The generator's `FakeD1Tokenizer`.
struct FakeD1Tokenizer: DecisionTokenizer {
    static let special: [String: Int] = [
        "<|startoftext|>": 1, "<|mask|>": 2, "<|reserved_7|>": 3, "<|reserved_8|>": 4,
        "<|reserved_9|>": 5, "<|reserved_10|>": 6, "<|reserved_11|>": 7,
    ]

    func encode(_ text: String) throws -> [Int] {
        return text.unicodeScalars.map { 10 + Int($0.value) % 300 }
    }

    func tokenize(_ text: String) throws -> [String] { text.unicodeScalars.map { String($0) } }
    func id(of token: String) -> Int? { Self.special[token] }
    var unknownToken: String? { nil }
}

/// The generator's `FakeGLiNERTokenizer`.
struct FakeGLiNERTokenizer: DecisionTokenizer {
    static let special = [
        "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]", "[L]", "[EXAMPLE]", "[OUTPUT]",
        "[DESCRIPTION]",
    ]

    func encode(_ text: String) throws -> [Int] { try tokenize(text).map { id(of: $0)! } }

    func tokenize(_ text: String) throws -> [String] {
        if Self.special.contains(text) { return [text] }
        var pieces = [String]()
        let words = text.unicodeScalars.split { $0.properties.isWhitespace }
        for word in words {
            let scalars = Array(word)
            for start in stride(from: 0, to: scalars.count, by: 4) {
                var view = String.UnicodeScalarView()
                view.append(contentsOf: scalars[start ..< min(start + 4, scalars.count)])
                pieces.append(start == 0 ? "\u{2581}" + String(view) : String(view))
            }
        }
        return pieces
    }

    func id(of token: String) -> Int? {
        if let index = Self.special.firstIndex(of: token) { return 1000 + index }
        var total = 0
        for (i, scalar) in token.unicodeScalars.enumerated() {
            total += (i + 1) * Int(scalar.value)
        }
        return 10 + total % 900
    }

    var unknownToken: String? { "[UNK]" }
}
