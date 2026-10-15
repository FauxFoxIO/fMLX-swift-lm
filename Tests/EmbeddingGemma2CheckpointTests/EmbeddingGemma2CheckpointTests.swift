// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXVLM
import Testing

/// The native container against Transformers' float32 EmbeddingGemma 2, end to end:
/// tokenizer, prompts, image and audio preprocessing, encoders, and mean pooling.
/// Each test runs only when its variable names a complete local snapshot.
@Suite(.serialized)
struct EmbeddingGemma2CheckpointTests {
    struct Reference {
        let purpose: String
        let content: [[Any]]
        let tokenCount: Int
        let values: [Float]
    }

    static func references() throws -> (cases: [Reference], image: (Int, Int), samples: Int) {
        let url = Bundle.module.url(forResource: "Resources", withExtension: nil)!
            .appendingPathComponent("embeddinggemma-2.json")
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        let cases = (json["cases"] as! [[String: Any]]).map {
            Reference(
                purpose: $0["purpose"] as! String, content: $0["content"] as! [[Any]],
                tokenCount: $0["token_count"] as! Int,
                values: ($0["values"] as! [Double]).map(Float.init))
        }
        let image = json["image"] as! [Int]
        return (cases, (image[0], image[1]), json["audio_samples"] as! Int)
    }

    /// The fixture generator's `synthetic_rgb`.
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

    /// The fixture generator's `synthetic_pcm`, scaled to [-1, 1].
    static func pcm(count: Int) -> [Float] {
        var state: Int64 = 12345
        return (0 ..< count).map { n in
            let t = Double(n) / 16000
            state = (1_103_515_245 * state + 12345) % 2_147_483_648
            let noise = (Double(state) / 2_147_483_648 - 0.5) * 800
            let value =
                6000 * sin(2 * Double.pi * 220 * t)
                + 3000 * sin(2 * Double.pi * (300 + 400 * t) * t) + noise
            return Float(Int16(max(-32768, min(32767, value.rounded(.toNearestOrEven))))) / 32768
        }
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        let dot = zip(a, b).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
        let na = a.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        let nb = b.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        return dot / (na * nb)
    }

    func check(
        variable: String, precision: EmbeddingGemma2Precision, minimumCosine: Double,
        maximumError: Float
    ) async throws {
        let (cases, image, samples) = try Self.references()
        let path = ProcessInfo.processInfo.environment[variable]!
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let container = try await EmbeddingGemma2Loader.loadContainer(
            from: directory,
            artifact: .init(identifier: variable, revision: directory.lastPathComponent),
            precision: precision)
        let rgb = Self.rgb(width: image.0, height: image.1)
        let audio = Self.pcm(count: samples)
        var worstCosine = 1.0
        var worstError: Float = 0
        for reference in cases {
            let content: [EmbeddingGemma2Content] = reference.content.map { item in
                switch item[0] as! String {
                case "text": .text(item[1] as! String)
                case "image": .image(.rgb(bytes: rgb, width: image.0, height: image.1))
                default: .audio(.pcm(samples: audio, sampleRate: 16000))
                }
            }
            let purpose: EmbeddingGemma2Purpose =
                reference.purpose == "query" ? .query() : .document()
            let result = try await container.embed(content, purpose: purpose, dimension: .d768)
            #expect(result.tokenCount == reference.tokenCount, "\(reference.content)")
            let cosine = Self.cosine(result.values, reference.values)
            let error = zip(result.values, reference.values).map { abs($0 - $1) }.max() ?? 0
            worstCosine = min(worstCosine, cosine)
            worstError = max(worstError, error)
            #expect(cosine >= minimumCosine, "\(reference.content): cosine \(cosine)")
            #expect(error <= maximumError, "\(reference.content): error \(error)")

            let truncated = try await container.embed(content, purpose: purpose, dimension: .d256)
            let prefix = Array(reference.values.prefix(256))
            #expect(Self.cosine(truncated.values, prefix) >= minimumCosine)
            let norm = truncated.values.reduce(0) { $0 + $1 * $1 }
            #expect(abs(norm - 1) < 1e-4)
        }
        print(
            "\(variable) \(precision): minimum cosine \(worstCosine), maximum error \(worstError)")
    }

    @Test(
        "BF16 export matches the float32 reference",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_EMBEDDING_GEMMA2_BF16"] != nil))
    func bfloat16() async throws {
        try await check(
            variable: "MLX_EMBEDDING_GEMMA2_BF16", precision: .bfloat16, minimumCosine: 0.999,
            maximumError: 0.01)
    }

    @Test(
        "BF16 export run in float32 matches the float32 reference",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_EMBEDDING_GEMMA2_BF16"] != nil))
    func float32() async throws {
        try await check(
            variable: "MLX_EMBEDDING_GEMMA2_BF16", precision: .float32, minimumCosine: 0.999999,
            maximumError: 1e-5)
    }

    @Test(
        "8-bit export stays close to the float32 reference",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_EMBEDDING_GEMMA2_8BIT"] != nil))
    func eightBit() async throws {
        try await check(
            variable: "MLX_EMBEDDING_GEMMA2_8BIT", precision: .bfloat16, minimumCosine: 0.999,
            maximumError: 0.01)
    }
}
