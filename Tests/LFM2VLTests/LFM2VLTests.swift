// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Reference data written by `tools/decision_models/generate_fixtures.py lfm2vl`.
private enum Fixture {
    static let directory = Bundle.module.url(forResource: "Resources", withExtension: nil)!

    static var reference: [String: Any] {
        let data = try! Data(contentsOf: directory.appendingPathComponent("lfm2-vl.json"))
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    static var tokens: [String: Int] { reference["tokens"] as! [String: Int] }

    /// The generator's `synthetic_rgb`, as uint8 [height, width, 3].
    static func rgb(width: Int, height: Int) -> MLXArray {
        let y = MLXArray(Int32(0) ..< Int32(height)).reshaped(height, 1, 1)
        let x = MLXArray(Int32(0) ..< Int32(width)).reshaped(1, width, 1)
        let c = MLXArray([Int32(0), 1, 2]).reshaped(1, 1, 3)
        return ((x * 3 + y * 5 + c * 70 + (x * y) % 37) % 256).asType(.uint8)
    }

    /// The generator's `checksum`.
    static func checksum(_ values: MLXArray) -> Int64 {
        let flat = values.flattened().asType(.int64)
        let weights = MLXArray(Int64(0) ..< Int64(flat.size)) % 9973 + 1
        return (flat * weights).sum().item(Int64.self) % ((1 << 61) - 1)
    }

    /// Token ids of a markup string such as `<|image_start|><image><image><|image_end|>`.
    static func ids(_ markup: String) -> [Int] {
        let tokens = tokens
        let pattern = try! NSRegularExpression(pattern: "<\\|[a-z0-9_]+\\|>|<image>")
        let range = NSRange(markup.startIndex..., in: markup)
        return pattern.matches(in: markup, range: range).map {
            tokens[String(markup[Range($0.range, in: markup)!])]!
        }
    }

    /// d1-3B's `processor_config.json` with overrides.
    static func configuration(_ overrides: [String: Any] = [:]) throws
        -> LFM2VLProcessorConfiguration
    {
        var settings: [String: Any] = [
            "do_image_splitting": true, "downsample_factor": 2, "encoder_patch_size": 16,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "max_image_tokens": 256,
            "max_num_patches": 1024, "max_pixels_tolerance": 2.0, "max_tiles": 10,
            "min_image_tokens": 64, "min_tiles": 2, "resample": 3,
            "rescale_factor": 0.00392156862745098, "tile_size": 512, "use_thumbnail": true,
            "image_processor_type": "Lfm2VlImageProcessor",
        ]
        settings.merge(overrides) { $1 }
        let json: [String: Any] = [
            "image_processor": settings, "processor_class": "Lfm2VlProcessor",
        ]
        return try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
    }
}

@Suite("LFM2-VL image processing")
struct LFM2VLImageProcessingTests {

    @Test("Tile ratios keep CPython's set order")
    func targetRatios() {
        for entry in Fixture.reference["ratios"] as! [[String: Any]] {
            let low = entry["min_tiles"] as! Int
            let high = entry["max_tiles"] as! Int
            let expected = (entry["ratios"] as! [[Int]]).map { ($0[0], $0[1]) }
            let actual = LFM2VLImageProcessing.targetRatios(minTiles: low, maxTiles: high)
            #expect(
                actual.map { [$0.0, $0.1] } == expected.map { [$0.0, $0.1] },
                "min \(low) max \(high)")
        }
    }

    @Test("Layouts, crops, and markup match Transformers byte for byte")
    func layoutsCropsAndMarkup() throws {
        let cases = Fixture.reference["layouts"] as! [[String: Any]]
        let tokens = Fixture.tokens
        #expect(cases.count >= 40)
        for entry in cases {
            let width = entry["width"] as! Int
            let height = entry["height"] as! Int
            let label = "\(width)x\(height) resample \(entry["resample"]!)"
            var overrides: [String: Any] = ["resample": entry["resample"]!]
            if let split = entry["do_image_splitting"] { overrides["do_image_splitting"] = split }
            let configuration = try Fixture.configuration(overrides)

            let layout = try LFM2VLImageProcessing.layout(
                width: width, height: height, configuration: configuration)
            #expect(layout.rows == entry["rows"] as! Int, "\(label)")
            #expect(layout.columns == entry["columns"] as! Int, "\(label)")
            #expect([layout.height, layout.width] == entry["size"] as! [Int], "\(label)")

            let crops = LFM2VLImageProcessing.crops(
                Fixture.rgb(width: width, height: height), layout: layout,
                configuration: configuration)
            let expected = entry["crops"] as! [[String: Any]]
            #expect(crops.count == expected.count, "\(label)")
            for (crop, reference) in zip(crops, expected) {
                #expect(crop.rows == reference["rows"] as! Int, "\(label)")
                #expect(crop.columns == reference["columns"] as! Int, "\(label)")
                let values = round(crop.patches * 127.5 + 127.5).asType(.int32)
                #expect(
                    Fixture.checksum(values)
                        == (reference["checksum"] as! NSNumber).int64Value, "\(label)")
            }

            let markup = try LFM2VLImageProcessing.imageTokens(
                layout: layout, configuration: configuration, id: { tokens[$0] })
            #expect(markup == Fixture.ids(entry["markup"] as! String), "\(label)")
        }
    }

    @Test("Settings decode nested or flat, and unsupported filters fail")
    func configuration() throws {
        let nested = try Fixture.configuration()
        #expect(nested.resample == .bicubic)
        #expect(nested.minTiles == 2 && nested.maxTiles == 10 && nested.useThumbnail)
        #expect(nested.useImageSpecialTokens)

        let flat = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self,
            from: Data(#"{"max_tiles": 4, "use_image_special_tokens": false}"#.utf8))
        #expect(flat.maxTiles == 4 && flat.resample == .bilinear && !flat.useImageSpecialTokens)

        #expect(throws: DecodingError.self) { try Fixture.configuration(["resample": 0]) }
        #expect(throws: DecodingError.self) {
            try Fixture.configuration(["min_tiles": 5, "max_tiles": 2])
        }
    }

    @Test("Placeholders must match the images")
    func placeholderCount() throws {
        let configuration = try Fixture.configuration()
        let image = UserInput.Image.array(Fixture.rgb(width: 64, height: 64))
        #expect(throws: VLMError.self) {
            try LFM2VLProcessor.prepare(
                tokens: [1, 10, 10], images: [image], processing: nil,
                configuration: configuration, id: { Fixture.tokens[$0] })
        }
        let text = try LFM2VLProcessor.prepare(
            tokens: [1, 2, 3], images: [], processing: nil, configuration: configuration,
            id: { Fixture.tokens[$0] })
        #expect(text.image == nil)
        #expect(text.text.tokens.asArray(Int.self) == [1, 2, 3])
    }
}

@Suite("LFM2-VL model")
struct LFM2VLModelTests {

    @Test("A tiny random LFM2-VL matches Transformers with a single and a tiled image")
    func tinyModel() throws {
        let directory = Fixture.directory.appendingPathComponent("tiny-lfm2-vl")
        let configuration = try JSONDecoder().decode(
            LFM2VLConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = LFM2VL(configuration)
        let weights = model.sanitize(
            weights: try loadArrays(url: directory.appendingPathComponent("model.safetensors")))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)

        let tiny = Fixture.reference["tiny"] as! [String: Any]
        let images = (tiny["images"] as! [[Int]]).map {
            UserInput.Image.array(Fixture.rgb(width: $0[0], height: $0[1]))
        }
        let image = Fixture.tokens["<image>"]!
        let input = try LFM2VLProcessor.prepare(
            tokens: [1, 40, 41, 42, image, 43, 50, image, 44, 50, 60, 61, 62], images: images,
            processing: nil, configuration: try Fixture.configuration(),
            id: { Fixture.tokens[$0] })
        #expect(input.text.tokens.asArray(Int.self) == tiny["input_ids"] as! [Int])
        #expect(input.image?.frames?.count == 8)

        let cache = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: PrefillParameters(chunking: .unchunked))
        else {
            Issue.record("LFM2-VL returned tokens instead of logits")
            return
        }
        let logits = output.logits[0].asType(.float32)
        let summary = tiny["logits"] as! [String: Any]
        let reference = summary["values"] as! [Double]
        let flat = logits.flattened().asArray(Float.self)
        #expect(flat.count == summary["count"] as! Int)
        var worst = 0.0
        for (index, value) in zip(summary["indices"] as! [Int], reference) {
            worst = max(worst, abs(Double(flat[index]) - value))
        }
        #expect(worst < 2e-3, "worst sampled logit error \(worst)")
        let last = logits[-1].asArray(Float.self).map(Double.init)
        let expected = tiny["last"] as! [Double]
        let lastError = zip(last, expected).map { abs($0 - $1) }.max() ?? .infinity
        #expect(lastError < 2e-3, "worst last-position logit error \(lastError)")
        let total = logits.sum().item(Double.self)
        #expect(abs(total - (summary["sum"] as! Double)) < 1e-4 * (summary["abs_sum"] as! Double))
    }
}

/// d1-3B's MLX export against MLX-VLM fed the Transformers image processor's crops.
/// Runs when `MLX_LFM2VL_D1_3B_8BIT` names a local LiquidAI/d1-3B-MLX-8bit snapshot.
@Suite("d1-3B checkpoint", .serialized)
struct D1ThreeBCheckpointTests {
    static let snapshot = ProcessInfo.processInfo.environment["MLX_LFM2VL_D1_3B_8BIT"]

    @Test(
        "d1-3B MLX 8-bit image decisions match MLX-VLM",
        .enabled(if: snapshot != nil, "set MLX_LFM2VL_D1_3B_8BIT to a local snapshot"))
    func imageDecisions() throws {
        let directory = URL(fileURLWithPath: Self.snapshot!)
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let model = LFM2VL(
            try JSONDecoder.json5().decode(LFM2VLConfiguration.self, from: configData))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        try loadWeights(
            modelDirectory: directory, model: model, quantization: base.quantization,
            perLayerQuantization: base.perLayerQuantization)
        let processorData = try Data(
            contentsOf: directory.appendingPathComponent("processor_config.json"))
        let configuration = try JSONDecoder.json5().decode(
            LFM2VLProcessorConfiguration.self, from: processorData)
        #expect(configuration.resample == .bicubic)

        let url = Fixture.directory.appendingPathComponent("d1-3b-mlx-8bit.json")
        let reference =
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        let tokens = reference["tokens"] as! [String: Int]
        let cases = reference["cases"] as! [[String: Any]]
        #expect(cases.count == 6)
        for entry in cases {
            let size = entry["image"] as! [Int]
            let label = "\(size[0])x\(size[1]) \(entry["question"]!)"
            let input = try LFM2VLProcessor.prepare(
                tokens: entry["input_ids"] as! [Int],
                images: [.array(Fixture.rgb(width: size[0], height: size[1]))],
                processing: nil, configuration: configuration, id: { tokens[$0] })
            #expect(
                input.text.tokens.asArray(Int.self) == entry["expanded_ids"] as! [Int], "\(label)")

            guard
                case .logits(let output) = try model.prepare(
                    input, cache: try model.newCache(parameters: nil), state: nil,
                    prefill: PrefillParameters())
            else {
                Issue.record("LFM2-VL returned tokens instead of logits")
                continue
            }
            let last = output.logits[0, -1].asType(.float32)
            let logz = last - logSumExp(last, axis: -1, keepDims: true)
            let ids = entry["logz_ids"] as! [Int]
            let actual = logz[MLXArray(ids.map(Int32.init))].asArray(Float.self).map(Double.init)
            let expected = entry["logz"] as! [Double]
            let worst = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
            #expect(worst < 0.05, "\(label): worst log-probability error \(worst)")

            // d1's readout: softmax over each option's best alias.
            let values = Dictionary(uniqueKeysWithValues: zip(ids, actual))
            let best = (entry["readout"] as! [[Int]]).map { group in
                group.map { values[$0]! }.max()!
            }
            let peak = best.max()!
            let exps = best.map { Foundation.exp($0 - peak) }
            let probabilities = exps.map { $0 / exps.reduce(0, +) }
            let reference = entry["probabilities"] as! [Double]
            let error = zip(probabilities, reference).map { abs($0 - $1) }.max() ?? .infinity
            #expect(error < 1e-2, "\(label): \(probabilities) vs \(reference)")
        }
    }
}
