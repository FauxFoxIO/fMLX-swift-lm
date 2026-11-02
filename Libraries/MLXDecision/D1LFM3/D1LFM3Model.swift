// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXLMCommon

/// One image as the LFM3-VL processor emits it: pixel patches [rows × columns, patch values] in
/// merge-block order, row major.
public struct D1LFM3Image: Sendable {
    public var pixels: [Float16]
    public var rows: Int
    public var columns: Int

    public init(pixels: [Float16], rows: Int, columns: Int) {
        self.pixels = pixels
        self.rows = rows
        self.columns = columns
    }
}

/// A loaded d1-2B trunk for a host that renders d1's prompt itself (C0 in Mirage). Calls run one at a
/// time. Each call returns the final-norm hidden state at every answer slot, or the option logits the
/// tied head gives them, from the model's own embedding table.
public final class D1LFM3Model: Sendable {
    public let configuration: D1LFM3Configuration
    /// The readout temperature in `calibration.json` (`default`), when present.
    public let temperature: Double?
    public let readsImages: Bool
    /// Bytes of the weight files.
    public let weightBytes: Int64

    private let state: SerialAccessContainer<D1LFM3Engine>

    private init(_ engine: consuming D1LFM3Engine, temperature: Double?, bytes: Int64) {
        configuration = engine.configuration
        readsImages = engine.readsImages
        self.temperature = temperature
        weightBytes = bytes
        state = SerialAccessContainer(consume engine)
    }

    /// Loads an MLX export (or the merged checkpoint) from a local directory. `dtype` overrides the
    /// activation type; float16 matches the fp32 reference best.
    public static func load(
        directory: URL, dtype: DType? = nil, vision: Bool = true,
        weightLoading: WeightLoading = .memoryMapped
    ) throws -> D1LFM3Model {
        let engine = try D1LFM3Engine.load(
            directory: directory, dtype: dtype, vision: vision, weightLoading: weightLoading)
        struct Calibration: Decodable { let `default`: Double? }
        let temperature =
            (try? Data(contentsOf: directory.appendingPathComponent("calibration.json")))
            .flatMap { try? JSONDecoder().decode(Calibration.self, from: $0) }?.default
        let bytes = DecisionWeights.byteCount((try? DecisionWeights.files(in: directory)) ?? [])
        return D1LFM3Model(engine, temperature: temperature, bytes: bytes)
    }

    /// Runs a text row and a one-image row once, so the first request runs warm.
    public func prepare() async throws {
        try await state.read { engine in
            try withError {
                let sequence = D1LFM3Sequence(
                    ids: [1, 2, 3], stateLength: 1, groups: [], slots: [2])
                eval(try engine.hidden(sequence, features: nil))
                if engine.readsImages {
                    let patches = MLXArray.zeros([16, engine.configuration.vision!.patchValues])
                    eval(try engine.imageFeatures(patches, rows: 4, columns: 4))
                }
            }
        }
    }

    /// The hidden state at each answer slot, [K][hidden] float32. `images` fill the image
    /// placeholders of `sequence` in order.
    public func hidden(_ sequence: D1LFM3Sequence, images: [D1LFM3Image] = []) async throws
        -> [[Float]]
    {
        try await state.read { engine in
            let hidden = try Self.run(engine, sequence, images)
            let width = hidden.dim(1)
            let values = hidden.asArray(Float.self)
            return (0 ..< hidden.dim(0)).map { Array(values[$0 * width ..< ($0 + 1) * width]) }
        }
    }

    /// Option logits per slot: for slot k, `readout[k][option]` lists the option's token forms, and an
    /// option scores the best of `hidden · embedding[form]`. Divide by ``temperature`` for the softmax.
    public func optionLogits(
        _ sequence: D1LFM3Sequence, images: [D1LFM3Image] = [], readout: [[[Int32]]]
    ) async throws -> [[Float]] {
        guard readout.count == sequence.slots.count else {
            throw DecisionError.invalidInput("one readout table per answer slot is required")
        }
        return try await state.read { engine in
            let hidden = try Self.run(engine, sequence, images)
            var results = [[Float]]()
            for (k, options) in readout.enumerated() {
                let tokens = options.flatMap { $0 }
                let scores = try withError {
                    engine.scores(hidden[k ..< k + 1], tokens: tokens)[0].asArray(Float.self)
                }
                var start = 0
                results.append(
                    options.map { forms in
                        defer { start += forms.count }
                        return forms.isEmpty
                            ? -.infinity : scores[start ..< start + forms.count].max()!
                    })
            }
            return results
        }
    }

    /// MLX reports failures (a Metal command-buffer timeout under GPU contention, say) through a
    /// handler that exits the process by default; inside `withError` they throw instead.
    private static func run(
        _ engine: D1LFM3Engine, _ sequence: D1LFM3Sequence, _ images: [D1LFM3Image]
    )
        throws -> MLXArray
    {
        try withError { try forward(engine, sequence, images) }
    }

    private static func forward(
        _ engine: D1LFM3Engine, _ sequence: D1LFM3Sequence, _ images: [D1LFM3Image]
    ) throws -> MLXArray {
        let patchValues = engine.configuration.vision?.patchValues ?? 0
        let features = try images.map { image -> MLXArray in
            guard image.pixels.count == image.rows * image.columns * patchValues else {
                throw DecisionError.invalidInput(
                    "image pixels do not match its \(image.rows)×\(image.columns) grid")
            }
            return try engine.imageFeatures(
                MLXArray(image.pixels, [image.rows * image.columns, patchValues]), rows: image.rows,
                columns: image.columns)
        }
        let hidden = try engine.hidden(
            sequence, features: features.isEmpty ? nil : concatenated(features))
        eval(hidden)
        guard isFinite(hidden).all().item(Bool.self) else { throw DecisionError.nonFiniteOutput }
        return hidden
    }
}
