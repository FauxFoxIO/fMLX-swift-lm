// Copyright © 2026 Faux Fox, Inc.
// Port of the d1-omni reference `vision.py` (LiquidAI/d1-omni-600M): LFM2-VL's smart resize and
// tiling, with torchvision's uint8 antialiased bilinear resize as MLX-VLM reproduces it.

import Foundation
import MLX

struct D1OmniPreparedCrop {
    /// [H*W, 3*16*16] RGB in [-1, 1], each patch ordered (row, column, channel).
    let patches: MLXArray
    let height: Int
    let width: Int
}

struct D1OmniPreparedImage {
    let crops: [D1OmniPreparedCrop]
    /// Prefix positions after the 2x2 projector.
    var tokenCount: Int { crops.reduce(0) { $0 + $1.height * $1.width / 4 } }
}

struct D1OmniImageLayout: Equatable, Sendable {
    let grid: (columns: Int, rows: Int)
    let thumbnail: (height: Int, width: Int)
    let tiled: Bool

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.grid == rhs.grid && lhs.thumbnail == rhs.thumbnail && lhs.tiled == rhs.tiled
    }
}

enum D1OmniImageProcessor {
    static let tile = 512
    static let patch = 16

    // The reference's `sorted(set(...), key=x*y)`; ties keep CPython's set order.
    static let ratios: [(Int, Int)] = [
        (1, 2), (2, 1), (3, 1), (1, 3), (2, 2), (4, 1), (1, 4), (5, 1), (1, 5), (1, 6), (6, 1),
        (3, 2), (2, 3), (7, 1), (1, 7), (4, 2), (2, 4), (1, 8), (8, 1), (1, 9), (3, 3), (9, 1),
        (2, 5), (5, 2), (10, 1), (1, 10),
    ]

    /// LFM2-VL's smart resize, tile grid, and thumbnail (64 to 256 prefix positions per crop).
    static func layout(width: Int, height: Int) throws -> D1OmniImageLayout {
        guard width >= 1, height >= 1 else { throw DecisionError.invalidInput("empty image") }
        let factor = 32.0
        let maximum = 256.0 * 1024
        let minimum = 64.0 * 1024
        let w0 = Double(width)
        let h0 = Double(height)
        func roundEven(_ x: Double) -> Double { x.rounded(.toNearestOrEven) }
        var h = max(factor, roundEven(h0 / factor) * factor)
        var w = max(factor, roundEven(w0 / factor) * factor)
        if h * w > maximum {
            let beta = (h0 * w0 / maximum).squareRoot()
            h = max(factor, (h0 / beta / factor).rounded(.down) * factor)
            w = max(factor, (w0 / beta / factor).rounded(.down) * factor)
        } else if h * w < minimum {
            let beta = (minimum / (h0 * w0)).squareRoot()
            h = (h0 * beta / factor).rounded(.up) * factor
            w = (w0 * beta / factor).rounded(.up) * factor
        }
        let large =
            max(16, roundEven(h0 / factor) * factor) * max(16, roundEven(w0 / factor) * factor)
            > maximum * 2
        var grid = (1, 1)
        if large {
            var best = Double.infinity
            for ratio in ratios {
                let diff = abs(w0 / h0 - Double(ratio.0) / Double(ratio.1))
                if diff < best
                    || (diff == best
                        && w0 * h0 > 0.5 * Double(tile * tile * ratio.0 * ratio.1))
                {
                    grid = ratio
                    best = diff
                }
            }
        }
        return D1OmniImageLayout(
            grid: (grid.0, grid.1), thumbnail: (Int(h), Int(w)), tiled: large)
    }

    static func prepare(_ image: DecisionRGBImage) throws -> D1OmniPreparedImage {
        let plan = try layout(width: image.width, height: image.height)
        var crops = [DecisionRGBImage]()
        if plan.tiled {
            let (columns, rows) = plan.grid
            let big = resize(image, height: rows * tile, width: columns * tile)
            for row in 0 ..< rows {
                for column in 0 ..< columns {
                    crops.append(
                        big.crop(x: column * tile, y: row * tile, width: tile, height: tile))
                }
            }
        }
        crops.append(resize(image, height: plan.thumbnail.height, width: plan.thumbnail.width))
        return D1OmniPreparedImage(crops: crops.map(patches))
    }

    static func patches(_ crop: DecisionRGBImage) -> D1OmniPreparedCrop {
        let rows = crop.height / patch
        let columns = crop.width / patch
        let normalized = (crop.pixels.asType(.float32) - 127.5) / 127.5
        let patches = normalized.reshaped(rows, patch, columns, patch, 3)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(rows * columns, 3 * patch * patch)
        return D1OmniPreparedCrop(patches: patches, height: rows, width: columns)
    }

    /// torchvision `resize(uint8, BILINEAR, antialias=True)`: width pass, then height pass,
    /// in the same int32 fixed point, so results are byte-exact.
    static func resize(_ image: DecisionRGBImage, height: Int, width: Int) -> DecisionRGBImage {
        var pixels = image.pixels
        if image.width != width {
            pixels = resizeAxis(pixels, axis: 1, output: width)
        }
        if image.height != height {
            pixels = resizeAxis(pixels, axis: 0, output: height)
        }
        return DecisionRGBImage(pixels: pixels)
    }

    struct Taps {
        let index: [Int]  // [output * count]
        let weights: [Int64]
        let count: Int
        let bits: Int
    }

    /// torch's uint8 antialiased bilinear taps: a triangle filter in int16 fixed point.
    static func taps(input: Int, output: Int) -> Taps {
        let scale = Double(input) / Double(output)
        let support = max(scale, 1)
        var lows = [Int](repeating: 0, count: output)
        var sizes = [Int](repeating: 0, count: output)
        for o in 0 ..< output {
            let center = scale * (Double(o) + 0.5)
            lows[o] = max(Int(center - support + 0.5), 0)
            sizes[o] = min(Int(center + support + 0.5), input) - lows[o]
        }
        let count = sizes.max() ?? 0
        var weights = [Double](repeating: 0, count: output * count)
        var largest = 0.0
        for o in 0 ..< output {
            let center = scale * (Double(o) + 0.5)
            var total = 0.0
            for j in 0 ..< count where j < sizes[o] {
                let x = (Double(j + lows[o]) - center + 0.5) * (1 / support)
                let w = max(0, 1 - abs(x))
                weights[o * count + j] = w
            }
            for j in 0 ..< count { total += weights[o * count + j] }
            for j in 0 ..< count {
                weights[o * count + j] /= total
                largest = max(largest, weights[o * count + j])
            }
        }
        var bits = 0
        while bits < 22, Int(0.5 + largest * Double(1 << (bits + 1))) < 1 << 15 {
            bits += 1
        }
        var index = [Int](repeating: 0, count: output * count)
        var fixed = [Int64](repeating: 0, count: output * count)
        for o in 0 ..< output {
            for j in 0 ..< count {
                index[o * count + j] = min(lows[o] + j, input - 1)
                fixed[o * count + j] = Int64(0.5 + weights[o * count + j] * Double(1 << bits))
            }
        }
        return Taps(index: index, weights: fixed, count: count, bits: bits)
    }

    private static func resizeAxis(_ pixels: MLXArray, axis: Int, output: Int) -> MLXArray {
        let taps = taps(input: pixels.dim(axis), output: output)
        let source = pixels.asType(.int32)
        let shape = axis == 1 ? [1, output, 1] : [output, 1, 1]
        var accumulator = MLXArray(Int32(1 << (taps.bits - 1)))
        for k in 0 ..< taps.count {
            let index = MLXArray((0 ..< output).map { Int32(taps.index[$0 * taps.count + k]) })
            let weight = MLXArray((0 ..< output).map { Int32(taps.weights[$0 * taps.count + k]) })
            accumulator = accumulator + take(source, index, axis: axis) * weight.reshaped(shape)
        }
        return clip(rightShift(accumulator, MLXArray(Int32(taps.bits))), min: 0, max: 255)
            .asType(.uint8)
    }
}
