// Copyright © 2026 Faux Fox, Inc.
// Port of Transformers' `Lfm2VlImageProcessor` and the image markup of `Lfm2VlProcessor`
// (Apache-2.0), with torch's uint8 antialiased resize so crops match the reference exactly.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

/// The image settings of an LFM2-VL checkpoint (`processor_config.json` or
/// `preprocessor_config.json`), with Transformers' defaults.
public struct LFM2VLProcessorConfiguration: Codable, Sendable {
    /// Resampling filters that the reference applies to uint8 images.
    public enum Resample: Int, Codable, Sendable {
        case lanczos = 1
        case bilinear = 2
        case bicubic = 3
    }

    public var imageMean: [Float]
    public var imageStd: [Float]
    public var rescaleFactor: Float
    public var tileSize: Int
    public var encoderPatchSize: Int
    public var downsampleFactor: Int
    public var doImageSplitting: Bool
    public var minTiles: Int
    public var maxTiles: Int
    public var useThumbnail: Bool
    public var minImageTokens: Int
    public var maxImageTokens: Int
    public var maxPixelsTolerance: Double
    public var resample: Resample
    /// Wrap each image in `<|image_start|>`/`<|image_end|>` and mark tiles and the thumbnail.
    public var useImageSpecialTokens: Bool

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case rescaleFactor = "rescale_factor"
        case tileSize = "tile_size"
        case encoderPatchSize = "encoder_patch_size"
        case downsampleFactor = "downsample_factor"
        case doImageSplitting = "do_image_splitting"
        case minTiles = "min_tiles"
        case maxTiles = "max_tiles"
        case useThumbnail = "use_thumbnail"
        case minImageTokens = "min_image_tokens"
        case maxImageTokens = "max_image_tokens"
        case maxPixelsTolerance = "max_pixels_tolerance"
        case resample
        case useImageSpecialTokens = "use_image_special_tokens"
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: CodingKeys.self)
        // `processor_config.json` nests the image settings; `preprocessor_config.json` does not.
        let c =
            root.contains(.imageProcessor)
            ? try root.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageProcessor) : root
        imageMean = try c.decodeIfPresent([Float].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([Float].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        rescaleFactor = try c.decodeIfPresent(Float.self, forKey: .rescaleFactor) ?? 1 / 255
        tileSize = try c.decodeIfPresent(Int.self, forKey: .tileSize) ?? 512
        encoderPatchSize = try c.decodeIfPresent(Int.self, forKey: .encoderPatchSize) ?? 16
        downsampleFactor = try c.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 2
        doImageSplitting = try c.decodeIfPresent(Bool.self, forKey: .doImageSplitting) ?? true
        minTiles = try c.decodeIfPresent(Int.self, forKey: .minTiles) ?? 2
        maxTiles = try c.decodeIfPresent(Int.self, forKey: .maxTiles) ?? 10
        useThumbnail = try c.decodeIfPresent(Bool.self, forKey: .useThumbnail) ?? true
        minImageTokens = try c.decodeIfPresent(Int.self, forKey: .minImageTokens) ?? 64
        maxImageTokens = try c.decodeIfPresent(Int.self, forKey: .maxImageTokens) ?? 256
        maxPixelsTolerance = try c.decodeIfPresent(Double.self, forKey: .maxPixelsTolerance) ?? 2
        let resampleCode = try c.decodeIfPresent(Int.self, forKey: .resample) ?? 2
        guard let resample = Resample(rawValue: resampleCode) else {
            throw DecodingError.dataCorruptedError(
                forKey: .resample, in: c,
                debugDescription: "LFM2-VL supports resample 1, 2, or 3, not \(resampleCode)")
        }
        self.resample = resample
        useImageSpecialTokens =
            try root.decodeIfPresent(Bool.self, forKey: .useImageSpecialTokens)
            ?? c.decodeIfPresent(Bool.self, forKey: .useImageSpecialTokens) ?? true
        guard imageMean.count == 3, imageStd.count == 3, tileSize > 0, encoderPatchSize > 0,
            downsampleFactor > 0, tileSize % encoderPatchSize == 0, minTiles >= 1,
            !doImageSplitting || minTiles <= maxTiles, minImageTokens >= 1,
            maxImageTokens >= minImageTokens
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "invalid LFM2-VL image processor settings"))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(imageMean, forKey: .imageMean)
        try c.encode(imageStd, forKey: .imageStd)
        try c.encode(rescaleFactor, forKey: .rescaleFactor)
        try c.encode(tileSize, forKey: .tileSize)
        try c.encode(encoderPatchSize, forKey: .encoderPatchSize)
        try c.encode(downsampleFactor, forKey: .downsampleFactor)
        try c.encode(doImageSplitting, forKey: .doImageSplitting)
        try c.encode(minTiles, forKey: .minTiles)
        try c.encode(maxTiles, forKey: .maxTiles)
        try c.encode(useThumbnail, forKey: .useThumbnail)
        try c.encode(minImageTokens, forKey: .minImageTokens)
        try c.encode(maxImageTokens, forKey: .maxImageTokens)
        try c.encode(maxPixelsTolerance, forKey: .maxPixelsTolerance)
        try c.encode(resample.rawValue, forKey: .resample)
        try c.encode(useImageSpecialTokens, forKey: .useImageSpecialTokens)
    }
}

/// How one image becomes encoder crops: one resized crop, or a grid of tiles and then a thumbnail.
public struct LFM2VLImageLayout: Sendable, Hashable {
    public let rows: Int
    public let columns: Int
    /// Size of the single crop, or of the thumbnail that follows the tiles.
    public let height: Int
    public let width: Int
    /// The image is cut into `rows` x `columns` tiles of the tile size.
    public let tiled: Bool
    /// A thumbnail of `height` x `width` follows the tiles.
    public let thumbnail: Bool

    public var isMultiTile: Bool { rows > 1 || columns > 1 }
}

/// One crop as encoder patches.
struct LFM2VLCrop {
    /// [rows * columns, patch * patch * 3], each patch ordered (row, column, channel), normalized.
    let patches: MLXArray
    /// The crop's patch grid.
    let rows: Int
    let columns: Int
}

enum LFM2VLImageProcessing {

    // MARK: Layout

    /// `sorted(set(ratios), key=lambda r: r[0] * r[1])` with ties in CPython's set order.
    static func targetRatios(minTiles: Int, maxTiles: Int) -> [(Int, Int)] {
        var ratios = [(Int, Int)]()
        if minTiles <= maxTiles {
            for n in minTiles ... maxTiles {
                for w in 1 ... n {
                    for h in 1 ... n where minTiles <= w * h && w * h <= maxTiles {
                        ratios.append((w, h))
                    }
                }
            }
        }
        let unique = PythonSetOrder.iterate(ratios)
        return unique.enumerated()
            .sorted { lhs, rhs in
                let a = lhs.element.0 * lhs.element.1
                let b = rhs.element.0 * rhs.element.1
                return a != b ? a < b : lhs.offset < rhs.offset
            }
            .map(\.element)
    }

    static func layout(width: Int, height: Int, configuration c: LFM2VLProcessorConfiguration)
        throws -> LFM2VLImageLayout
    {
        guard width >= 1, height >= 1 else {
            throw VLMError.imageProcessingFailure("the image is empty")
        }
        func roundByFactor(_ n: Int, _ factor: Int) -> Int {
            Int((Double(n) / Double(factor)).rounded(.toNearestOrEven)) * factor
        }
        let patch = c.encoderPatchSize
        let factor = patch * c.downsampleFactor
        let pixelsPerToken = patch * patch * c.downsampleFactor * c.downsampleFactor
        let maxPixels = Double(c.maxImageTokens * pixelsPerToken)
        let minPixels = Double(c.minImageTokens * pixelsPerToken)
        let h0 = Double(height)
        let w0 = Double(width)

        // smart_resize
        var h = max(factor, roundByFactor(height, factor))
        var w = max(factor, roundByFactor(width, factor))
        if Double(h * w) > maxPixels {
            let beta = (h0 * w0 / maxPixels).squareRoot()
            h = max(factor, Int((h0 / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((w0 / beta / Double(factor)).rounded(.down)) * factor)
        } else if Double(h * w) < minPixels {
            let beta = (minPixels / (h0 * w0)).squareRoot()
            h = Int((h0 * beta / Double(factor)).rounded(.up)) * factor
            w = Int((w0 * beta / Double(factor)).rounded(.up)) * factor
        }

        // _is_image_too_large
        let largeH = max(patch, roundByFactor(height, factor))
        let largeW = max(patch, roundByFactor(width, factor))
        let large = Double(largeH * largeW) > maxPixels * c.maxPixelsTolerance
        let (minTiles, maxTiles) = c.doImageSplitting ? (c.minTiles, c.maxTiles) : (1, 1)
        guard large, !(minTiles == 1 && maxTiles == 1) else {
            return LFM2VLImageLayout(
                rows: 1, columns: 1, height: h, width: w, tiled: false, thumbnail: false)
        }

        // find_closest_aspect_ratio
        let aspect = w0 / h0
        let area = w0 * h0
        var best = (1, 1)
        var bestDiff = Double.infinity
        for ratio in targetRatios(minTiles: minTiles, maxTiles: maxTiles) {
            let diff = abs(aspect - Double(ratio.0) / Double(ratio.1))
            if diff < bestDiff {
                bestDiff = diff
                best = ratio
            } else if diff == bestDiff,
                area > 0.5 * Double(c.tileSize * c.tileSize * ratio.0 * ratio.1)
            {
                best = ratio
            }
        }
        let thumbnail = c.useThumbnail && best.0 * best.1 != 1
        return LFM2VLImageLayout(
            rows: best.1, columns: best.0, height: h, width: w, tiled: true, thumbnail: thumbnail)
    }

    // MARK: Crops

    /// The crops of one uint8 RGB image [height, width, 3], tiles first, as the reference orders them.
    static func crops(
        _ pixels: MLXArray, layout: LFM2VLImageLayout, configuration c: LFM2VLProcessorConfiguration
    ) -> [LFM2VLCrop] {
        var images = [MLXArray]()
        if layout.tiled {
            let tile = c.tileSize
            let big = resize(
                pixels, height: layout.rows * tile, width: layout.columns * tile,
                filter: c.resample)
            for row in 0 ..< layout.rows {
                for column in 0 ..< layout.columns {
                    images.append(
                        big[(row * tile) ..< ((row + 1) * tile),
                            (column * tile) ..< ((column + 1) * tile)])
                }
            }
            if layout.thumbnail {
                images.append(
                    resize(pixels, height: layout.height, width: layout.width, filter: c.resample))
            }
        } else {
            images.append(
                resize(pixels, height: layout.height, width: layout.width, filter: c.resample))
        }
        let mean = MLXArray(c.imageMean.map { $0 / c.rescaleFactor })
        let std = MLXArray(c.imageStd.map { $0 / c.rescaleFactor })
        let patch = c.encoderPatchSize
        return images.map { image in
            let rows = image.dim(0) / patch
            let columns = image.dim(1) / patch
            let normalized = (image.asType(.float32) - mean) / std
            let patches = normalized.reshaped(rows, patch, columns, patch, 3)
                .transposed(0, 2, 1, 3, 4)
                .reshaped(rows * columns, patch * patch * 3)
            return LFM2VLCrop(patches: patches, rows: rows, columns: columns)
        }
    }

    // MARK: Markup

    /// The tokens one `<image>` placeholder expands to, as `Lfm2VlProcessor` writes them.
    static func imageTokens(
        layout: LFM2VLImageLayout, configuration c: LFM2VLProcessorConfiguration,
        id: (String) -> Int?
    ) throws -> [Int] {
        func token(_ text: String) throws -> Int {
            guard let value = id(text) else {
                throw VLMError.processing("the tokenizer has no \(text) token")
            }
            return value
        }
        let image = try token("<image>")
        let special = c.useImageSpecialTokens
        let patch = c.encoderPatchSize
        let df = c.downsampleFactor
        func count(_ height: Int, _ width: Int) -> Int {
            let rows = (height / patch + df - 1) / df
            let columns = (width / patch + df - 1) / df
            return rows * columns
        }
        var result = [Int]()
        if special { result.append(try token("<|image_start|>")) }
        if layout.isMultiTile {
            let perTile = count(c.tileSize, c.tileSize)
            for row in 1 ... layout.rows {
                for column in 1 ... layout.columns {
                    if special { result.append(try token("<|img_row_\(row)_col_\(column)|>")) }
                    result += Array(repeating: image, count: perTile)
                }
            }
            if layout.thumbnail {
                if special { result.append(try token("<|img_thumbnail|>")) }
                result += Array(repeating: image, count: count(layout.height, layout.width))
            }
        } else if layout.tiled {
            // A 1x1 grid is one tile-sized crop.
            result += Array(repeating: image, count: count(c.tileSize, c.tileSize))
        } else {
            result += Array(repeating: image, count: count(layout.height, layout.width))
        }
        if special { result.append(try token("<|image_end|>")) }
        return result
    }

    // MARK: Resize

    struct Taps {
        /// [output * count] source indices and int16 fixed-point weights.
        let index: [Int32]
        let weights: [Int32]
        let count: Int
        let bits: Int
    }

    /// torch's uint8 antialiased taps (`_compute_index_ranges_int16_weights`).
    static func taps(
        input: Int, output: Int, filter: LFM2VLProcessorConfiguration.Resample
    ) -> Taps {
        let half: Double
        let kernel: (Double) -> Double
        switch filter {
        case .bilinear:
            half = 1
            kernel = { x in
                let x = abs(x)
                return x < 1 ? 1 - x : 0
            }
        case .bicubic:
            half = 2
            kernel = { x in
                let a = -0.5
                let x = abs(x)
                if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
                if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
                return 0
            }
        case .lanczos:
            half = 3
            kernel = { x in
                func sinc(_ x: Double) -> Double {
                    if x == 0 { return 1 }
                    let y = x * Double.pi
                    return Foundation.sin(y) / y
                }
                let x = abs(x)
                return x < 3 ? sinc(x) * sinc(x / 3) : 0
            }
        }
        let scale = Double(input) / Double(output)
        let support = scale >= 1 ? half * scale : half
        let count = Int(support.rounded(.up)) * 2 + 1
        let invscale = scale >= 1 ? 1 / scale : 1
        var weights = [Double](repeating: 0, count: output * count)
        var lows = [Int](repeating: 0, count: output)
        var largest = 0.0
        for o in 0 ..< output {
            let center = scale * (Double(o) + 0.5)
            let low = max(Int(center - support + 0.5), 0)
            let size = min(max(min(Int(center + support + 0.5), input) - low, 0), count)
            lows[o] = low
            var total = 0.0
            for j in 0 ..< size {
                let w = kernel((Double(j + low) - center + 0.5) * invscale)
                weights[o * count + j] = w
                total += w
            }
            if total != 0 {
                for j in 0 ..< size {
                    weights[o * count + j] /= total
                    largest = max(largest, weights[o * count + j])
                }
            }
        }
        var bits = 0
        while bits < 22, Int(0.5 + largest * Double(1 << (bits + 1))) < 1 << 15 {
            bits += 1
        }
        var index = [Int32](repeating: 0, count: output * count)
        var fixed = [Int32](repeating: 0, count: output * count)
        for o in 0 ..< output {
            for j in 0 ..< count {
                index[o * count + j] = Int32(min(lows[o] + j, input - 1))
                let v = weights[o * count + j] * Double(1 << bits)
                fixed[o * count + j] = Int32(v < 0 ? Int(-0.5 + v) : Int(0.5 + v))
            }
        }
        return Taps(index: index, weights: fixed, count: count, bits: bits)
    }

    /// torchvision `resize(uint8, antialias=True)` on [height, width, 3]: width pass, then height
    /// pass, each in torch's int32 fixed point, so results are byte-exact.
    static func resize(
        _ pixels: MLXArray, height: Int, width: Int, filter: LFM2VLProcessorConfiguration.Resample
    ) -> MLXArray {
        var result = pixels
        if result.dim(1) != width { result = resizeAxis(result, axis: 1, output: width, filter) }
        if result.dim(0) != height { result = resizeAxis(result, axis: 0, output: height, filter) }
        return result
    }

    private static func resizeAxis(
        _ pixels: MLXArray, axis: Int, output: Int, _ filter: LFM2VLProcessorConfiguration.Resample
    ) -> MLXArray {
        let taps = taps(input: pixels.dim(axis), output: output, filter: filter)
        let source = pixels.asType(.int32)
        let shape = axis == 1 ? [1, output, 1] : [output, 1, 1]
        var accumulator = MLXArray(Int32(1 << (taps.bits - 1)))
        for k in 0 ..< taps.count {
            let index = MLXArray((0 ..< output).map { taps.index[$0 * taps.count + k] })
            let weight = MLXArray((0 ..< output).map { taps.weights[$0 * taps.count + k] })
            accumulator = accumulator + take(source, index, axis: axis) * weight.reshaped(shape)
        }
        // Negative sums clamp to 0 either way; clamping first keeps the shift arithmetic-free.
        let shifted = rightShift(maximum(accumulator, MLXArray(Int32(0))), MLXArray(Int32(taps.bits)))
        return minimum(shifted, MLXArray(Int32(255))).asType(.uint8)
    }

    // MARK: Pixels

    /// The image's own 8-bit samples [height, width, 3], without color management, as PIL reads
    /// them. File images are turned upright by their EXIF orientation.
    static func pixels(_ image: UserInput.Image, processing: UserInput.Processing?) throws
        -> MLXArray
    {
        if case .array(let array) = image, processing?.resize == nil {
            return try rgb(array)
        }
        var ciImage: CIImage
        switch image {
        case .url(let url):
            guard let loaded = CIImage(contentsOf: url, options: [.applyOrientationProperty: true])
            else { throw VLMError.imageProcessingFailure("cannot load \(url.lastPathComponent)") }
            ciImage = loaded
        default:
            ciImage = try image.asCIImage()
        }
        ciImage = MediaProcessing.apply(ciImage, processing: processing)
        return try render(ciImage)
    }

    /// An [H, W, 3|4] or [3|4, H, W] array, uint8 or float in 0...255 (or 0...1), as uint8 RGB.
    private static func rgb(_ array: MLXArray) throws -> MLXArray {
        guard array.ndim == 3 else {
            throw VLMError.imageProcessingFailure("an image array needs 3 dimensions")
        }
        var x = array
        if [3, 4].contains(x.dim(0)), ![3, 4].contains(x.dim(-1)) { x = x.transposed(1, 2, 0) }
        guard [3, 4].contains(x.dim(-1)), x.dim(0) > 0, x.dim(1) > 0 else {
            throw VLMError.imageProcessingFailure("an image array needs 3 or 4 channels")
        }
        x = x[0..., 0..., ..<3]
        if x.dtype == .uint8 { return x }
        var values = x.asType(.float32)
        if values.max().item(Float.self) <= 1 { values = values * 255 }
        return clip(round(values), min: 0, max: 255).asType(.uint8)
    }

    private static let rawContext = CIContext(options: [
        .workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .cacheIntermediates: false,
    ])

    private static func render(_ image: CIImage) throws -> MLXArray {
        let extent = image.extent
        guard !extent.isInfinite, extent.width >= 1, extent.height >= 1 else {
            throw VLMError.imageProcessingFailure("the image has no finite extent")
        }
        let width = Int(extent.width.rounded())
        let height = Int(extent.height.rounded())
        var data = Data(count: width * height * 4)
        data.withUnsafeMutableBytes { pointer in
            rawContext.render(
                image, toBitmap: pointer.baseAddress!, rowBytes: width * 4, bounds: extent,
                format: .RGBA8, colorSpace: nil)
        }
        let rgba = MLXArray(data, [height, width, 4], type: UInt8.self)
        let rgb = rgba[0..., 0..., ..<3]
        // Core Image renders premultiplied alpha; PIL keeps straight samples.
        let alpha = rgba[0..., 0..., 3 ..< 4].asType(.float32)
        let straight = round(rgb.asType(.float32) * 255 / maximum(alpha, MLXArray(Float(1))))
        return MLX.where(
            alpha .== 255, rgb, clip(straight, min: 0, max: 255).asType(.uint8))
    }
}

/// CPython's iteration order for a set of `(Int, Int)` tuples built by adding items in order.
enum PythonSetOrder {
    /// Python's `hash((a, b))` for small non-negative ints (xxHash-based `tuplehash`).
    static func hash(_ a: Int, _ b: Int) -> UInt64 {
        let prime1: UInt64 = 11_400_714_785_074_694_791
        let prime2: UInt64 = 14_029_467_366_897_019_727
        let prime5: UInt64 = 2_870_177_450_012_600_261
        var acc = prime5
        for lane in [UInt64(a), UInt64(b)] {
            acc = acc &+ lane &* prime2
            acc = (acc << 31) | (acc >> 33)
            acc = acc &* prime1
        }
        acc = acc &+ (2 ^ (prime5 ^ 3_527_539))
        return acc == UInt64.max ? 1_546_275_796 : acc
    }

    static func iterate(_ items: [(Int, Int)]) -> [(Int, Int)] {
        typealias Entry = (key: (Int, Int), hash: UInt64)
        let linearProbes = 9
        var table = [Entry?](repeating: nil, count: 8)
        var used = 0

        func slot(for hash: UInt64, in table: [Entry?], matching key: (Int, Int)?) -> Int {
            let mask = table.count - 1
            var perturb = hash
            var i = Int(hash & UInt64(mask))
            while true {
                let probes = i + linearProbes <= mask ? linearProbes : 0
                for j in i ... (i + probes) {
                    guard let entry = table[j] else { return j }
                    if let key, entry.hash == hash, entry.key == key { return j }
                }
                perturb >>= 5
                i = Int((UInt64(i) &* 5 &+ 1 &+ perturb) & UInt64(mask))
            }
        }

        for key in items {
            let h = hash(key.0, key.1)
            let index = slot(for: h, in: table, matching: key)
            if table[index] != nil { continue }
            table[index] = (key, h)
            used += 1
            if used * 5 >= (table.count - 1) * 3 {
                var size = 8
                while size <= used * 4 { size <<= 1 }
                var resized = [Entry?](repeating: nil, count: size)
                for case let entry? in table {
                    resized[slot(for: entry.hash, in: resized, matching: nil)] = entry
                }
                table = resized
            }
        }
        return table.compactMap { $0?.key }
    }
}
