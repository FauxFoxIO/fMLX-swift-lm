// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import Testing

@testable import MLXDecision

#if canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

struct D1OmniImageTests {
    @Test("Smart resize, grid, and thumbnail match LFM2-VL")
    func layouts() throws {
        let cases = try Fixtures.json("d1-image-layouts.json") as! [[String: Any]]
        for entry in cases {
            let layout = try D1OmniImageProcessor.layout(
                width: entry["width"] as! Int, height: entry["height"] as! Int)
            let grid = entry["grid"] as! [Int]
            let thumbnail = entry["thumbnail"] as! [Int]
            #expect(layout.grid.columns == grid[0] && layout.grid.rows == grid[1], "\(entry)")
            #expect(
                layout.thumbnail.height == thumbnail[0] && layout.thumbnail.width == thumbnail[1])
            #expect(layout.tiled == entry["tiled"] as! Bool)
        }
        #expect(throws: DecisionError.self) { try D1OmniImageProcessor.layout(width: 0, height: 4) }
    }

    @Test("Tiles and thumbnails are byte-exact with torchvision's uint8 resize")
    func crops() throws {
        let reference = try Fixtures.json("tiny-d1-omni/reference.json") as! [String: Any]
        for entry in reference["images"] as! [[String: Any]] {
            let width = entry["width"] as! Int
            let height = entry["height"] as! Int
            let bytes = Fixtures.rgb(width: width, height: height)
            #expect(Fixtures.checksum(bytes) == entry["pixel_checksum"] as! Int64)
            let image = DecisionRGBImage(bytes: bytes, width: width, height: height)
            let layout = try D1OmniImageProcessor.layout(width: width, height: height)
            var crops = [DecisionRGBImage]()
            if layout.tiled {
                let big = D1OmniImageProcessor.resize(
                    image, height: layout.grid.rows * 512, width: layout.grid.columns * 512)
                for row in 0 ..< layout.grid.rows {
                    for column in 0 ..< layout.grid.columns {
                        crops.append(
                            big.crop(x: column * 512, y: row * 512, width: 512, height: 512))
                    }
                }
            }
            crops.append(
                D1OmniImageProcessor.resize(
                    image, height: layout.thumbnail.height, width: layout.thumbnail.width))
            let expected = entry["crops"] as! [[String: Any]]
            #expect(crops.count == expected.count)
            for (crop, reference) in zip(crops, expected) {
                #expect(crop.height == reference["height"] as! Int)
                #expect(crop.width == reference["width"] as! Int)
                #expect(
                    Fixtures.checksum(crop.bytes) == reference["checksum"] as! Int64,
                    "\(width)x\(height)")
            }
            let prepared = try D1OmniImageProcessor.prepare(image)
            #expect(prepared.tokenCount == (entry["prefix_shape"] as! [Int])[0])
        }
    }

    @Test("Patches keep row, column, channel order in [-1, 1]")
    func patchOrder() {
        var bytes = [UInt8](repeating: 0, count: 32 * 16 * 3)
        bytes[(0 * 32 + 17) * 3 + 2] = 255  // second patch, first row, column 1, blue
        let crop = D1OmniImageProcessor.patches(
            DecisionRGBImage(bytes: bytes, width: 32, height: 16))
        #expect(crop.height == 1 && crop.width == 2)
        let values = crop.patches.asArray(Float.self)
        #expect(values[768 + 1 * 3 + 2] == 1)
        #expect(values[0] == -1)
        #expect(values.filter { $0 == 1 }.count == 1)
    }

    @Test("Antialiased bilinear rows sum to one")
    func bilinearRows() {
        for (input, output) in [(16, 32), (16, 7), (16, 16), (16, 3), (16, 64)] {
            let matrix = D1OmniVisionEmbeddings.bilinearMatrix(input: input, output: output)
            for row in 0 ..< output {
                let total = matrix[(row * input) ..< ((row + 1) * input)].reduce(0, +)
                #expect(abs(total - 1) < 1e-5)
            }
        }
        let identity = D1OmniVisionEmbeddings.bilinearMatrix(input: 4, output: 4)
        #expect(identity == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    }

    @Test("Media inputs are validated")
    func mediaValidation() throws {
        #expect(throws: DecisionError.self) {
            try DecisionMedia.rgb(.rgb(bytes: [0, 0], width: 1, height: 1))
        }
        #expect(throws: DecisionError.self) {
            try DecisionMedia.rgb(.encoded(Data([1, 2, 3])))
        }
        let image = try DecisionMedia.rgb(.rgb(bytes: [1, 2, 3, 4, 5, 6], width: 2, height: 1))
        #expect(image.crop(x: 1, y: 0, width: 1, height: 1).bytes == [4, 5, 6])
    }

    #if canImport(ImageIO)
    /// PNG data for packed RGB bytes, with an optional EXIF orientation.
    static func png(_ bytes: [UInt8], width: Int, height: Int, orientation: Int = 1) throws -> Data
    {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< width * height {
            for channel in 0 ..< 3 { rgba[pixel * 4 + channel] = bytes[pixel * 3 + channel] }
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let image = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(
            destination, image,
            [kCGImagePropertyOrientation as String: orientation] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        return data as Data
    }

    @Test("Encoded images decode to their RGB bytes and turn upright by EXIF orientation")
    func encodedImages() throws {
        let bytes = Fixtures.rgb(width: 40, height: 24)
        let decoded = try DecisionMedia.rgb(.encoded(try Self.png(bytes, width: 40, height: 24)))
        #expect(decoded.width == 40 && decoded.height == 24)
        #expect(decoded.bytes == bytes)
        // Orientation 6: the stored image is shown rotated 90 degrees clockwise.
        let rotated = try DecisionMedia.rgb(
            .encoded(try Self.png(bytes, width: 40, height: 24, orientation: 6)))
        #expect(rotated.width == 24 && rotated.height == 40)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("decision-\(UUID().uuidString).png")
        try Self.png(bytes, width: 40, height: 24).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(try DecisionMedia.rgb(.file(url)).bytes == bytes)
    }
    #endif
}
