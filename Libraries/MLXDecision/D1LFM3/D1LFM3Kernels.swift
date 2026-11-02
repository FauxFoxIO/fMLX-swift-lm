// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX

enum D1LFM3Kernels {
    /// One Metal pass for `C * conv(B * X)`: each thread reads its channel's window, accumulates in float32.
    nonisolated(unsafe) static let shortConvolutionKernel = MLXFast.metalKernel(
        name: "d1_lfm3_short_conv",
        inputNames: ["bcx", "taps"],
        outputNames: ["y"],
        source: """
            uint channel = thread_position_in_grid.x;
            uint t = thread_position_in_grid.y;
            if (channel >= D) { return; }
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                int s = int(t) - (K - 1) + k;
                if (s >= 0) {
                    size_t row = size_t(s) * 3 * D;
                    acc += float(taps[k * D + channel]) * float(bcx[row + channel])
                        * float(bcx[row + 2 * D + channel]);
                }
            }
            y[size_t(t) * D + channel] = T(float(bcx[size_t(t) * 3 * D + D + channel]) * acc);
            """)

    /// `C * conv(B * X)` for `bcx` [1, L, 3D] = [B | C | X] and causal taps [k, D].
    static func shortConvolution(_ bcx: MLXArray, weight: MLXArray, fused: Bool = true) -> MLXArray
    {
        let (length, width) = (bcx.dim(1), bcx.dim(2) / 3)
        if fused, bcx.dim(0) == 1 {
            return shortConvolutionKernel(
                [bcx, weight.asType(bcx.dtype)],
                template: [("T", bcx.dtype), ("D", width), ("K", weight.dim(0))],
                grid: (width, length, 1), threadGroup: (min(256, width), 1, 1),
                outputShapes: [[1, length, width]], outputDTypes: [bcx.dtype])[0]
        }
        let parts = split(bcx, parts: 3, axis: -1)
        let bx = parts[0] * parts[2]
        let taps = weight.dim(0)
        let padded = MLX.padded(bx, widths: [0, [taps - 1, 0], 0])
        var y = padded[0..., 0 ..< length] * weight[0]
        for tap in 1 ..< taps {
            y = y + padded[0..., tap ..< (tap + length)] * weight[tap]
        }
        return parts[1] * y
    }
}

extension D1LFM3Engine {
    /// The short convolution as the trunk runs it, fused into one kernel or as array operations.
    public static func shortConvolution(_ bcx: MLXArray, weight: MLXArray, fused: Bool) -> MLXArray
    {
        D1LFM3Kernels.shortConvolution(bcx, weight: weight, fused: fused)
    }
}
