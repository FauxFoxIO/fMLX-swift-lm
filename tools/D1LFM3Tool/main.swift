// Copyright © 2026 Faux Fox, Inc.
// `fmlx-d1`: converts, quantizes, evaluates, and measures Liquid d1-2B (LFM3-VL) on MLX.

import Foundation
import MLX
import MLXDecision
import MLXLMCommon

struct Arguments {
    var positional = [String]()
    var options = [String: String]()
    var flags = Set<String>()

    init(_ arguments: [String]) {
        var i = 0
        while i < arguments.count {
            let a = arguments[i]
            if a.hasPrefix("--") {
                let name = String(a.dropFirst(2))
                if i + 1 < arguments.count, !arguments[i + 1].hasPrefix("--") {
                    options[name] = arguments[i + 1]
                    i += 2
                    continue
                }
                flags.insert(name)
            } else {
                positional.append(a)
            }
            i += 1
        }
    }

    func url(_ name: String) throws -> URL {
        guard let value = options[name] else { throw ToolError("missing --\(name)") }
        return URL(fileURLWithPath: (value as NSString).expandingTildeInPath)
    }

    func int(_ name: String) -> Int? { options[name].flatMap(Int.init) }

    var weightLoading: WeightLoading { flags.contains("mapped") ? .memoryMapped : .copy }

    var dtype: DType? {
        switch options["dtype"] {
        case "float16", "fp16": .float16
        case "bfloat16", "bf16": .bfloat16
        case "float32", "fp32": .float32
        default: nil
        }
    }
}

struct ToolError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func milliseconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
}

let arguments = Arguments(Array(CommandLine.arguments.dropFirst()))
if let limit = arguments.int("cache-mb") {
    Memory.cacheLimit = limit * 1024 * 1024
}

do {
    switch arguments.positional.first {
    case "run": try runSet(arguments)
    case "convert": try convert(arguments)
    case "imatrix": try collectImportance(arguments)
    case "bench": try bench(arguments)
    case "prefix-bench": try prefixBench(arguments)
    case "sensitivity": try sensitivity(arguments)
    case "profile": try profile(arguments)
    case "hessian": try collectHessians(arguments)
    default:
        log("usage: fmlx-d1 run|convert|imatrix|bench|prefix-bench|sensitivity …")
        exit(2)
    }
} catch {
    log("error: \(error)")
    exit(1)
}
