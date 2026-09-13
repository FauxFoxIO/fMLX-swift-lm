// Copyright © 2026 Faux Fox.

import XCTest

@testable import FMLXText

final class NativeTextPrefillChunkPolicyTests: XCTestCase {
    func testOnlyResidentQwen35MoEChangesTheConfiguredChunkSize() {
        XCTAssertEqual(
            resolvedPrefillChunkSize(
                configured: 96, policy: .configured, isResidentQwen35MoE: true), 96)
        XCTAssertEqual(
            resolvedPrefillChunkSize(
                configured: 96, policy: .balanced, isResidentQwen35MoE: true), 256)
        XCTAssertEqual(
            resolvedPrefillChunkSize(
                configured: 96, policy: .throughput, isResidentQwen35MoE: true), 512)
        XCTAssertEqual(
            resolvedPrefillChunkSize(
                configured: 1024, policy: .balanced, isResidentQwen35MoE: true), 1024)
        XCTAssertEqual(
            resolvedPrefillChunkSize(
                configured: 1024, policy: .throughput, isResidentQwen35MoE: true), 1024)

        for policy in [
            NativeTextPrefillChunkPolicy.configured, .balanced, .throughput,
        ] {
            XCTAssertEqual(
                resolvedPrefillChunkSize(
                    configured: 96, policy: policy, isResidentQwen35MoE: false), 96)
        }
    }
}
