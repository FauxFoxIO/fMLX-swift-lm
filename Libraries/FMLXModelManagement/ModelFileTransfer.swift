// Copyright © 2026 Faux Fox.

import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

final class ModelFileTransfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let destination: URL
    private let expectedBytes: Int64
    private let sessionConfiguration: URLSessionConfiguration
    private let progress: @Sendable (Int64) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Int64, Error>?
    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var handle: FileHandle?
    private var receivedBytes: Int64
    private var completed = false
    private var responseAccepted = false

    init(
        destination: URL, existingBytes: Int64, expectedBytes: Int64,
        sessionConfiguration: URLSessionConfiguration = .ephemeral,
        progress: @escaping @Sendable (Int64) -> Void
    ) {
        self.destination = destination
        self.receivedBytes = existingBytes
        self.expectedBytes = expectedBytes
        self.sessionConfiguration = sessionConfiguration
        self.progress = progress
    }

    func start(url: URL) async throws -> Int64 {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                guard !completed else {
                    lock.unlock()
                    continuation.resume(throwing: CancellationError())
                    return
                }
                self.continuation = continuation
                do {
                    if !FileManager.default.fileExists(atPath: destination.path) {
                        FileManager.default.createFile(atPath: destination.path, contents: nil)
                    }
                    let handle = try FileHandle(forWritingTo: destination)
                    try handle.seekToEnd()
                    self.handle = handle
                    var request = URLRequest(url: url)
                    request.timeoutInterval = 30 * 60
                    request.setValue("fMLX-swift-lm", forHTTPHeaderField: "User-Agent")
                    if receivedBytes > 0 {
                        request.setValue("bytes=\(receivedBytes)-", forHTTPHeaderField: "Range")
                    }
                    let session = URLSession(
                        configuration: sessionConfiguration, delegate: self, delegateQueue: nil)
                    self.session = session
                    let task = session.dataTask(with: request)
                    self.task = task
                    lock.unlock()
                    task.resume()
                } catch {
                    lock.unlock()
                    finish(.failure(error))
                }
            }
        } onCancel: {
            cancel()
        }
    }

    func cancel() {
        finish(.failure(CancellationError()))
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        guard request.url?.scheme?.lowercased() == "https" else {
            completionHandler(nil)
            finish(.failure(FMLXModelManagementError.invalidRepositoryResponse))
            return
        }
        completionHandler(request)
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let response = response as? HTTPURLResponse,
            (200 ..< 300).contains(response.statusCode)
        else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            completionHandler(.cancel)
            finish(.failure(FMLXModelManagementError.downloadFailed(statusCode: status)))
            return
        }
        lock.lock()
        do {
            if receivedBytes > 0 && response.statusCode == 200 {
                try handle?.truncate(atOffset: 0)
                try handle?.seek(toOffset: 0)
                receivedBytes = 0
            } else if receivedBytes > 0 && response.statusCode != 206 {
                throw FMLXModelManagementError.invalidRepositoryResponse
            }
            responseAccepted = true
            lock.unlock()
            completionHandler(.allow)
        } catch {
            lock.unlock()
            completionHandler(.cancel)
            finish(.failure(error))
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        guard !completed, responseAccepted, let handle else {
            lock.unlock()
            return
        }
        do {
            try handle.write(contentsOf: data)
            receivedBytes += Int64(data.count)
            let receivedBytes = receivedBytes
            let oversized = expectedBytes > 0 && receivedBytes > expectedBytes
            lock.unlock()
            progress(receivedBytes)
            if oversized {
                finish(
                    .failure(
                        FMLXModelManagementError.sizeMismatch(
                            file: destination.lastPathComponent,
                            expected: expectedBytes, actual: receivedBytes)))
            }
        } catch {
            lock.unlock()
            finish(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
    ) {
        if let error {
            finish(.failure(error))
        } else {
            lock.lock()
            let bytes = receivedBytes
            lock.unlock()
            finish(.success(bytes))
        }
    }

    private func finish(_ result: Result<Int64, Error>) {
        let continuation: CheckedContinuation<Int64, Error>?
        let session: URLSession?
        let task: URLSessionDataTask?
        let handle: FileHandle?
        lock.lock()
        guard !completed else {
            lock.unlock()
            return
        }
        completed = true
        continuation = self.continuation
        self.continuation = nil
        session = self.session
        self.session = nil
        task = self.task
        self.task = nil
        handle = self.handle
        self.handle = nil
        lock.unlock()
        task?.cancel()
        try? handle?.close()
        session?.invalidateAndCancel()
        continuation?.resume(with: result)
    }
}
