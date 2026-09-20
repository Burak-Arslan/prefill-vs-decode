// Prefill vs decode benchmark for iOS, built on MLX Swift.
//
// Runs the same three workload shapes we measure on the desktop and on Android,
// so the three platforms can be compared on one table:
//
//   classify  short prompt, very short output
//   extract   long prompt, short output
//   generate  short prompt, long output
//
// Method notes that matter:
//   - every prompt carries a fresh nonce, so a cached KV prefix can never be
//     reused and prefill is always really performed
//   - one warm-up run per workload is discarded
//   - three timed runs per workload, median reported
//   - prompt and generation timings come from the framework's own completion
//     info, not from our own clock around the call

import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import SwiftUI
import Tokenizers

/// Same model family and precision we measure on the other two platforms.
private let modelConfiguration = LLMRegistry.qwen3_1_7b_4bit

private let instructions = "You are a helpful assistant."

/// A paragraph repeated to build the long prompt. Deterministic on purpose.
private let filler = """
    The technician arrived at the substation at 08:14 and logged the panel \
    identifier, the meter serial number, the tariff class, and the last recorded \
    index value before opening the cabinet door.
    """

struct Workload {
    let name: String
    let prompt: String
    let maxTokens: Int
}

struct RunResult {
    let promptTokens: Int
    let generationTokens: Int
    let promptTime: Double
    let generateTime: Double

    var prefillTokensPerSecond: Double { Double(promptTokens) / promptTime }
    var decodeTokensPerSecond: Double { Double(generationTokens) / generateTime }
}

struct WorkloadResult {
    let workload: String
    let promptTokens: Int
    let generationTokens: Int
    let prefillMedian: Double
    let decodeMedian: Double
    let promptTimeMedianMs: Double
}

@MainActor @Observable public class ModelLoader {

    enum State {
        case idle
        case loading(Task<ModelContainer, Error>)
        case loaded(ModelContainer)
    }

    public var progress = 0.0
    public var isLoaded: Bool {
        switch state {
        case .idle, .loading: false
        case .loaded: true
        }
    }

    private var state = State.idle

    public func model() async throws -> ModelContainer {
        switch self.state {
        case .idle:
            let task = Task {
                try await #huggingFaceLoadModelContainer(
                    configuration: modelConfiguration
                ) { value in
                    Task { @MainActor in
                        self.progress = value.fractionCompleted
                    }
                }
            }
            self.state = .loading(task)
            let model = try await task.value
            self.state = .loaded(model)
            return model

        case .loading(let task):
            return try await task.value

        case .loaded(let model):
            return model
        }
    }
}

@MainActor @Observable public class ChatModel {

    private let model: ModelContainer

    public var log: [String] = []
    public var isBusy = false
    public var finished = false

    public init(model: ModelContainer) {
        self.model = model
    }

    private func workloads() -> [Workload] {
        let longPrompt = String(repeating: filler + " ", count: 40)
        return [
            Workload(
                name: "classify",
                prompt:
                    "Answer with one word, yes or no. Is an electricity meter an "
                    + "industrial measurement device? /no_think",
                maxTokens: 16),
            Workload(
                name: "extract",
                prompt: longPrompt
                    + "\n\nReturn JSON with the fields panel_id, meter_serial, "
                    + "tariff_class. Answer with JSON only. /no_think",
                maxTokens: 60),
            Workload(
                name: "generate",
                prompt:
                    "Explain to a junior engineer why reading a model weight from "
                    + "memory is the slow part of text generation. /no_think",
                maxTokens: 400),
        ]
    }

    private func nonce() -> String {
        let letters = "abcdefghijklmnopqrstuvwxyz0123456789"
        return String((0 ..< 12).map { _ in letters.randomElement()! })
    }

    /// One request. A new session each time so no KV cache survives between runs.
    private func runOnce(_ workload: Workload) async throws -> RunResult {
        var parameters = GenerateParameters()
        parameters.maxTokens = workload.maxTokens

        let session = ChatSession(
            model,
            instructions: instructions,
            generateParameters: parameters)

        let prompt = "Reference \(nonce()). " + workload.prompt
        var info: GenerateCompletionInfo?

        for try await item in session.streamDetails(
            to: prompt, role: .user, images: [], videos: []
        ) {
            if case .info(let i) = item {
                info = i
            }
        }

        guard let info else {
            throw NSError(
                domain: "bench", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "no completion info returned"])
        }

        return RunResult(
            promptTokens: info.promptTokenCount,
            generationTokens: info.generationTokenCount,
            promptTime: info.promptTime,
            generateTime: info.generateTime)
    }

    private func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        let middle = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    private func emit(_ line: String) {
        log.append(line)
        print("BENCH \(line)")
    }

    public func runBenchmark() {
        guard !isBusy else { return }
        isBusy = true

        Task {
            emit("device=\(await UIDevice.current.modelIdentifier) ios=\(UIDevice.current.systemVersion)")
            emit("model=\(modelConfiguration.name)")

            var results: [WorkloadResult] = []

            for workload in workloads() {
                do {
                    emit("\(workload.name): warmup")
                    _ = try await runOnce(workload)

                    var runs: [RunResult] = []
                    for index in 1 ... 3 {
                        let run = try await runOnce(workload)
                        runs.append(run)
                        emit(
                            String(
                                format: "%@ run %d: prompt %d tok in %.0f ms (%.0f tok/s), "
                                    + "gen %d tok (%.1f tok/s)",
                                workload.name, index, run.promptTokens,
                                run.promptTime * 1000, run.prefillTokensPerSecond,
                                run.generationTokens, run.decodeTokensPerSecond))
                    }

                    let result = WorkloadResult(
                        workload: workload.name,
                        promptTokens: runs[0].promptTokens,
                        generationTokens: runs[0].generationTokens,
                        prefillMedian: median(runs.map(\.prefillTokensPerSecond)),
                        decodeMedian: median(runs.map(\.decodeTokensPerSecond)),
                        promptTimeMedianMs: median(runs.map { $0.promptTime * 1000 }))
                    results.append(result)
                } catch {
                    emit("\(workload.name): failed \(error.localizedDescription)")
                }
            }

            emit("--- summary ---")
            for result in results {
                emit(
                    String(
                        format: "%@: prompt %d tok, prefill %.0f tok/s, decode %.1f tok/s, "
                            + "prompt time %.0f ms",
                        result.workload, result.promptTokens, result.prefillMedian,
                        result.decodeMedian, result.promptTimeMedianMs))
            }

            isBusy = false
            finished = true
        }
    }
}

extension UIDevice {
    /// Hardware identifier such as iPhone16,1 so the write-up names the real chip.
    var modelIdentifier: String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let mirror = Mirror(reflecting: systemInfo.machine)
        return mirror.children.reduce(into: "") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            identifier += String(UnicodeScalar(UInt8(value)))
        }
    }
}
