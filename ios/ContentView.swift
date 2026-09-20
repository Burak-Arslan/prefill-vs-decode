// Benchmark UI. One button, one log. Nothing else on screen so the device is
// not doing extra work while measuring.

import MLXLMCommon
import SwiftUI

struct ContentView: View {

    let loader: ModelLoader

    @State var bench: ChatModel?
    @State var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Prefill vs decode")
                .font(.headline)

            if let error {
                Text("Error: \(error)")
                    .foregroundStyle(.red)
            } else if !loader.isLoaded {
                ProgressView("Downloading model", value: loader.progress, total: 1)
            } else if let bench {
                if !bench.isBusy && !bench.finished {
                    Button("Run benchmark") {
                        bench.runBenchmark()
                    }
                    .buttonStyle(.borderedProminent)
                }

                if bench.isBusy {
                    HStack(spacing: 8) {
                        ProgressView()
                        Text("running")
                    }
                }

                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(bench.log.enumerated().map { $0 }, id: \.offset) { _, line in
                            Text(line)
                                .font(.system(size: 11, design: .monospaced))
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            Spacer()
        }
        .padding()
        .task {
            // Keep the screen awake: a locked device suspends the app mid-run.
            UIApplication.shared.isIdleTimerDisabled = true
            do {
                let model = try await loader.model()
                let bench = ChatModel(model: model)
                self.bench = bench
                // Start without waiting for a tap so the run can be driven remotely.
                bench.runBenchmark()
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}
