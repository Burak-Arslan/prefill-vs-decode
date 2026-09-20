# iOS benchmark

These two files replace `ChatModel.swift` and `ContentView.swift` in the
`LLMBasic` example app of
[mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples).

`BenchmarkModel.swift` runs the three workloads. It starts each request with a
fresh nonce so no KV cache prefix can be reused, discards a warm-up run, takes
the median of three timed runs, and reads prompt and generation timings from the
framework's own `GenerateCompletionInfo` rather than from a clock around the
call.

`ContentView.swift` starts the run without waiting for a tap and disables the
idle timer, because a locked phone suspends the app in the middle of a run.

Results are printed with a `BENCH` prefix, so a console launch captures them:

```bash
xcrun devicectl device process launch --console --device <device-id> <bundle-id> \
  | grep BENCH
```
