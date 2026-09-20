# Prefill vs decode, measured on real phones

A language model runs in two phases, and they behave nothing alike. This repo
holds the tools and the raw numbers behind the post
[Tokens per second is two numbers](https://baarslann.dev).

- **Prefill** reads the prompt. Every prompt token goes through the model in one
  pass, which makes it matrix times matrix work and compute bound. It decides how
  long the user waits for the first token.
- **Decode** writes the answer, one token at a time. Each step pulls every model
  weight through memory, which makes it matrix times vector work and memory
  bandwidth bound. It decides how fast the answer flows.

Every number here was measured on the hardware listed below on 20 September 2026.
Nothing is copied from a vendor page.

## The headline result

Same phone, same app, same model, same settings. Only the accelerator changes.

| | TPU | CPU | ratio |
| --- | ---: | ---: | ---: |
| Prefill | 2005.7 tok/s | 206.7 tok/s | **9.7x** |
| Decode | 20.0 tok/s | 18.7 tok/s | **1.07x** |
| Time to first token | 0.18 s | 1.30 s | 7.2x |
| Session warm-up | 3045 ms | 700 ms | 4.3x slower |

The neural core multiplies matrices faster. It cannot make memory faster. So the
phase that is matrix work gains an order of magnitude, the phase that is memory
traffic gains nothing, and the warm-up is charged either way.

## Hardware

| Device | Chip | Memory | OS |
| --- | --- | --- | --- |
| iPhone 15 Pro | A17 Pro | 8 GB | iOS 26.6.2 |
| Pixel 10 Pro XL | Tensor G5, PowerVR D-Series GPU | 16 GB | Android 17 |
| MacBook Pro | Apple M3 Max, 16 CPU / 40 GPU cores | 48 GB | macOS 26.4.1 |

## Workload shapes

Chat is not the only job a small model does on a phone, and the three common
shapes stress different phases.

| Workload | Prompt | Output | Phase that decides the wait |
| --- | --- | --- | --- |
| classify | short | very short | prefill plus fixed overhead |
| extract | long | short | prefill |
| generate | short | long | decode |

Measured on an iPhone 15 Pro with Qwen3 1.7B 4-bit through MLX:

| Workload | Prompt | Time to first token | Decode |
| --- | ---: | ---: | ---: |
| classify | 51 tok | 196 ms | 44.6 tok/s |
| extract | 1692 tok | 6919 ms | 36.6 tok/s |
| generate | 54 tok | 573 ms | 29.6 tok/s |

Same model, same phone, same settings. The wait for the first token moves by 35x
because the prompt got longer.

## What is in here

| Path | What it is |
| --- | --- |
| `bench.py` | Times prefill and decode separately against an LM Studio server, three workloads, median of three runs |
| `cache_check.py` | Demonstrates the repeated-prompt trap |
| `android_sweep.sh` | Runs `llama-bench` on a connected phone with cooldowns between runs |
| `ios/` | The MLX Swift benchmark that produced the iPhone numbers |
| `results/` | Raw output of every run, plus `SUMMARY.md` |

### Running the desktop benchmark

Start an LM Studio server on port 1234, then:

```bash
python3 bench.py --runs 3 --out results.json
python3 bench.py --models qwen/qwen3-1.7b --runs 1   # single model
```

### Running the Android benchmark

Build `llama-bench` for arm64 and push it with a GGUF model. The build flags
matter more than anything else here:

```bash
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF \
  -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+i8mm"
cmake --build build-android --target llama-bench -j 8
bash android_sweep.sh
```

### Running the iOS benchmark

The two files in `ios/` replace `ChatModel.swift` and `ContentView.swift` in the
`LLMBasic` example app of
[mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples). The app
downloads the model on first launch, then runs the three workloads and prints
each result. Build it to a real device:

```bash
xcodebuild -scheme LLMBasic -destination "platform=iOS,id=<device-id>" \
  -allowProvisioningUpdates -skipMacroValidation \
  DEVELOPMENT_TEAM=<team> PRODUCT_BUNDLE_IDENTIFIER=<bundle-id> build
xcrun devicectl device install app --device <device-id> <path>/LLMBasic.app
xcrun devicectl device process launch --console --device <device-id> <bundle-id>
```

## Method rules that changed the numbers

These are not style preferences. Each one moved a result by a factor, and each
was found the hard way while producing this data.

**Never repeat a benchmark prompt.** A server that keeps the KV cache skips
prefill entirely on the second run. Same prompt: 98 ms to first token. Unique
prefix: 599 ms. That is a 6.1x overstatement. Every request here starts with a
fresh nonce. See `results/prompt-cache-trap.txt`.

**Check what your build actually compiled.** The first Android binary fell back
to a baseline armv8-a kernel because a cmake feature test failed, while the CPU
reports `asimddp`, `i8mm` and `sve2`. Prefill read 22 tok/s. With the right
`-DGGML_CPU_ARM_ARCH` flag the same device, model and prompt read 107 tok/s. See
`results/android-build-flags.txt`.

**Follow the reasoning tokens.** Qwen streams them as `reasoning_content`,
gpt-oss as `reasoning`. Watch only `content` and the first token looks late while
decode looks impossible. One run reported 97,671 tok/s, which is a buffered
stream, not a measurement. Two larger models also spent 399 of a 400 token budget
thinking, leaving a single token for the answer.

**A phone with the screen off is not a benchmark rig.** The iOS run stalled at
13 MB of a 1 GB download for twenty minutes because the app was suspended. It
finished in under a minute once it returned to the foreground.

**Decode is not one number per model.** On the same phone with the same settings:
44.6 tok/s over six generated tokens, 36.6 over fifty, 29.6 over four hundred.
The KV cache grows and the device warms up.

**Do not report a decode rate from a handful of tokens.** Below roughly two dozen
generated tokens the figure is noise. Those cells are left blank here.

## Honest limits

- Runtimes differ between devices. MLX on iOS, llama.cpp and LiteRT-LM on
  Android, llama.cpp on macOS. Read down a device, not across devices.
- All measurements are text input. Image input keeps the same mechanism but
  pushes even more weight onto prefill, since a photo encodes into hundreds of
  tokens. That is measured in a follow-up, not claimed here.
- The Android CPU figures are the mean of two separate runs of the same
  configuration.
- The Vulkan GPU result is specific to this device. The driver reports no integer
  dot product support and no matrix cores, which is what the 4-bit kernels need.

## License

MIT for the tooling. The measurement data is free to reuse with attribution.
