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

Every number here was measured on the hardware listed below. Two of them, the
TPU and CPU rows of the accelerator comparison, are reported by Google's own
benchmark screen inside AI Edge Gallery rather than timed by this code, and they
are labelled as such.

## The headline result

Same phone, same app, same model family, same benchmark settings. Two things
change together: the accelerator, and the model build the app loads for it. The
CPU build is 2.6 GB and the TPU build is 4.0 GB, so this is a comparison of two
paths, not of one file on two cores. The figures come from the app's own
benchmark screen.

| | TPU | CPU | ratio |
| --- | ---: | ---: | ---: |
| Prefill | 2005.7 tok/s | 206.7 tok/s | **9.7x** |
| Decode | 20.0 tok/s | 18.7 tok/s | **1.07x** |
| Time to first token | 0.18 s | 1.30 s | 7.2x |
| Session warm-up | 3045 ms | 700 ms | TPU 4.3x slower |

The CPU column is the mean of two runs, and those two runs differ from each
other by 5.5 percent on decode. The decode ratio therefore sits inside the noise
of the CPU column alone. The TPU column is a single run. On first init the order
reverses: TPU 3958 ms against a CPU mean of 6375 ms.

The direction fits the mechanism. A neural core multiplies matrices faster, and
prefill is matrix work. Decode pulls every weight through memory for every token,
and this core does not change the memory bandwidth of this phone. That is not a
general law. An accelerator with enough on-chip memory can speed up decode too.
It does not happen here.

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
| classify | 51 tok | 196 ms | not reported, only 5 to 6 tokens generated |
| extract | 1692 tok | 6919 ms | 36.6 tok/s |
| generate | 54 tok | 573 ms | 29.6 tok/s |

Same model and same phone. The wait for the first token moves by 35x. Prompt
length is the dominant term but not the only one: the generate row has a 54 token
prompt and still takes 573 ms, against 196 ms for a 51 token prompt. The two runs
differ in generation budget, 400 tokens against 16, and that difference is not
isolated here.

## What is in here

| Path | What it is |
| --- | --- |
| `bench.py` | Times prefill and decode separately against an LM Studio server, three workloads, median of up to three successful runs. What it calls prefill is prompt tokens divided by client side time to first token, so request and queue overhead is inside it |
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
reports `asimddp`, `i8mm` and `sve2`. At four threads prefill read 22.4 tok/s
before the fix and 92.6 tok/s after, a 4.1x difference on identical settings. See
`results/android-build-flags.txt`.

**Follow the reasoning tokens.** Qwen streams them as `reasoning_content`,
gpt-oss as `reasoning`. Watch only `content` and the first token looks late while
decode looks impossible, tens of thousands of tokens per second, which is a
buffered stream rather than a measurement. Those broken runs came from an early
version of the script and are not in `results/`. Two larger models also spent 399
of a 400 token budget thinking, leaving a single token for the answer. The
`/no_think` switch is applied only to models whose name contains qwen3, and on
qwen3.6-27b it did not take effect.

**A phone with the screen off is not a benchmark rig.** The iOS run stalled at
13 MB of a 1 GB download for twenty minutes because the app was suspended. It
finished in under a minute once it returned to the foreground.

**Decode is not one number per model.** On the same phone with the same settings,
36.6 tok/s over fifty generated tokens and 29.6 over four hundred. My first
explanation was the growing KV cache, and the data does not support it: the run
with the larger context, 1741 tokens, decoded faster than the run with 454. The
remaining candidate is thermal drift, which I did not record. So the honest claim
is narrower: decode falls as generation grows, and this data does not isolate why.

**Do not report a decode rate from a handful of tokens.** Below 24 generated
tokens the figure is noise. `bench.py` returns nothing for those runs, and the
classify row in the tables above leaves the decode cell empty for the same
reason.

## Honest limits

- Runtimes differ between devices. MLX on iOS, llama.cpp and LiteRT-LM on
  Android, llama.cpp on macOS. Read down a device, not across devices.
- All measurements are text input. Image input keeps the same mechanism but
  pushes even more weight onto prefill, since a photo encodes into hundreds of
  tokens. That is measured in a follow-up, not claimed here.
- The Android CPU figures in the accelerator table are the mean of two separate
  runs. Everything else on Android comes from llama-bench's own repetitions.
- The decode rate is computed as `(n-1)/t` in `bench.py` and as `n/t` in the iOS
  app. The difference is under one percent for long outputs and larger for short
  ones.
- No memory bandwidth figure is quoted anywhere here, so the "bandwidth divided
  by bytes read per token" rule cannot be checked against these numbers directly.
- `results/desktop-lmstudio-mlx.json` is an exploratory sweep across six models.
  None of the headline numbers come from it.
- The Vulkan GPU result is specific to this device. The driver reports no integer
  dot product support and no matrix cores, which is what the 4-bit kernels need.

## License

MIT for the tooling. The measurement data is free to reuse with attribution.

## Versions

| Piece | Version |
| --- | --- |
| llama.cpp (macOS build) | b3cf0325 |
| llama.cpp (Android build) | same checkout, cross compiled with NDK 27.1.12297006 |
| Android CPU flags | `-DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+i8mm"` |
| AI Edge Gallery | 1.0.19, Tensor G5 build |
| LM Studio server API | v0 and v1 endpoints on port 1234 |
| iOS model | `mlx-community/Qwen3-1.7B-4bit` through MLX Swift |
| Android and macOS model files | `Qwen3-1.7B-Q4_K_M.gguf` (1.03 GiB) and `Qwen3-1.7B-Q8_0.gguf` (1.70 GiB), unsloth GGUF builds |

Exact file hashes are not recorded, which is a gap. If you rerun this and get
different numbers, that is the first thing to check.
