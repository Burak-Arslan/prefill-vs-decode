# Summary of every measurement

All numbers below were produced on real hardware for this post. The TPU and CPU
rows of the accelerator table are reported by Google's AI Edge Gallery benchmark
screen, not timed by this code.

## The phone table

Same job shape on two phones, plus the accelerator comparison on one of them.

| Device | Runtime | Model | Prefill tok/s | Decode tok/s |
| --- | --- | --- | ---: | ---: |
| iPhone 15 Pro (A17 Pro) | MLX Swift, extract workload | Qwen3 1.7B 4-bit | 245 | 36.6 |
| Pixel 10 Pro XL (G5) | llama.cpp CPU, 4 threads | Qwen3 1.7B Q4_K_M | 92.6 | 17.4 |
| Pixel 10 Pro XL (G5) | llama.cpp CPU, 6 threads | Qwen3 1.7B Q4_K_M | 107.6 | 15.7 |
| Pixel 10 Pro XL (G5) | llama.cpp Vulkan GPU | Qwen3 1.7B Q4_K_M | 24 | 2.5 |
| Pixel 10 Pro XL (G5) | LiteRT-LM CPU | Gemma 4 E2B | 207 | 18.7 |
| Pixel 10 Pro XL (G5) | LiteRT-LM TPU | Gemma 4 E2B | 2006 | 20.0 |

Runtimes differ across rows, so read down a device, not across devices.

## What the accelerator actually buys you

Same app, same model, same settings, only the accelerator changes.

| | TPU | CPU | ratio |
| --- | ---: | ---: | ---: |
| Prefill | 2005.7 tok/s | 206.7 tok/s | 9.7x |
| Decode | 20.0 tok/s | 18.7 tok/s | 1.07x |
| Time to first token | 0.18 s | 1.30 s | 7.2x |
| Session warm-up | 3045 ms | 700 ms | TPU 4.3x slower |

The two accelerators load different model builds, 2.6 GB for CPU and 4.0 GB for
TPU, so the accelerator is not the only variable. The CPU column is the mean of
two runs which differ by 5.5 percent on decode, so the 1.07x decode ratio is
inside that noise.

## The same phone job, three shapes

iPhone 15 Pro, Qwen3 1.7B 4-bit through MLX.

| Workload | Prompt | Time to first token | Decode tok/s |
| --- | ---: | ---: | ---: |
| classify, 5 to 6 output tokens | 51 tok | 196 ms | too few tokens to report |
| extract, 49 to 55 output tokens | 1692 tok | 6919 ms | 36.6 |
| generate, 392 to 400 output tokens | 54 tok | 573 ms | 29.6 |

A long prompt costs seconds before anything appears. A long answer costs decode
rate, and that rate falls as the answer grows. The classify and generate rows
have nearly the same prompt length and very different first token times, and this
data does not explain that difference.

## Desktop, for scale only

MacBook Pro M3 Max, the same GGUF files that were pushed to the Pixel.

| Runtime | Model | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: |
| llama.cpp Metal | Qwen3 1.7B Q4_K_M | 3566 | 206.9 |
| llama.cpp Metal | Qwen3 1.7B Q8_0 | 3729 | 150.3 |

Cutting the weight file from 1.70 to 1.03 GiB, a 39 percent reduction, lifts
decode by 38 percent and moves prefill by 4.6 percent in the other direction. A
pure bandwidth model would have predicted a 65 percent decode gain, so the rule
gives the direction and not the magnitude.
