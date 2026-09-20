# Summary of every measurement

All numbers below were produced on real hardware for this post, on 20 September
2026. Nothing is quoted from a vendor page.

## The phone table

Same job shape on two phones, plus the accelerator comparison on one of them.

| Device | Runtime | Model | Prefill tok/s | Decode tok/s |
| --- | --- | --- | ---: | ---: |
| iPhone 15 Pro (A17 Pro) | MLX Swift | Qwen3 1.7B 4-bit | 245 | 36.6 |
| Pixel 10 Pro XL (G5) | llama.cpp CPU | Qwen3 1.7B Q4_K_M | 108 | 17.4 |
| Pixel 10 Pro XL (G5) | llama.cpp Vulkan GPU | Qwen3 1.7B Q4_K_M | 24 | 2.5 |
| Pixel 10 Pro XL (G5) | LiteRT-LM CPU | Gemma 4 E2B | 207 | 18.7 |
| Pixel 10 Pro XL (G5) | LiteRT-LM TPU | Gemma 4 E2B | 2006 | 20.0 |

Runtimes differ across rows, so read down a device, not across devices.

## What the accelerator actually buys you

Same app, same model, same settings, only the accelerator changes.

| | TPU | CPU | TPU advantage |
| --- | ---: | ---: | ---: |
| Prefill | 2005.7 tok/s | 206.7 tok/s | 9.7x |
| Decode | 20.0 tok/s | 18.7 tok/s | 1.07x |
| Time to first token | 0.18 s | 1.30 s | 7.2x |
| Session warm-up | 3045 ms | 700 ms | 4.3x slower |

The neural core multiplies matrices faster. It cannot make memory faster. Prefill
is matrix work, so it gains an order of magnitude. Decode reads every weight for
every token, so it gains nothing, and the warm-up cost is charged either way.

## The same phone job, three shapes

iPhone 15 Pro, Qwen3 1.7B 4-bit through MLX.

| Workload | Prompt | Time to first token | Decode tok/s |
| --- | ---: | ---: | ---: |
| classify, 16 output tokens | 51 tok | 196 ms | 44.6 |
| extract, 60 output tokens | 1692 tok | 6919 ms | 36.6 |
| generate, 400 output tokens | 54 tok | 573 ms | 29.6 |

A long prompt costs seconds before anything appears. A long answer costs decode
rate, and that rate falls as the answer grows.

## Desktop, for scale only

MacBook Pro M3 Max, the same GGUF files that were pushed to the Pixel.

| Runtime | Model | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: |
| llama.cpp Metal | Qwen3 1.7B Q4_K_M | 3566 | 206.9 |
| llama.cpp Metal | Qwen3 1.7B Q8_0 | 3729 | 150.3 |

Halving the weights leaves prefill alone and lifts decode by 38 percent. That is
the compute bound and the memory bound in one pair of rows.
