#!/bin/bash
# Pixel 10 Pro XL, llama.cpp CPU. Cooldown between runs so thermal drift does not
# leak into the numbers. Big cores only (cpu2-7, mask fc).
ADB=~/Library/Android/sdk/platform-tools/adb
OUT=~/Documents/BLOG/bench/android_results.txt
: > "$OUT"
for model in qwen3-1.7b-q4_k_m.gguf qwen3-1.7b-q8_0.gguf; do
  for t in 2 4 6; do
    echo "### $model threads=$t" | tee -a "$OUT"
    $ADB shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp taskset fc ./llama-bench -m $model -p 512 -n 128 -r 3 -t $t 2>/dev/null" \
      | grep -E "pp512|tg128" | tee -a "$OUT"
    sleep 45
  done
done
echo "bitti" | tee -a "$OUT"
