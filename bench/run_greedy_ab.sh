#!/bin/bash
# Controlled decode comparison at the user's depth: same prompt, greedy (content fixed, so
# acceptance is comparable), old attention kernel vs new. Decode runs the small_t route in both
# arms, so any difference here is noise or host state, not the kernel swap.
cd /home/ai/bench || exit 1
mkdir -p battery10

echo "=== MAX GREEDY OLD KERNEL ==="
NINFER_VOLTA_SPLITD=0 python3 ninfer_bench.py --arm g_old --device 1 --max-context 245000 \
    --kv-capacity 245000 --prefill-chunk 2048 --long-tokens 280000 --reply-tokens 256 \
    --spec mtp --draft-tokens 3 --extra=--greedy --out battery10/g_old.json

echo "=== MAX GREEDY NEW KERNEL ==="
python3 ninfer_bench.py --arm g_new --device 1 --max-context 245000 --kv-capacity 245000 \
    --prefill-chunk 2048 --long-tokens 280000 --reply-tokens 256 --spec mtp --draft-tokens 3 \
    --extra=--greedy --out battery10/g_new.json

echo GREEDY_AB_COMPLETE
