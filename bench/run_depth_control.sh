#!/bin/bash
# Depth control: same synthetic content at 85k, new kernel then old kernel, each with a
# cold pass and an identical repeat (cache hit) so the cache question is settled directly.
set -u
LOG=/home/ai/ninfer-serve.log
BIN=/home/ai/ninfer-v100/build-v100/apps/ninfer-serve
NEWBAK=/home/ai/bench/ninfer-serve-splitd-20260928
OLDBAK=/home/ai/bench/ninfer-serve-base-20260928
MODEL=/home/ai/models/qwen3_8_27b_nvfp4.ninfer
cd /home/ai/bench || exit 1

run_engine () {
  pkill -x ninfer-serve 2>/dev/null; sleep 5; : > $LOG
  nohup stdbuf -oL -eL $BIN $MODEL --host 127.0.0.1 --port 8110 --model-id qwen3.8-27b-uncen \
    --max-context 245000 --kv-capacity 245000 --prefill-chunk 2048 --max-concurrency 1 \
    --kv-dtype int8 --device-state-slots 1 --host-state-slots 2 --host-kv-mib 1024 \
    --spec mtp --draft-tokens 3 --lm-head-draft --device 1 > $LOG 2>&1 &
  for i in $(seq 1 40); do grep -aq 'listening on' $LOG && return 0; sleep 5; done
  return 1
}

echo "=== NEW kernel @85k $(date +%H:%M:%S) ==="
cp -p $NEWBAK $BIN
run_engine
python3 probe_live.py --label new85k --long-tokens 98000 --reply-tokens 256 --repeat \
  --out /home/ai/bench/probe_new85k.json --salt probe-new85k 2>&1

echo "=== OLD kernel @85k $(date +%H:%M:%S) ==="
cp -p $OLDBAK $BIN
run_engine
python3 probe_live.py --label old85k --long-tokens 98000 --reply-tokens 256 --repeat \
  --out /home/ai/bench/probe_old85k.json --salt probe-old85k 2>&1

echo "=== restore NEW kernel as production $(date +%H:%M:%S) ==="
cp -p $NEWBAK $BIN
run_engine
grep -a 'capacity' $LOG | tail -1
md5sum $BIN
echo DEPTH_CONTROL_DONE
