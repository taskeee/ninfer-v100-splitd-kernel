#!/usr/bin/env python3
"""NInfer V100 measurement harness.

One process per arm: start engine -> readiness -> requests -> report.
Readiness uses HTTP /v1/models (TCP connect is not a valid signal: the port
accepts while weights are still loading).
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOME = os.path.expanduser("~")
MODEL = os.path.join(HOME, "models", "qwen3_8_27b_nvfp4.ninfer")
BIN = os.path.join(HOME, "ninfer-v100", "build-v100", "apps", "ninfer-serve")
PORT = 8110
MODEL_ID = "qwen3.8-27b-uncen"


def log(msg):
    print("[bench] " + msg, flush=True)


def free_port(port):
    """Kill any previous engine; pgrep -f would match this wrapper, so use -x."""
    subprocess.run(
        "for p in $(ps -eo pid=,comm= | awk '$2 ~ /^ninfer/ {print $1}'); do "
        "s=$(ps -o stat= -p $p 2>/dev/null); case \"$s\" in Z*) ;; *) kill $p 2>/dev/null ;; esac; done",
        shell=True, executable="/bin/bash", capture_output=True)
    for _ in range(60):
        listening = subprocess.run(
            "ss -ltn 2>/dev/null | grep -q ':%d ' && echo yes || echo no" % port,
            shell=True, executable="/bin/bash", capture_output=True, text=True).stdout.strip()
        if listening == "no":
            return True
        time.sleep(1)
    return False


def http_json(path, payload=None, timeout=3600):
    url = "http://127.0.0.1:%d%s" % (PORT, path)
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_ready(deadline_s):
    """HTTP readiness plus a real generation probe: an empty /v1/models reply
    while weights load looks identical to a broken server."""
    started = time.time()
    while time.time() - started < deadline_s:
        try:
            body = http_json("/v1/models", timeout=10)
            if body.get("data"):
                return time.time() - started
        except Exception:
            pass
        time.sleep(3)
    return None


def wait_generate(deadline_s):
    """A 1-token request is the only proof the engine can actually generate."""
    started = time.time()
    while time.time() - started < deadline_s:
        try:
            out = chat([{"role": "user", "content": "hi"}], 1, timeout=120)
            if out and out.get("choices"):
                return time.time() - started
        except Exception:
            pass
        time.sleep(5)
    return None


def chat(messages, max_tokens, timeout=3600, salt=None, stream=False):
    payload = {"model": MODEL_ID, "messages": messages,
               "max_tokens": max_tokens, "stream": stream}
    if salt is not None:
        payload["cache_salt"] = salt
    return http_json("/v1/chat/completions", payload, timeout=timeout)


def make_prompt(tokens_target):
    """Deterministic long prose prompt whose sentences are all distinct.

    Repetitive filler is an invalid prefill yardstick: a repeated prefix makes
    the attention work cheaper than real text and inflates the number."""
    topics = [
        ("register allocation", "spills", "the occupancy of the fused kernel"),
        ("shared memory staging", "bank conflicts", "the pipeline depth"),
        ("tensor core fragment order", "swizzling", "the operand feed rate"),
        ("graph capture", "replay", "the launch overhead per round"),
        ("KV page layout", "fragmentation", "the bytes read per token"),
        ("draft window length", "acceptance", "the emitted tokens per round"),
        ("quantised weight layout", "dequantisation", "the arithmetic intensity"),
        ("prefill chunk size", "overlap", "the time to the first token"),
    ]
    sentences = []
    # ~3 characters per token for this template, so allow 4x for safety and
    # trim: capping the character budget at the target made long prompts come
    # out at a quarter of the requested depth.
    for i in range(tokens_target * 2):
        subject, detail, effect = topics[i % len(topics)]
        sentences.append(
            "Entry %05d examines how %s interacts with %s when the workload "
            "changes, and records the resulting effect on %s for the %d-th "
            "measurement in this sequence." % (i, subject, detail, effect, i))
    text = " ".join(sentences)
    return text[: tokens_target * 4]


def make_lookup_prompt(filler_tokens, print_tokens):
    """Context-lookup eligibility test: a distinctive block the model is asked
    to reproduce verbatim.

    A short ask cannot expose it: the learned MTP window is only `--draft-tokens`.
    Emitted tokens per engine round is `completion_tokens / (draft_n - accepted)`,
    because the accepted drafts and the verified target token share one round."""
    numbered = " ".join("token_%04d" % i for i in range(print_tokens))
    head = ("Reproduce the following block exactly as written, character for "
            "character, with no commentary before or after it:\n\n")
    tail = "\n\nBegin the reproduction now."
    return head + make_prompt(filler_tokens) + "\n\n" + numbered + tail


def one_measure(label, prompt_text, max_tokens, salt, timeout=3600):
    t0 = time.time()
    try:
        resp = chat([{"role": "user", "content": prompt_text}], max_tokens,
                    timeout=timeout, salt=salt)
    except urllib.error.HTTPError as exc:
        return {"label": label, "error": "HTTP %s: %s" % (exc.code, exc.read()[:300])}
    except Exception as exc:
        return {"label": label, "error": "%s: %s" % (type(exc).__name__, exc)}
    wall = time.time() - t0
    t = resp.get("timings", {}) or {}
    usage = resp.get("usage", {}) or {}
    completion = usage.get("completion_tokens")
    draft_n = t.get("draft_n")
    return {
        "label": label,
        "wall_s": round(wall, 3),
        "cache_n": t.get("cache_n"),
        "prompt_n": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_tok_s": t.get("prompt_per_second"),
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "decode_tok_s": t.get("predicted_per_second"),
        "draft_n": draft_n,
        "draft_accepted": t.get("draft_n_accepted"),
        "accept_rate": (round(t["draft_n_accepted"] / draft_n, 4)
                        if draft_n else None),
        # Emitted tokens per draft token. The engine does not document whether
        # draft_n counts rounds or draft positions, so this is only a within-arm
        # denominator; the decode tok/s and accept_rate carry the conclusion.
        "emit_per_draft": (round(completion / draft_n, 3)
                           if (completion and draft_n) else None),
        "completion_tokens": completion,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }


def gpu_mem():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True).stdout
        return [ln.strip() for ln in out.strip().splitlines()]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm label for the report")
    ap.add_argument("--max-context", type=int, default=131072)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--draft-tokens", type=int, default=3)
    ap.add_argument("--spec", default="mtp")
    ap.add_argument("--kv-capacity", default="auto")
    ap.add_argument("--device", type=int, default=1,
                    help="WSL enumerates the 3080Ti as 0 and the V100 as 1; "
                         "without this the engine refuses Volta at startup")
    ap.add_argument("--seed", type=int, default=1234,
                    help="fixed sampler seed: without it draft acceptance drifts "
                         "0.36-0.75 between runs and decode numbers are not comparable")
    ap.add_argument("--extra", default="", help="extra engine args, space separated")
    ap.add_argument("--reply-tokens", type=int, default=192)
    ap.add_argument("--long-tokens", type=int, default=16000,
                    help="approximate prompt size for the long-context test")
    ap.add_argument("--lookup-filler-tokens", type=int, default=2000,
                    help="filler size for the verbatim-reproduction arm")
    ap.add_argument("--lookup-print-tokens", type=int, default=400,
                    help="how many tokens the model is asked to reproduce")
    ap.add_argument("--salt", default=None)
    ap.add_argument("--startup-deadline", type=int, default=420)
    ap.add_argument("--out", default=None)

    # `--extra --vision` reads as an option to argparse, so unknown engine flags
    # are folded into extra rather than rejected.
    args, unknown = ap.parse_known_args()
    if unknown:
        args.extra = (args.extra + " " + " ".join(unknown)).strip()

    salt = args.salt or ("arm-%s-%d" % (args.arm, int(time.time())))
    report = {"arm": args.arm, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": {k: v for k, v in vars(args).items() if k != "extra"},
              "extra": args.extra, "results": []}

    if not free_port(PORT):
        report["fatal"] = "port %d still busy after cleanup" % PORT
        print(json.dumps(report, ensure_ascii=False)); return 1

    engine_args = [BIN, MODEL,
                   "--host", "127.0.0.1", "--port", str(PORT),
                   "--model-id", MODEL_ID,
                   "--device", str(args.device),
                   "--max-context", str(args.max_context),
                   "--kv-capacity", str(args.kv_capacity),
                   "--prefill-chunk", str(args.prefill_chunk),
                   "--max-concurrency", "1",
                   "--kv-dtype", args.kv_dtype,
                   "--device-state-slots", "1",
                   "--host-state-slots", "2",
                   "--host-kv-mib", "1024",
                   "--seed", str(args.seed),
                   "--log-stats-interval-ms", "0"]
    # The engine only accepts --spec mtp|dflash|dflash2: `--spec none` (or an empty value) is how
    # the no-speculation arm is expressed, since --spec none is rejected at startup and the bench
    # would then merely time out.
    if args.spec and args.spec != "none":
        engine_args += ["--spec", args.spec, "--draft-tokens", str(args.draft_tokens),
                        "--lm-head-draft"]
    if args.extra:
        engine_args += args.extra.split()

    logfile = os.path.join(HOME, "bench-%s.log" % args.arm)
    logf = open(logfile, "w")
    proc = subprocess.Popen(["stdbuf", "-oL", "-eL"] + engine_args,
                            stdout=logf, stderr=subprocess.STDOUT, cwd=HOME)
    report["engine_log"] = logfile
    report["engine_pid"] = proc.pid
    log("arm=%s pid=%d chunk=%d kv=%s ctx=%d" %
        (args.arm, proc.pid, args.prefill_chunk, args.kv_dtype, args.max_context))

    try:
        t_ready = wait_ready(args.startup_deadline)
        if t_ready is None:
            report["fatal"] = "readiness timeout (%ds)" % args.startup_deadline
            report["gpu"] = gpu_mem()
            return finish(report, args, logf, proc)
        t_gen = wait_generate(180)
        if t_gen is None:
            report["fatal"] = "generation probe failed"
            report["gpu"] = gpu_mem()
            return finish(report, args, logf, proc)
        report["ready_s"] = round(t_ready + t_gen, 1)
        report["gpu_after_load"] = gpu_mem()

        # 1) short prompt: decode baseline at shallow context
        report["results"].append(one_measure(
            "short-prompt", "Explain kernel launch geometry in one paragraph.",
            args.reply_tokens, salt + "-short"))

        # 2) long cold prefill: the TTFT / prefill arm
        long_prompt = make_prompt(args.long_tokens)
        report["long_prompt_chars"] = len(long_prompt)
        report["results"].append(one_measure(
            "long-cold", long_prompt, args.reply_tokens, salt + "-long"))
        # 3) same prompt again: identical prefix => cache hit expected
        report["results"].append(one_measure(
            "long-warm", long_prompt, args.reply_tokens, salt + "-long"))

        # 4) verbatim reproduction: the only shape that can expose context lookup
        lookup_prompt = make_lookup_prompt(args.lookup_filler_tokens,
                                           args.lookup_print_tokens)
        report["lookup_prompt_chars"] = len(lookup_prompt)
        report["results"].append(one_measure(
            "lookup-copy", lookup_prompt, args.reply_tokens, salt + "-lookup"))

        report["gpu_end"] = gpu_mem()
        return finish(report, args, logf, proc)
    finally:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def finish(report, args, logf, proc):
    logf.flush()
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
