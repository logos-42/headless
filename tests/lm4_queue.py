#!/usr/bin/env python3
"""LM4 队列驱动 — 顺序执行配置列表, 断点续跑 + 看门狗。

runs.json:
  [{"tag": "q1_d256", "args": ["--wfr-bands","13","--joint-steps","20000","--joint-only"]}, ...]

用法:
  python3 tests/lm4_queue.py results/q1_runs.json --gpu 0 --max-min 60

行为:
  - tag 已有结果且 argv 指纹一致 → SKIP
  - 单 run 超时 → 杀掉并记录 TIMEOUT, 继续下一个
  - 全部日志追加到 results/queue_gpu<N>.log
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
RESULTS = ROOT / "results"


def sig_for(extra, out):
    """与 run_lm4_wave.py 内记录的 argv_sig 保持同一构造方式。"""
    return " ".join(sorted([*extra, "--out", out]))


def already_done(tag, extra, out):
    p = RESULTS / tag / "lm4_wave_results.json"
    if not p.exists():
        return False
    try:
        cfg = json.load(open(p)).get("_config") or {}
    except Exception:
        return False
    return cfg.get("argv_sig") == sig_for(extra, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_json")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-min", type=float, default=60.0)
    args = ap.parse_args()

    runs = json.load(open(ROOT / args.runs_json)) if not os.path.isabs(args.runs_json) \
        else json.load(open(args.runs_json))
    # 日志按 runs 文件名区分, 避免同一 GPU 上多个队列互相混写
    stem = Path(args.runs_json).stem
    log = RESULTS / f"queue_{stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def say(msg):
        line = f"[{time.strftime('%m-%d_%H:%M')}] gpu{args.gpu} {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    say(f"队列启动: {len(runs)} 项, 超时 {args.max_min} min")
    for i, r in enumerate(runs, 1):
        tag, extra = r["tag"], r.get("args", [])
        out = f"results/{tag}"
        if already_done(tag, extra, out):
            say(f"[{i}/{len(runs)}] SKIP {tag} (已完成且指纹一致)")
            continue
        (ROOT / "results" / tag).mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-u", "tests/run_lm4_wave.py", *extra, "--out", out]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        say(f"[{i}/{len(runs)}] START {tag}")
        t0 = time.time()
        with open(ROOT / "results" / f"{tag}.log", "w") as lf:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                 stdout=lf, stderr=subprocess.STDOUT)
            rc = None
            while True:
                try:
                    rc = p.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    if (time.time() - t0) / 60.0 > args.max_min:
                        p.kill()
                        p.wait()
                        rc = "TIMEOUT"
                        break
        say(f"[{i}/{len(runs)}] DONE {tag} rc={rc} ({(time.time()-t0)/60:.1f} min)")
    say("队列结束")


if __name__ == "__main__":
    main()
