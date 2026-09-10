#!/usr/bin/env python3
"""LM4 — 下载 RBSP-A EMFISIS MAG L3, 提取独立电磁波特征 (本机并行)

MAG L3 (1秒) 参数: Mag[3], Magnitude[nT], delta[deg], lambda[deg], rms[nT], coordinates[km]
取独立特征 (与 density 无解析关系):
  Magnitude (背景磁场), rms (磁场波动 = 波活动), lambda (纬度), delta (经度)
降采样 6s 对齐 DENSITY L4。

用法: python3 dl_mag.py --start 2015-01 --months 3 --workers 4
输出: data/wave/mag_YYYYMM.npz  (time, feats=[Magnitude, rms, lambda, delta])
"""
import os, json, time, argparse, urllib.request
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

HAPI = "https://cdaweb.gsfc.nasa.gov/hapi/data"
DS = "RBSP-A_MAGNETOMETER_1SEC-GEI_EMFISIS-L3"
BASE = os.path.dirname(os.path.abspath(__file__))
COLS = [2, 5, 3, 4]   # Magnitude, rms, delta, lambda


def fetch_day(day):
    d2 = day + timedelta(days=1)
    url = (f"{HAPI}?id={DS}&time.min={day.strftime('%Y-%m-%dT%H:%M:%SZ')}"
           f"&time.max={d2.strftime('%Y-%m-%dT%H:%M:%SZ')}&format=json")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=150) as r:
                d = json.load(r)
            code = d.get("status", {}).get("code")
            if code == 1201:
                return day, []
            if code != 1200:
                raise RuntimeError(f"status {code}")
            # 降采样 6s
            rows = d["data"][::6]
            out = []
            for r_ in rows:
                try:
                    feats = [float(r_[c]) if r_[c] is not None else float("nan") for c in COLS]
                except (TypeError, ValueError):
                    continue
                if any(v < -1e30 for v in feats):
                    continue
                out.append((r_[0], feats))
            return day, out
        except Exception as e:
            if attempt == 2:
                print(f"  [warn] {day.date()}: {e}", flush=True)
                return day, []
            time.sleep(2 * (attempt + 1))
    return day, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01")
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    y, m = map(int, args.start.split("-"))
    out_dir = os.path.join(BASE, "..", "data", "wave")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(args.months):
        yy, mm = y + (m - 1 + i) // 12, (m - 1 + i) % 12 + 1
        d0 = datetime(yy, mm, 1)
        d1 = datetime(yy + (mm == 12), (mm % 12) + 1, 1)
        days = [d0 + timedelta(days=k) for k in range((d1 - d0).days)]
        print(f"=== MAG {yy}-{mm:02d} ({len(days)}d, {args.workers}w) ===", flush=True)
        t0 = time.time()
        recs = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch_day, d) for d in days]
            done = 0
            for f in as_completed(futs):
                _, rows = f.result()
                recs.extend(rows)
                done += 1
                if done % 10 == 0 or done == len(days):
                    print(f"  {done}/{len(days)}d, {len(recs)}pts, {time.time()-t0:.0f}s", flush=True)
        if not recs:
            print(f"  [skip] {yy}-{mm:02d}", flush=True)
            continue
        recs.sort(key=lambda x: x[0])
        out = os.path.join(out_dir, f"mag_{yy}{mm:02d}.npz")
        np.savez_compressed(out, time=np.array([r[0] for r in recs]),
                            feats=np.array([r[1] for r in recs], dtype=np.float32))
        print(f"[save] {os.path.basename(out)} ({os.path.getsize(out)/1024:.0f} KB, {len(recs)} pts, {time.time()-t0:.0f}s)", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
