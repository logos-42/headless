#!/usr/bin/env python3
"""LM4 电磁波持续学习 — 本机并行下载 (CDAWeb 从本机快 17x)

数据: RBSP-A_DENSITY_EMFISIS-L4 (Van Allen Probes, 6秒采样)
     含 density [cm^-3] / bmag [nT] / fce / fpe / wpe_over_wce / fuh
用法: python3 dl_local.py --start 2015-01 --months 6 --workers 6
输出: data/wave/rbsp_a_YYYYMM.npz
"""
import os, json, time, argparse, urllib.request, gzip
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

HAPI = "https://cdaweb.gsfc.nasa.gov/hapi/data"
DS = "RBSP-A_DENSITY_EMFISIS-L4"
BASE = os.path.dirname(os.path.abspath(__file__))


def fetch_day(day):
    d2 = day + timedelta(days=1)
    t0 = day.strftime("%Y-%m-%dT%H:%M:%SZ")
    t1 = d2.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{HAPI}?id={DS}&time.min={t0}&time.max={t1}&format=json"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                d = json.load(r)
            code = d.get("status", {}).get("code")
            if code == 1201:
                return day, []
            if code != 1200:
                raise RuntimeError(f"status {code}")
            return day, d["data"]
        except Exception as e:
            if attempt == 2:
                print(f"  [warn] {t0[:10]}: {e}", flush=True)
                return day, []
            time.sleep(2 * (attempt + 1))
    return day, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
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
        print(f"=== {yy}-{mm:02d} ({len(days)} days, {args.workers} workers) ===", flush=True)
        t_start = time.time()
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_day, d): d for d in days}
            done = 0
            for f in as_completed(futs):
                _, data = f.result()
                rows.extend(data)
                done += 1
                if done % 10 == 0 or done == len(days):
                    print(f"  {done}/{len(days)} days, {len(rows)} pts, {time.time()-t_start:.0f}s", flush=True)

        if not rows:
            print(f"  [skip] {yy}-{mm:02d} 无数据", flush=True)
            continue
        # 按时间排序
        rows.sort(key=lambda r: r[0])
        times = [r[0] for r in rows]
        vals, freqs = [], []
        for r in rows:
            row = []
            for v in r[1:]:
                if isinstance(v, list):
                    row.append(float("nan"))
                elif v is None:
                    row.append(float("nan"))
                else:
                    fv = float(v)
                    row.append(float("nan") if fv < -1e30 else fv)
            vals.append(row)
        out = os.path.join(out_dir, f"rbsp_a_{yy}{mm:02d}.npz")
        np.savez_compressed(out, time=np.array(times),
                            vals=np.array(vals, dtype=np.float32))
        print(f"[save] {os.path.basename(out)} ({os.path.getsize(out)/1024:.0f} KB, {len(rows)} pts, {time.time()-t_start:.0f}s)", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
