#!/usr/bin/env python3
"""LM4 B 阶段 — 下载 MAG 的卫星位置 (coordinates) 用于构造 L-shell 特征

用 `RBSP-A_MAGNETOMETER_4SEC-GEI_EMFISIS-L3` (4s 采样, 含 coordinates):
  比 1SEC 版小 4 倍 (3MB/天 vs 13MB/天), 12 个月约 1GB。

CSV 17 列布局:
  [0]Time [1-3]Mag_xyz [4]Magnitude [5]delta [6]lambda [7]rms
  [8-10]coordinates_xyz(km, GEI) [11]range_flag [12]partition [13]MET
  [14]calState [15]magInvalid [16]magFill

只保留: time + coordinates(3) + Magnitude + 有效性标志(magFill/magInvalid)
L-shell 在 load 侧算: r_Re=|pos|/6371, 偶极 L ≈ r / cos²(λ_geo), sinλ = z/r

用法: python3 tests/dl_mag_pos.py --start 2015-01 --months 12 --workers 4
输出: data/wave/magpos_YYYYMM.npz (time[], coords[N,3] f32, mag[N], flags[N,2] u8)
"""
import argparse, calendar, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE = "https://cdaweb.gsfc.nasa.gov/hapi"
DS = "RBSP-A_MAGNETOMETER_4SEC-GEI_EMFISIS-L3"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "wave"


def fetch_day(day: str, retries: int = 3):
    url = (f"{BASE}/data?id={DS}&time.min={day}T00:00:00Z"
           f"&time.max={day}T23:59:59Z&format=csv")
    raw = None
    last = None
    for attempt in range(retries):
        try:
            raw = urllib.request.urlopen(url, timeout=600).read().decode("utf-8", "replace")
            break
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    if raw is None:
        return day, None, f"下载失败: {last}"
    if raw.lstrip().startswith("{"):
        return day, None, "no data"

    T, C, M, FL = [], [], [], []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) != 17:
            continue
        try:
            coord = (float(f[8]), float(f[9]), float(f[10]))
            mag = float(f[4])
            fl = (int(float(f[16])), int(float(f[15])))   # magFill, magInvalid
        except ValueError:
            continue
        T.append(f[0]); C.append(coord); M.append(mag); FL.append(fl)
    if not T:
        return day, None, "解析后无有效行"
    return day, (T, np.asarray(C, dtype=np.float32),
                 np.asarray(M, dtype=np.float32),
                 np.asarray(FL, dtype=np.uint8)), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01")
    ap.add_argument("--months", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    y0, m0 = (int(x) for x in args.start.split("-"))

    for mi in range(args.months):
        y = y0 + (m0 - 1 + mi) // 12
        m = (m0 - 1 + mi) % 12 + 1
        nd = calendar.monthrange(y, m)[1]
        days = [f"{y:04d}-{m:02d}-{d:02d}" for d in range(1, nd + 1)]
        print(f"=== MAGPOS {y}-{m:02d} ({nd}d, {args.workers}w) ===", flush=True)
        t0 = time.time(); res = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_day, d): d for d in days}
            for i, fut in enumerate(as_completed(futs), 1):
                day, payload, st = fut.result()
                if payload is not None:
                    res[day] = payload
                if i % 10 == 0 or i == len(days):
                    print(f"  {i}/{len(days)}d, ok={len(res)}, {time.time()-t0:.0f}s", flush=True)
        if not res:
            print(f"  !! {y}-{m:02d} 无数据", flush=True); continue
        T, C, M, FL = [], [], [], []
        for day in sorted(res):
            t, c, mm, fl = res[day]
            T.extend(t); C.append(c); M.append(mm); FL.append(fl)
        fn = out_dir / f"magpos_{y:04d}{m:02d}.npz"
        np.savez_compressed(fn, time=np.array(T, dtype=object),
                            coords=np.concatenate(C), mag=np.concatenate(M),
                            flags=np.concatenate(FL))
        print(f"[save] {fn.name} ({fn.stat().st_size/1024/1024:.1f} MB, {len(T)} recs, "
              f"{time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
