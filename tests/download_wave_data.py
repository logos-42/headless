#!/usr/bin/env python3
"""LM4 电磁波持续学习 — NASA CDAWeb (Van Allen Probes RBSP-A) 数据下载

数据源 (HAPI 2.0):
  - RBSP-A_DENSITY_EMFISIS-L4        : 等离子体密度 + 磁场 + 特征频率 (6秒)
  - RBSP-A_MAGNETOMETER_1SEC-GEI_EMFISIS-L3 : 三分量磁场 (1秒)

输出: 每个月一个 .npz, 含对齐后的 时间/密度/磁场/等离子体特征

用法:
  python3 download_wave_data.py --start 2015-01 --months 3
"""
import os, sys, json, time, argparse, urllib.request, urllib.error
from datetime import datetime, timedelta, timezone

HAPI = "https://cdaweb.gsfc.nasa.gov/hapi/data"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wave")


def fetch(ds, t0, t1, retries=3):
    """拉一段时间的数据 (HAPI JSON)."""
    url = (f"{HAPI}?id={ds}&time.min={t0}&time.max={t1}&format=json")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)
            code = d.get("status", {}).get("code")
            if code == 1201:      # no data for time range
                return None
            if code != 1200:
                raise RuntimeError(f"HAPI status {code}: {d.get('status')}")
            return d
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [warn] {ds} {t0[:10]} 失败: {e}", flush=True)
                return None
            time.sleep(3 * (attempt + 1))
    return None


def download_month(ds_list, year, month, out_dir):
    """下载一个月, 逐天下载后合并."""
    import numpy as np
    if month == 12:
        nxt = datetime(year + 1, 1, 1)
    else:
        nxt = datetime(year, month + 1, 1)
    day = datetime(year, month, 1)

    all_rows = {ds: [] for ds in ds_list}
    ndays = 0
    while day < nxt:
        d2 = day + timedelta(days=1)
        t0 = day.strftime("%Y-%m-%dT%H:%M:%SZ")
        t1 = d2.strftime("%Y-%m-%dT%H:%M:%SZ")
        got = False
        for ds in ds_list:
            d = fetch(ds, t0, t1)
            if d:
                all_rows[ds].extend(d["data"])
                got = True
        ndays += 1 if got else 0
        print(f"  {t0[:10]} {'ok' if got else 'no-data'}", flush=True)
        day = d2

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"rbsp_a_{year}{month:02d}.npz")
    save = {}
    for ds in ds_list:
        rows = all_rows[ds]
        if not rows:
            continue
        times = [r[0] for r in rows]
        vals = []
        for r in rows:
            row = []
            for v in r[1:]:
                if isinstance(v, list):
                    row.append(float("nan"))   # 频谱类跳过 (用单独脚本)
                elif v is None:
                    row.append(float("nan"))
                else:
                    fv = float(v)
                    # CDAWeb fill value -1e31 → nan
                    if fv < -1e30:
                        fv = float("nan")
                    row.append(fv)
            vals.append(row)
        save[ds + "__time"] = np.array(times)
        save[ds + "__vals"] = np.array(vals, dtype=np.float32)
    np.savez_compressed(out, **save)
    print(f"[save] {out} ({os.path.getsize(out)/1024:.0f} KB, {ndays} days)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01", help="起始月 YYYY-MM")
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    y, m = map(int, args.start.split("-"))
    ds_list = ["RBSP-A_DENSITY_EMFISIS-L4",
               "RBSP-A_MAGNETOMETER_1SEC-GEI_EMFISIS-L3"]
    for i in range(args.months):
        yy, mm = y + (m - 1 + i) // 12, (m - 1 + i) % 12 + 1
        print(f"=== {yy}-{mm:02d} ===", flush=True)
        download_month(ds_list, yy, mm, args.out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
