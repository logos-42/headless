#!/usr/bin/env python3
"""LM4 特征升级 — 下载 RBSP-A WFR 波功率谱 (EMFISIS-L2, MERGED 版, 本机并行)

为什么是这个数据集:
  - `HFR-SPECTRA` / `WNA-SURVEY` 都是**稀疏/突发模式**, 大量时段 "no data"(实测 2015 全年多数日期无数据)
  - `WFR-SPECTRAL-MATRIX-DIAGONAL-MERGED` 是 **survey+burst 合并版, 全年连续**(6s 采样, 与 DENSITY L4 同频)

内容: 6 个对角功率谱, 各 65 个频段 (~2 Hz ~ 10 kHz)
  BuBu / BvBv / BwBw  — 磁功率谱密度 [nT^2/Hz]
  EuEu / EvEv / EwEw  — 电功率谱密度 [(V/m)^2/Hz]
  (这是真的"电磁波谱": 哨声/合声/嘶声/磁声波等活动随等离子体状态变化,
   且与 density **无解析关系** —— 不像 fpe/fuh 那样是标签泄漏)

CSV 列布局 (395 列):
  [0]=Time  [1:66]=BuBu  [66:131]=BvBv  [131:196]=BwBw
  [196:261]=EuEu  [261:326]=EvEv  [326:391]=EwEw
  [391]=LWEzGainW  [392]=LWExEyGainUV  [393]=SCMGain  [394]=MET

用法: python3 tests/dl_wfr.py --start 2015-01 --months 1 --workers 4
输出: data/wave/wfr_YYYYMM.npz  (time[], psd[N,6,65] float16 log10, gains)
"""
import argparse, io, json, sys, time, urllib.request, calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE = "https://cdaweb.gsfc.nasa.gov/hapi"
DS = "RBSP-A_WFR-SPECTRAL-MATRIX-DIAGONAL-MERGED_EMFISIS-L2"
NB = 65                      # 频段数
NPARAM = 6                   # BuBu BvBv BwBw EuEu EvEv EwEw
FILL = -1.0e31               # HAPI fill
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "wave"


def fetch_day(day: str, retries: int = 3):
    """下载一整天 (UTC), 返回 (times, psd[N,6,65] float32 log10, gains[N,3]) 或 None"""
    t0 = f"{day}T00:00:00Z"
    t1 = f"{day}T23:59:59Z"
    url = f"{BASE}/data?id={DS}&time.min={t0}&time.max={t1}&format=csv"
    last = None
    for attempt in range(retries):
        try:
            raw = urllib.request.urlopen(url, timeout=600).read().decode("utf-8", "replace")
            break
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    else:
        return day, None, f"下载失败: {last}"

    if raw.lstrip().startswith("{"):
        return day, None, "no data"

    times, rows = [], []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) != 1 + NPARAM * NB + 3 + 1:
            continue
        try:
            vals = np.asarray(f[1:1 + NPARAM * NB], dtype=np.float64).reshape(NPARAM, NB)
            gains = np.asarray(f[1 + NPARAM * NB:1 + NPARAM * NB + 3], dtype=np.float64)
        except ValueError:
            continue
        times.append(f[0])
        rows.append(np.concatenate([vals.ravel(), gains]))
    if not rows:
        return day, None, "解析后无有效行"

    arr = np.asarray(rows, dtype=np.float64)          # (N, 390+3)
    psd = arr[:, :NPARAM * NB].reshape(-1, NPARAM, NB)
    gains = arr[:, NPARAM * NB:]
    # 填值/非正 → NaN, 再取 log10
    with np.errstate(invalid="ignore", divide="ignore"):
        psd = np.where((psd > 0) & (psd < 1e30), psd, np.nan)
        psd = np.log10(psd)
    return day, (times, psd.astype(np.float16), gains.astype(np.float32)), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01", help="起始月 YYYY-MM")
    ap.add_argument("--months", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    y0, m0 = (int(x) for x in args.start.split("-"))

    for mi in range(args.months):
        y = y0 + (m0 - 1 + mi) // 12
        m = (m0 - 1 + mi) % 12 + 1
        ndays = calendar.monthrange(y, m)[1]
        days = [f"{y:04d}-{m:02d}-{d:02d}" for d in range(1, ndays + 1)]
        tag = f"{y:04d}{m:02d}"
        print(f"=== WFR {y}-{m:02d} ({ndays}d, {args.workers}w) ===", flush=True)

        t_start = time.time()
        results = {}
        n_ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_day, d): d for d in days}
            for i, fut in enumerate(as_completed(futs), 1):
                day, payload, status = fut.result()
                if payload is not None:
                    results[day] = payload
                    n_ok += 1
                if i % 5 == 0 or i == len(days):
                    print(f"  {i}/{len(days)}d, ok={n_ok}, {time.time()-t_start:.0f}s", flush=True)

        if not results:
            print(f"  !! {tag} 无数据", flush=True)
            continue
        all_t, all_p, all_g = [], [], []
        for day in sorted(results):
            t, p, g = results[day]
            all_t.extend(t); all_p.append(p); all_g.append(g)
        psd = np.concatenate(all_p, axis=0)
        gains = np.concatenate(all_g, axis=0)
        fn = out_dir / f"wfr_{tag}.npz"
        np.savez_compressed(fn, time=np.array(all_t, dtype=object),
                            psd=psd, gains=gains, n_bands=NB)
        mb = fn.stat().st_size / 1024 / 1024
        print(f"[save] {fn.name} ({mb:.1f} MB, {len(all_t)} recs, {time.time()-t_start:.0f}s)", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
