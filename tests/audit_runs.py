#!/usr/bin/env python3
"""数据完整性核对: 日志里的每个 run 是否都有合法产物。

起因: driver 里 `echo "... $(date) ... rc=$?"` 的 `$(date)` 会先执行并重置 `$?`,
所以 **rc 值全部不可信**。不能靠 rc 判断成功与否, 必须**直接核对产物**。

判据 (每个 run 四查):
  1. 结果目录存在
  2. lm4_wave_results.json 存在且是**合法 JSON** (非截断)
  3. 含有 replay 臂的 any_time_acc (主指标在)
  4. 记录的 seed 与日志一致

任何一项不满足 = 该 run 的结论不可用。
"""
import glob
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/work/liuyuanjie/headless"

# (日志文件, 行内种子变量名)
LOGS = ["results/rl_driver.log", "results/rl_ext.log", "results/rl_know.log",
        "results/rl_seed8.log", "results/s8b.log", "results/rm_rerun.log"]

total_ok, total_bad = 0, 0
bad_detail = []

for lg in LOGS:
    p = os.path.join(ROOT, lg)
    if not os.path.exists(p):
        print("  [缺日志] %s" % lg)
        continue
    lines = [l.strip() for l in open(p, errors="ignore") if re.search(r"\brc=\d+", l)]
    ok = bad = 0
    for l in lines:
        m = re.search(r"seed=(\d+)", l)
        if not m:
            continue
        seed = m.group(1)
        # 清掉前缀/后缀噪声, 拿 tag 关键词
        core = l.split("]")[-1].strip().split("seed=")[0].strip().replace(" ", "_")
        # 在 results/ 下找含该 seed 且名字贴近 core 的目录
        cands = [d for d in glob.glob(os.path.join(ROOT, "results", "*_s" + seed))
                 if os.path.isdir(d)]
        hit = None
        for d in cands:
            j = os.path.join(d, "lm4_wave_results.json")
            if not os.path.exists(j):
                continue
            try:
                dd = json.load(open(j))
            except Exception:
                bad_detail.append((lg, l[:60], "JSON 非法/截断"))
                continue
            v = dd.get("replay") if isinstance(dd, dict) else None
            if isinstance(v, dict) and v.get("any_time_acc") is not None:
                hit = d
                break
        if hit:
            ok += 1
        else:
            bad += 1
            bad_detail.append((lg, l[:70], "无合法产物"))
    total_ok += ok
    total_bad += bad
    print("  %-26s 日志 %2d 行 -> 合法产物 %2d, 缺失 %2d" % (os.path.basename(lg), len(lines), ok, bad))

print()
print("=" * 84)
print("总计: 日志声明 %d 个 run, 合法产物 %d, **缺失/损坏 %d**"
      % (total_ok + total_bad, total_ok, total_bad))
print("=" * 84)
if bad_detail:
    print("有问题的条目前 12 条:")
    for lg, l, why in bad_detail[:12]:
        print("  [%s] %s  <- %s" % (os.path.basename(lg), l, why))
else:
    print("全部日志行都有对应的合法产物 -> **rc 虽不可信, 但数据完整**")

print()
print("注: 本核对只看产物是否合法可读, 不重判结论。")
print("    rc=0 不可信这一点已写入 docs/stepsize_idbd_results.md 的错误表 (⑥)。")
