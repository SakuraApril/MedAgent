# -*- coding: utf-8 -*-
"""回归基线：跑一遍全量指标，和 baseline.json 对比，退化就退出码 1

用法：
    python tools/check.py           # 对比基线
    python tools/check.py --update  # 把当前结果写成新基线（确认提升后再用）

为什么需要它（面试会问「怎么保证改代码不把效果改坏」）：
    规则/检索/门控都是手调的，改一个同义词表就可能让某类问题掉分。
    没有基线的话，每次改动只能靠人肉试几个问题，等于没有回归。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

BASELINE = os.path.join(ROOT, "eval", "baseline.json")

# 允许的波动：低于基线超过这个值就算退化
TOLERANCE = 0.02


def collect():
    import metrics

    r = metrics.retrieval_metrics()
    i = metrics.intent_metrics()
    g = metrics.gate_metrics()
    t = metrics.task_metrics()
    p = metrics.perf_metrics()

    return {
        "retrieval": {"recall@1": r["recall@1"], "recall@2": r["recall@2"], "mrr": r["mrr"]},
        "intent": {"acc": i["acc"]},
        "gate": {"误放率": g["误放率"], "漏拦率": g["漏拦率"]},
        "task": {
            "红旗召回率": t["红旗召回率"],
            "分诊准确率": t["分诊准确率"],
            "免责覆盖率": t["免责覆盖率"],
        },
        "perf": {"avg_steps": p["avg_steps"], "p95_ms": p["p95_ms"]},
    }


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    current = collect()
    flat = flatten(current)

    if "--update" in sys.argv or not os.path.exists(BASELINE):
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        print("已写入基线：eval/baseline.json")
        for k, v in flat.items():
            print(f"  {k:28s} {v}")
        return 0

    with open(BASELINE, encoding="utf-8") as f:
        baseline = flatten(json.load(f))

    fails = []
    print(f"{'指标':30s}{'基线':>10s}{'当前':>10s}  状态")
    print("-" * 60)
    for key, base in baseline.items():
        cur = flat.get(key)
        if cur is None:
            fails.append(f"{key}: 基线里有、当前没有")
            continue

        # 延迟是「越低越好」，其余都是「越高越好」
        # 延迟本身抖动大，给宽松阈值，避免机器负载导致假报错
        if key.endswith("_ms"):
            worse = cur > max(base * 2, base + 50)
        else:
            worse = cur < base - TOLERANCE
        status = "FAIL" if worse else "ok"
        if worse:
            fails.append(f"{key}: {base} -> {cur}")
        print(f"{key:30s}{base:>10}{cur:>10}  {status}")

    print("-" * 60)
    if fails:
        print(f"\n❌ 回归 {len(fails)} 项：")
        for f_ in fails:
            print("  · " + f_)
        print("\n如果这是有意的改动，确认后运行：python tools/check.py --update")
        return 1

    print("\n✅ 全部指标不低于基线")
    return 0


if __name__ == "__main__":
    sys.exit(main())
