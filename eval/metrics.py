# -*- coding: utf-8 -*-
"""MedAgent 量化指标报告

    python eval/metrics.py

输出两块：
  1. 检索指标：Recall@1 / Recall@2 / MRR（基于 eval/qrels.json 标注集）
  2. 任务指标：红旗召回、分诊准确率、拒答触发、免责覆盖（基于四套用例）
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.core import Agent                # noqa: E402
from agent.guard import DISCLAIMER          # noqa: E402
from agent.tools import search_kb, KB_INDEX  # noqa: E402

EVAL = os.path.join(ROOT, "eval")
agent = Agent()


def load(name):
    with open(os.path.join(EVAL, name), encoding="utf-8") as f:
        return json.load(f)


# ───────────────────────────── 检索指标
def retrieval_metrics(top_k=5):
    items = load("qrels.json")["items"]
    hit1 = hit2 = hit3 = hit5 = 0
    rr_sum = 0.0
    details = []

    for it in items:
        results = search_kb(it["query"], top_k=top_k)
        ids = [r["id"] for r in results]
        rank = None
        for i, doc_id in enumerate(ids, start=1):
            if doc_id in it["relevant"]:
                rank = i
                break
        hit1 += 1 if rank == 1 else 0
        hit2 += 1 if rank and rank <= 2 else 0
        hit3 += 1 if rank and rank <= 3 else 0
        hit5 += 1 if rank and rank <= 5 else 0
        rr_sum += (1.0 / rank) if rank else 0.0
        details.append({"query": it["query"], "rank": rank, "got": ids})

    n = len(items)
    return {
        "n": n,
        "recall@1": round(hit1 / n, 3),
        "recall@2": round(hit2 / n, 3),
        "recall@3": round(hit3 / n, 3),
        "recall@5": round(hit5 / n, 3),
        "mrr": round(rr_sum / n, 3),
        "details": details,
        "chunks": len(KB_INDEX.chunks),
    }


# ───────────────────────────── 任务指标
def task_metrics():
    buckets = {}
    for name in ["cases.json", "multiturn.json", "holdout.json", "blind.json"]:
        buckets[name] = load(name)["cases"]

    def run_case(c):
        if "turns" in c:
            res = None
            for turn in c["turns"]:
                res = agent.run(turn, "metrics-" + c["id"])
            return res
        return agent.run(c["query"], "metrics")

    stats = {
        "red_flag_cases": 0, "red_flag_hit": 0,
        "triage_cases": 0, "triage_hit": 0,
        "diag_cases": 0, "diag_refused": 0,
        "rx_cases": 0, "rx_refused": 0,
        "answers": 0, "with_disclaimer": 0,
    }
    red_fp = 0          # 期望非红旗但被判红旗
    red_neg = 0

    for name, cases in buckets.items():
        for c in cases:
            res = run_case(c)
            exp = c["expect"]
            ans = res["answer"]
            stats["answers"] += 1
            if DISCLAIMER in ans:
                stats["with_disclaimer"] += 1

            if "red_flag" in exp:
                if exp["red_flag"]:
                    stats["red_flag_cases"] += 1
                    stats["red_flag_hit"] += 1 if res["flags"]["red_flag"] else 0
                else:
                    red_neg += 1
                    red_fp += 1 if res["flags"]["red_flag"] else 0

            if exp.get("dept"):
                stats["triage_cases"] += 1
                m = re.search(r"建议首诊科室：(.+?)\*\*", ans)
                got = m.group(1).strip() if m else ""
                stats["triage_hit"] += 1 if exp["dept"] in got else 0

            if exp.get("refuse_diagnosis"):
                stats["diag_cases"] += 1
                stats["diag_refused"] += 1 if res["flags"]["refused_diagnosis"] else 0

            if exp.get("refuse_prescription"):
                stats["rx_cases"] += 1
                stats["rx_refused"] += 1 if res["flags"]["refused_prescription"] else 0

    def rate(a, b):
        return round(a / b, 3) if b else None

    return {
        "红旗召回率": rate(stats["red_flag_hit"], stats["red_flag_cases"]),
        "红旗误报率（负例）": rate(red_fp, red_neg),
        "分诊准确率": rate(stats["triage_hit"], stats["triage_cases"]),
        "诊断类拒答率": rate(stats["diag_refused"], stats["diag_cases"]),
        "处方类拒答率": rate(stats["rx_refused"], stats["rx_cases"]),
        "免责覆盖率": rate(stats["with_disclaimer"], stats["answers"]),
        "样本数": {k: v for k, v in stats.items() if v},
    }


def perf_metrics():
    """性能指标：步数分布与延迟分位（跑 42 条用例）"""
    lat, steps = [], []
    buckets = {}
    for name in ["cases.json", "multiturn.json", "holdout.json", "blind.json"]:
        for c in load(name)["cases"]:
            if "turns" in c:
                res = None
                for turn in c["turns"]:
                    res = agent.run(turn, "perf-" + c["id"])
            else:
                res = agent.run(c["query"], "perf")
            lat.append(res["latency_ms"])
            steps.append(res["steps"])
            intent = res.get("intent") or "?"
            buckets.setdefault(intent, []).append(res["steps"])

    lat.sort()
    def pct(p):
        if not lat:
            return 0
        idx = min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))
        return lat[idx]

    return {
        "n": len(lat),
        "avg_steps": round(sum(steps) / len(steps), 2) if steps else 0,
        "max_steps": max(steps) if steps else 0,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "avg_ms": round(sum(lat) / len(lat), 1) if lat else 0,
        "steps_by_intent": {k: round(sum(v) / len(v), 2) for k, v in buckets.items()},
    }


def intent_metrics():
    from agent import intent as intent_mod
    items = load("intent.json")["items"]
    ok = 0
    wrong = []
    for it in items:
        got = intent_mod.classify(it["query"])["intent"]
        if got == it["intent"]:
            ok += 1
        else:
            wrong.append((it["query"], it["intent"], got))
    return {"n": len(items), "acc": round(ok / len(items), 3), "wrong": wrong}


def gate_metrics():
    """门控指标：误放率（该拦没拦）与漏拦率（不该拦却拦）"""
    items = load("gate.json")["items"]
    should_gate = 0
    false_accept = 0      # 该拦没拦
    should_pass = 0
    false_reject = 0      # 不该拦却拦
    details = []

    for it in items:
        res = agent.run(it["query"], "gate-" + str(abs(hash(it["query"])) % 10000))
        conf = res.get("confidence") or {}
        gated = conf.get("action") == "ask_more"
        ok = (gated == it["expect_gate"])
        if it["expect_gate"]:
            should_gate += 1
            if not gated:
                false_accept += 1
        else:
            should_pass += 1
            if gated:
                false_reject += 1
        details.append({"query": it["query"], "expect_gate": it["expect_gate"],
                        "action": conf.get("action"), "score": conf.get("total"), "ok": ok})

    return {
        "n": len(items),
        "should_gate": should_gate,
        "should_pass": should_pass,
        "误放率": round(false_accept / should_gate, 3) if should_gate else None,
        "漏拦率": round(false_reject / should_pass, 3) if should_pass else None,
        "准确率": round(sum(1 for d in details if d["ok"]) / len(items), 3),
        "details": details,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    r = retrieval_metrics()
    print("=" * 56)
    print("检索指标（标注集 %d 条，知识库 %d 块）" % (r["n"], r["chunks"]))
    print("=" * 56)
    print("Recall@1 = %.1f%%" % (r["recall@1"] * 100))
    print("Recall@2 = %.1f%%" % (r["recall@2"] * 100))
    print("Recall@3 = %.1f%%" % (r["recall@3"] * 100))
    print("Recall@5 = %.1f%%" % (r["recall@5"] * 100))
    print("MRR      = %.3f" % r["mrr"])
    missed = [d for d in r["details"] if d["rank"] is None or d["rank"] > 1]
    if missed:
        print("\n非首位命中的查询：")
        for d in missed:
            print("  %-22s rank=%s got=%s" % (d["query"], d["rank"], d["got"]))

    pm = perf_metrics()
    print("\n" + "=" * 56)
    print("性能指标（%d 次请求，无大模型调用）" % pm["n"])
    print("=" * 56)
    print("平均步数 = %.2f（最多 %d 步）" % (pm["avg_steps"], pm["max_steps"]))
    print("延迟 P50 = %d ms / P95 = %d ms / 均值 %.1f ms" % (pm["p50_ms"], pm["p95_ms"], pm["avg_ms"]))
    print("分意图步数：" + "，".join(f"{k}={v}" for k, v in sorted(pm["steps_by_intent"].items())))

    im = intent_metrics()
    print("\n" + "=" * 56)
    print("意图识别（%d 条，五类）" % im["n"])
    print("=" * 56)
    print("准确率 = %.1f%%" % (im["acc"] * 100))
    for q, exp, got in im["wrong"]:
        print("  错分: %-20s 期望=%s 实际=%s" % (q, exp, got))

    g = gate_metrics()
    print("\n" + "=" * 56)
    print("门控指标（%d 条：应拦 %d / 应放 %d）" % (g["n"], g["should_gate"], g["should_pass"]))
    print("=" * 56)
    print("误放率 = %.1f%%（该拦没拦）" % (g["误放率"] * 100))
    print("漏拦率 = %.1f%%（不该拦却拦）" % (g["漏拦率"] * 100))
    print("门控准确率 = %.1f%%" % (g["准确率"] * 100))
    for d in g["details"]:
        if not d["ok"]:
            print("  未通过: %-18s 期望拦=%s 实际=%s score=%s" % (d["query"], d["expect_gate"], d["action"], d["score"]))

    t = task_metrics()
    print("\n" + "=" * 56)
    print("任务指标（四套用例合计 %d 条）" % t["样本数"].get("answers", 0))
    print("=" * 56)
    for k in ["红旗召回率", "红旗误报率（负例）", "分诊准确率",
              "诊断类拒答率", "处方类拒答率", "免责覆盖率"]:
        v = t[k]
        print("%-14s %s" % (k, "—" if v is None else "%.1f%%" % (v * 100)))
