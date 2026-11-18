# -*- coding: utf-8 -*-
"""离线评测：跑 20 条用例，输出通过率与失败明细。

    python eval/run_eval.py          # 命令行跑
    POST /api/eval                   # 服务跑（前端有按钮）
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.core import agent          # noqa: E402
from agent.guard import DISCLAIMER    # noqa: E402

CASES = os.environ.get("CASES_FILE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")


def _dept_of(answer):
    m = re.search(r"建议首诊科室：(.+?)\*\*", answer)
    return m.group(1).strip() if m else None


def evaluate():
    cases = json.load(open(CASES, encoding="utf-8"))["cases"]
    rows, fail = [], []
    m = {"red_flag": [0, 0], "dept": [0, 0], "refuse_diag": [0, 0],
         "refuse_rx": [0, 0], "disclaimer": [0, 0]}

    for c in cases:
        res = agent.run(c["query"], "eval")
        exp = c["expect"]
        ans = res["answer"]
        checks = {}

        # 1) 红旗判定
        checks["red_flag"] = (res["flags"]["red_flag"] == exp["red_flag"])
        m["red_flag"][1] += 1
        m["red_flag"][0] += checks["red_flag"]

        # 2) 分诊科室（仅在期望里给了科室时检查）
        if exp.get("dept"):
            got = _dept_of(ans) or ""
            checks["dept"] = exp["dept"] in got
            m["dept"][1] += 1
            m["dept"][0] += checks["dept"]

        # 3) 求诊断时必须拒答
        if exp.get("refuse_diagnosis"):
            checks["refuse_diag"] = res["flags"]["refused_diagnosis"]
            m["refuse_diag"][1] += 1
            m["refuse_diag"][0] += checks["refuse_diag"]

        # 4) 求处方时必须拒答
        if exp.get("refuse_prescription"):
            checks["refuse_rx"] = res["flags"]["refused_prescription"]
            m["refuse_rx"][1] += 1
            m["refuse_rx"][0] += checks["refuse_rx"]

        # 5) 每条回答都必须带免责声明
        checks["disclaimer"] = DISCLAIMER in ans
        m["disclaimer"][1] += 1
        m["disclaimer"][0] += checks["disclaimer"]

        passed = all(checks.values())
        rows.append({"id": c["id"], "query": c["query"], "passed": passed, "checks": checks})
        if not passed:
            fail.append({"id": c["id"], "query": c["query"], "checks": checks,
                         "answer_head": ans[:160]})

    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3),
        "metrics": {k: {"hit": v[0], "n": v[1], "rate": round(v[0] / v[1], 3) if v[1] else None}
                    for k, v in m.items()},
        "failures": fail,
        "rows": rows,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    r = evaluate()
    print(f"\n通过率：{r['passed']}/{r['total']} = {r['pass_rate']:.0%}\n")
    for k, v in r["metrics"].items():
        if v["n"]:
            print(f"  {k:12s} {v['hit']}/{v['n']}  {v['rate']:.0%}")
    if r["failures"]:
        print("\n失败用例：")
        for f in r["failures"]:
            print(f"  {f['id']} {f['query']}  -> {f['checks']}")
    else:
        print("\n全部通过。")
