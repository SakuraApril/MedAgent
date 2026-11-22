# -*- coding: utf-8 -*-
"""降级验收：注入失败，确认主链路仍然返回答案

    python tools/verify_degrade.py

不 mock 掉整个 Agent，只把某个工具换成会抛异常的函数，
验证「检索挂了」时用户仍然拿到科室建议 + 就医提示，而不是 500。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent import tools                      # noqa: E402
from agent.core import Agent                 # noqa: E402


def break_retrieval():
    """把工具表里的 search_kb 换成会抛异常的函数，返回还原函数"""
    original = tools.TOOLS["search_kb"]
    tools.TOOLS["search_kb"] = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("模拟：检索引擎超时"))

    def restore():
        tools.TOOLS["search_kb"] = original

    return restore


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("降级验收（注入失败，确认不阻塞主链路）")
    ok_all = True

    # 1) 检索异常：仍应返回答案，并记录 retrieval_error
    restore = break_retrieval()
    try:
        res = Agent().run("拉肚子还发烧，一天跑好几趟厕所", "verify-degrade")
    finally:
        restore()
    reasons = [d["reason"] for d in (res["flags"].get("degraded") or [])]
    ok = bool(res["answer"]) and "retrieval_error" in reasons
    print(f"  {'✅' if ok else '❌'} {'检索异常':10s} 降级={reasons or '无'} 答案 {len(res['answer'])} 字")
    ok_all &= ok

    # 2) 正常路径：不应出现 retrieval_error
    res2 = Agent().run("突然胸痛，呼吸困难", "verify-normal")
    r2 = [d["reason"] for d in (res2["flags"].get("degraded") or [])]
    ok2 = "retrieval_error" not in r2
    print(f"  {'✅' if ok2 else '❌'} {'正常路径':10s} 降级={r2 or '无'}")
    ok_all &= ok2

    print()
    print("✅ 降级链路验证通过" if ok_all else "❌ 降级链路验证失败")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
