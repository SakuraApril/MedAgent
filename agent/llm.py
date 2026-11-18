# -*- coding: utf-8 -*-
"""可选的 LLM 改写层。

默认关闭：没有 OPENAI_API_KEY 时，Agent 完全靠规则引擎运行，
这样离线也能演示、评测结果可复现。设置 key 后，规则生成的答案
会被润色成更自然的表达，但**护栏仍然在前后各拦一次**（先护栏，再润色，再护栏）。
"""
import json
import os
import urllib.request

TIMEOUT = 8


def is_enabled():
    return bool(os.environ.get("OPENAI_API_KEY"))


PROMPT = """你是一名医学科普助手。请把下面这段结构化回答改写得更自然、更有同理心，
但必须遵守：
1) 不做诊断、不下结论、不推荐具体药物剂量；
2) 保留原有的科室建议与急救提示，不得删减；
3) 不得新增任何原文没有的医学结论；
4) 不超过 220 字。
只输出改写后的正文。

原文：
{answer}
"""


def polish(answer, question):
    if not is_enabled():
        return None
    body = json.dumps({
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": PROMPT.format(answer=answer)}],
        "temperature": 0.3,
        "max_tokens": 500,
    }).encode("utf-8")
    req = urllib.request.Request(
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
