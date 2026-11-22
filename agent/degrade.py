# -*- coding: utf-8 -*-
"""降级：任何一步失败都不许阻塞主链路

三级降级（越靠后越保守）：
    1. retrieval_error   检索异常 / 没有命中 → 只给科室 + 提示，不编造依据
    2. llm_failed        大模型不可用或超时 → 用模板答案（本项目默认就是模板）
    3. timeout           超出步数或时间预算 → 兜底话术 + 建议就医

为什么把它单独抽出来（面试点）：
    降级不是「try/except 兜一下」，而是要有**明确的规则和可观测的计数**：
    哪一步降级了、降级率多少、降级后用户拿到的是什么。
    否则线上只会表现为「答案变短了」，没人知道发生过降级。
"""

REASONS = {
    "retrieval_error": "检索异常，改为无依据的科室建议",
    "llm_failed": "大模型不可用，改用模板答案",
    "timeout": "超出时间/步数预算，返回兜底话术",
    "empty_result": "检索无命中，只给科室与就医提示",
}


def mark(flags, reason, detail=""):
    """记录一次降级（同一原因只记一次，避免刷屏）"""
    bucket = flags.setdefault("degraded", [])
    for item in bucket:
        if item["reason"] == reason:
            return
    bucket.append({
        "reason": reason,
        "detail": detail or REASONS.get(reason, ""),
    })


def describe(flags):
    """给前端/trace 用的一句话"""
    bucket = flags.get("degraded") or []
    if not bucket:
        return "未降级"
    return "；".join(f"{b['reason']}（{b['detail']}）" for b in bucket)
