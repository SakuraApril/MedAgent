# -*- coding: utf-8 -*-
"""Agent 主循环：输入一句话 → 输出结构化回答 + 完整轨迹。

轨迹（trace）是这个项目最重要的可视化产物：
每一步「想了什么、调了哪个工具、看到什么」，都能被前端和审计记录。
"""
import time

from . import guard, llm
from .tools import check_red_flag, drug_lookup, search_kb, triage


class Agent:
    def __init__(self, max_steps=6, timeout_s=8.0):
        self.max_steps = max_steps
        self.timeout_s = timeout_s

    # ────────────────────────────── 内部：记录一步
    def _step(self, trace, thought, action, observation, t0):
        trace.append({
            "n": len(trace) + 1,
            "thought": thought,
            "action": action,
            "observation": observation,
            "ms": int((time.time() - t0) * 1000),
        })

    # ────────────────────────────── 主入口
    def run(self, query, session_id="anonymous"):
        t0 = time.time()
        trace = []
        flags = {"red_flag": False, "refused_diagnosis": False, "refused_prescription": False}

        # 预算：超时直接返回保底回答
        def over_budget():
            return (time.time() - t0) > self.timeout_s or len(trace) >= self.max_steps

        # ── 第 1 步：红旗症状（最高优先级，先救命）
        hit, rules = check_red_flag(query)
        flags["red_flag"] = hit
        self._step(trace, "先判断有没有立刻危及生命的信号",
                   "check_red_flag", f"命中 {len(rules)} 条：{[r['name'] for r in rules]}" if hit else "未命中",
                   t0)
        if hit:
            return self._finish(guard.emergency_answer(rules), trace, flags, t0)

        if over_budget():
            return self._finish(guard.with_disclaimer("处理超时，请重试或直接就医。"), trace, flags, t0)

        # ── 第 2 步：分诊（该挂哪个科）
        tri = triage(query)
        self._step(trace, "根据症状关键词推断首诊科室",
                   "triage", f"推荐 {tri['dept']}（命中：{tri['matched'] or '无'}）", t0)

        # ── 第 3 步：知识库（就诊前可以做什么）
        docs = search_kb(query, top_k=2)
        self._step(trace, "查找与症状相关的科普条目",
                   "search_kb", f"命中 {len(docs)} 条：{[d['title'] for d in docs]}", t0)

        # ── 第 4 步：药品（只给注意事项，不给剂量）
        drugs = drug_lookup(query)
        self._step(trace, "检查是否提到具体药物（仅提示注意事项）",
                   "drug_lookup", f"命中 {len(drugs)} 种：{[d['name'] for d in drugs]}", t0)

        # ── 第 5 步：护栏判断
        parts = []
        if guard.detect_diagnosis(query):
            flags["refused_diagnosis"] = True
            parts.append("**先说清楚一件事**\n" + guard.refuse_diagnosis().split("\n\n")[0])
        if guard.detect_prescription(query):
            flags["refused_prescription"] = True
            parts.append("**关于用药**\n" + guard.refuse_prescription().split("\n\n")[0])

        # ── 第 6 步：组装回答
        parts.append(self._compose(query, tri, docs, drugs))
        answer = "\n\n".join(p for p in parts if p)
        answer = guard.soften(answer)
        answer = guard.with_disclaimer(answer)

        # ── 第 7 步：可选润色（有 key 时；润色后再过一遍护栏）
        polished = llm.polish(answer, query)
        if polished:
            polished = guard.soften(polished)
            answer = guard.with_disclaimer(polished)
            self._step(trace, "用大模型润色表达（不新增医学结论）", "llm.polish", "已润色", t0)

        return self._finish(answer, trace, flags, t0)

    # ────────────────────────────── 组装
    def _compose(self, query, tri, docs, drugs):
        lines = [f"**建议首诊科室：{tri['dept']}**"]
        if tri["matched"]:
            lines.append(f"（依据你提到的：{'、'.join(tri['matched'])}）")
        if tri["alternatives"]:
            lines.append(f"如症状不典型，也可能需要：{'、'.join(tri['alternatives'])}。由医生首诊后决定转诊。")
        if tri.get("note"):
            lines.append(tri["note"])

        if docs:
            lines.append("\n**就诊前可以先做的**")
            for d in docs:
                lines.append(f"· {d['title']}：{d['text']}")

        if drugs:
            lines.append("\n**用药提醒（仅通用注意事项，不是处方）**")
            for d in drugs:
                lines.append(f"· {d['name']}（{d['class']}）：常见用于{d['common_use']}。注意：{'；'.join(d['cautions'])}")

        lines.append("\n**什么时候必须立刻就医**")
        lines.append("胸痛胸闷、呼吸困难、意识改变、抽搐、大出血、剧烈头痛、单侧肢体无力或口角歪斜、"
                     "严重过敏、高热伴颈强直、孕期出血或腹痛——出现任一条，直接急诊或 120，不要等门诊。")
        return "\n".join(lines)

    # ────────────────────────────── 收尾
    def _finish(self, answer, trace, flags, t0):
        return {
            "answer": answer,
            "trace": trace,
            "flags": flags,
            "latency_ms": int((time.time() - t0) * 1000),
            "steps": len(trace),
        }


agent = Agent()
