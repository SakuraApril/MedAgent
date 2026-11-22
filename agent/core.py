# -*- coding: utf-8 -*-
"""Agent 主循环：记忆 → 红旗 → 分诊 → RAG 检索 → 药品 → 护栏 → 组装。

轨迹（trace）是这个项目最重要的可视化产物：
每一步「想了什么、调了哪个工具、看到什么、花了多久」，都能被前端与审计记录。
"""
import time

from . import confidence, guard, llm
from .memory import store
from .tools import dispatch, TOOL_SPECS, check_red_flag, drug_lookup, search_kb, triage


class Agent:
    def __init__(self, max_steps=8, timeout_s=8.0):
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.tools = TOOL_SPECS

    # ────────────────────────────── 记录一步
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
        flags = {"red_flag": False, "refused_diagnosis": False,
                 "refused_prescription": False, "memory_used": False}

        mem = store.get(session_id)

        # ── 第 0 步：记忆（指代消解）
        effective = mem.context_text(query)
        if effective != query:
            flags["memory_used"] = True
            self._step(trace, "这句话不完整，先取回上下文",
                       "memory.context_text",
                       f"合并为：{effective[:40]}…", t0)
        else:
            self._step(trace, "检查是否有可复用的上下文",
                       "memory.context_text", mem.summary_line() or "新会话，无历史", t0)

        def over_budget():
            return (time.time() - t0) > self.timeout_s or len(trace) >= self.max_steps

        # ── 第 1 步：红旗（最高优先级，先救命）
        hit, rules = dispatch("check_red_flag", text=effective)
        flags["red_flag"] = hit
        self._step(trace, "先判断有没有立刻危及生命的信号", "check_red_flag",
                   f"命中 {len(rules)} 条：{[r['name'] for r in rules]}" if hit else "未命中", t0)
        if hit:
            mem.update_slots(effective, red_hits=rules)
            answer = guard.emergency_answer(rules)
            mem.add("user", query)
            mem.add("assistant", answer)
            return self._finish(answer, trace, flags, t0, mem)

        if over_budget():
            answer = guard.with_disclaimer("处理超时，请重试或直接就医。")
            return self._finish(answer, trace, flags, t0, mem)

        # ── 第 2 步：分诊
        tri = dispatch("triage", text=effective)
        self._step(trace, "根据症状关键词推断首诊科室", "triage",
                   f"推荐 {tri['dept']}（命中：{tri['matched'] or '无'}）", t0)

        # ── 第 3 步：RAG 检索
        docs = dispatch("search_kb", query=effective, top_k=2)
        obs = "；".join(f"{d['title']}({d['score']})" for d in docs) if docs else "无命中"
        self._step(trace, "向量检索知识库（TF-IDF + 余弦相似度）", "search_kb", obs, t0)

        # ── 第 4 步：药品
        drugs = dispatch("drug_lookup", text=effective)
        self._step(trace, "检查是否提到具体药物（仅提示注意事项）", "drug_lookup",
                   f"命中 {len(drugs)} 种：{[d['name'] for d in drugs]}", t0)

        # ── 第 5 步：护栏判断
        parts = []
        if guard.detect_diagnosis(query):
            flags["refused_diagnosis"] = True
            parts.append("**先说清楚一件事**\n" + guard.refuse_diagnosis().split("\n\n")[0])
        if guard.detect_prescription(query):
            flags["refused_prescription"] = True
            parts.append("**关于用药**\n" + guard.refuse_prescription().split("\n\n")[0])

        # ── 第 6 步：置信度闸门（有没有资格回答）
        conf = confidence.evaluate(query, tri, docs, drugs)
        self._step(trace, "评估证据是否足以给出结论", "confidence.evaluate",
                   f"score={conf['total']} 阈值={conf['threshold']} → {conf['action']}"
                   f"（E={conf['parts']['evidence']} C={conf['parts']['coverage']} S={conf['parts']['specific']}）",
                   t0)

        if conf["action"] == "ask_more":
            answer = guard.with_disclaimer(confidence.ask_more_answer())
            mem.update_slots(effective)
            mem.add("user", query)
            mem.add("assistant", answer)
            return self._finish(answer, trace, flags, t0, mem, conf)

        # ── 第 7 步：组装
        parts.append(self._compose(tri, docs, drugs, mem, flags["memory_used"]))
        body = "\n\n".join(p for p in parts if p)
        if conf["action"] == "caveat":
            body = confidence.caveat_line(conf) + "\n\n" + body
        answer = guard.with_disclaimer(guard.soften(body))

        # ── 第 7 步：可选润色（有 key 时；润色后再过一遍护栏）
        polished = llm.polish(answer, query)
        if polished:
            answer = guard.with_disclaimer(guard.soften(polished))
            self._step(trace, "用大模型润色表达（不新增医学结论）", "llm.polish", "已润色", t0)

        mem.update_slots(effective, triage=tri)
        mem.add("user", query)
        mem.add("assistant", answer)
        return self._finish(answer, trace, flags, t0, mem, conf)

    # ────────────────────────────── 组装
    def _compose(self, tri, docs, drugs, mem, memory_used):
        lines = [f"**建议首诊科室：{tri['dept']}**"]
        if tri["matched"]:
            lines.append(f"（依据你提到的：{'、'.join(tri['matched'])}）")
        if tri["alternatives"]:
            lines.append(f"如症状不典型，也可能需要：{'、'.join(tri['alternatives'])}。由医生首诊后决定转诊。")
        if tri.get("note"):
            lines.append(tri["note"])

        if memory_used:
            s = mem.summary_line()
            if s:
                lines.append(f"\n**结合上文**：{s}。下面的建议基于你前面说的情况。")

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
    def _finish(self, answer, trace, flags, t0, mem, conf=None):
        return {
            "answer": answer,
            "trace": trace,
            "flags": flags,
            "memory": mem.to_dict(),
            "confidence": conf,
            "latency_ms": int((time.time() - t0) * 1000),
            "steps": len(trace),
        }


agent = Agent()
