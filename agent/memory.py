# -*- coding: utf-8 -*-
"""记忆层：会话级短期记忆 + 结构化槽位。

- 短期：保留最近 N 轮对话，用于指代消解（“它”“还疼”“那要挂什么科”）
- 槽位：把症状、科室、红旗、持续时间抽成结构化字段，跨轮累积
"""
import re
import time

MAX_TURNS = 10
PRONOUN = re.compile(r"^(它|这个|那个|这|那|还是|还|仍然|继续|然后|所以|那要)")


class SessionMemory:
    def __init__(self, session_id, max_turns=MAX_TURNS):
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns = []          # [{"role": "user"/"assistant", "text": ..., "ts": ...}]
        self.slots = {
            "symptoms": [],      # 累积症状关键词
            "dept": None,        # 已给出的首诊科室
            "red_flags": [],     # 已出现的危险信号
            "duration": None,    # 持续时间
        }

    # ── 写入
    def add(self, role, text):
        self.turns.append({"role": role, "text": text, "ts": int(time.time())})
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]

    def update_slots(self, text, triage=None, red_hits=None):
        t = text or ""
        for sym in ["发热", "咳嗽", "腹痛", "腹泻", "头痛", "头晕", "胸痛", "皮疹", "呕吐", "失眠"]:
            if sym in t and sym not in self.slots["symptoms"]:
                self.slots["symptoms"].append(sym)
        m = re.search(r"(\d+)\s*(天|小时|周|个月|年)", t)
        if m:
            self.slots["duration"] = m.group(0)
        if triage and triage.get("dept"):
            self.slots["dept"] = triage["dept"]
        if red_hits:
            for r in red_hits:
                if r["name"] not in self.slots["red_flags"]:
                    self.slots["red_flags"].append(r["name"])

    # ── 读取
    def last_user_text(self):
        for t in reversed(self.turns):
            if t["role"] == "user":
                return t["text"]
        return ""

    def needs_context(self, query):
        """短句或以指代词开头 → 需要拼接上文。"""
        q = (query or "").strip()
        return len(q) <= 8 or bool(PRONOUN.match(q))

    def context_text(self, query):
        """返回用于理解的合并文本（不改变给用户的原文）。"""
        if not self.needs_context(query):
            return query
        prev = self.last_user_text()
        if not prev:
            return query
        extra = " ".join(self.slots["symptoms"])
        return f"{prev} {query} {extra}"

    def summary_line(self):
        parts = []
        if self.slots["symptoms"]:
            parts.append("已提及：" + "、".join(self.slots["symptoms"]))
        if self.slots["duration"]:
            parts.append("持续：" + self.slots["duration"])
        if self.slots["dept"]:
            parts.append("上次建议科室：" + self.slots["dept"])
        return "；".join(parts)

    def to_dict(self):
        return {"session_id": self.session_id, "turns": len(self.turns), "slots": self.slots}


class MemoryStore:
    """进程内会话表。生产环境应换成 Redis / 数据库并加过期。"""

    def __init__(self):
        self._store = {}

    def get(self, session_id):
        if session_id not in self._store:
            self._store[session_id] = SessionMemory(session_id)
        return self._store[session_id]

    def clear(self, session_id):
        self._store.pop(session_id, None)

    def stats(self):
        return {"sessions": len(self._store),
                "turns": sum(len(m.turns) for m in self._store.values())}


store = MemoryStore()
