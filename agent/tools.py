# -*- coding: utf-8 -*-
"""工具集：分诊、红旗、知识库检索（RAG）、药品查询。

每个工具都用 JSON Schema 声明参数，既能被规则策略调度，
也能直接交给支持 function calling 的大模型调用（见 TOOL_SPECS）。
"""
import json
import os
import re

from . import rag

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


_RED = _load("red_flags.json")["rules"]
_TRIAGE = _load("triage_rules.json")
_KB = _load("kb.json")["docs"]
_DRUGS = _load("drugs.json")["drugs"]
_SYN = _load("synonyms.json")["map"]

# 知识库索引在启动时构建一次（生产环境应换成向量库 + 增量更新）
KB_INDEX = rag.TfidfIndex(_KB)


def _norm(s):
    return re.sub(r"\s+", "", (s or "")).lower()


def expand(s):
    """把口语说法映射到规范术语（通用处理，不为某条用例打补丁）。"""
    t = (s or "").lower()
    extra = []
    for colloquial, canon in _SYN.items():
        if colloquial in t:
            extra.extend(canon)
    return t + " " + " ".join(extra)


# ────────────────────────────────── 工具 1：红旗症状
def check_red_flag(text):
    """返回 (是否命中, 命中的规则列表)。规则 = 主关键词 + 必含词，避免误报。"""
    t = _norm(expand(text))
    hits = []
    for r in _RED:
        if not any(_norm(k) in t for k in r["keywords"]):
            continue
        must = [_norm(m) for m in r.get("must_any", [])]
        if must and not any(m in t for m in must):
            continue
        hits.append({"id": r["id"], "name": r["name"], "advice": r["advice"]})
    return (len(hits) > 0), hits


# ────────────────────────────────── 工具 2：分诊
def triage(text):
    """返回 {dept, matched, alternatives}。命中规则按优先级 + 命中词数排序。"""
    t = _norm(expand(text))
    scored = []
    for d in _TRIAGE["departments"]:
        matched = [s for s in d["symptoms"] if _norm(s) in t]
        if matched:
            scored.append({"dept": d["dept"], "priority": d["priority"],
                           "matched": matched, "score": d["priority"] * 10 + len(matched)})
    if not scored:
        fb = _TRIAGE["fallback"]
        return {"dept": fb["dept"], "matched": [], "alternatives": [], "note": fb["note"]}
    scored.sort(key=lambda x: -x["score"])
    best = scored[0]
    return {
        "dept": best["dept"],
        "matched": best["matched"],
        "alternatives": [s["dept"] for s in scored[1:3]],
        "note": "",
    }


# ────────────────────────────────── 工具 3：知识库（RAG）
def search_kb(query, top_k=2):
    """走 RAG 链路：同义词扩展 → 分块 → TF-IDF 向量 → 余弦相似度 → top-k（带分数）。

    这里补上了 expand()：分诊和红旗判断一直用它，检索却漏了，
    导致「拉肚子」查不到「腹泻」那篇文档——是检索召回率的主要损失点。
    """
    return KB_INDEX.search(expand(query), top_k=top_k)


# ────────────────────────────────── 工具 4：药品查询
def drug_lookup(text):
    t = _norm(text)
    out = []
    for d in _DRUGS:
        names = [d["name"]] + d.get("alias", [])
        if any(_norm(n) in t for n in names):
            out.append(d)
    return out


# ────────────────────────────────── 工具声明（可直接给 LLM 做 function calling）
TOOL_SPECS = [
    {"name": "check_red_flag",
     "description": "判断描述中是否包含需要立即就医的危险信号（胸痛伴冷汗、意识改变、卒中征象等）",
     "parameters": {"type": "object",
                    "properties": {"text": {"type": "string", "description": "用户对症状的描述"}},
                    "required": ["text"]}},
    {"name": "triage",
     "description": "根据症状推断应当首诊的科室",
     "parameters": {"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]}},
    {"name": "search_kb",
     "description": "在健康科普知识库中检索相关条目（向量检索，返回带相似度分数）",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 2}},
                    "required": ["query"]}},
    {"name": "drug_lookup",
     "description": "查询提到了哪些药物，返回通用名与通用注意事项（不给剂量）",
     "parameters": {"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]}},
]

_FUNCS = {"check_red_flag": check_red_flag, "triage": triage,
          "search_kb": search_kb, "drug_lookup": drug_lookup}


def dispatch(name, **kwargs):
    """统一调度入口：规则策略和 LLM 都走这里，便于审计与单测。"""
    if name not in _FUNCS:
        raise KeyError(f"unknown tool: {name}")
    return _FUNCS[name](**kwargs)


TOOLS = _FUNCS
