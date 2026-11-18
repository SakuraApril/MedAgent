# -*- coding: utf-8 -*-
"""工具集：分诊、红旗、知识库检索、药品查询。

设计原则：每个工具只做一件可解释的事，输入输出都是纯数据，
便于被 Agent 编排、也便于单独做单元测试。
"""
import json
import os
import re

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


_RED = _load("red_flags.json")["rules"]
_TRIAGE = _load("triage_rules.json")
_KB = _load("kb.json")["docs"]
_DRUGS = _load("drugs.json")["drugs"]
_SYN = _load("synonyms.json")["map"]


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


# ────────────────────────────────── 工具 3：知识库检索
def search_kb(query, top_k=2):
    """极简检索：标题/标签/正文的关键词命中数打分。真实产品应替换为向量检索 + rerank。"""
    q = _norm(query)
    grams = {q[i:i + 2] for i in range(max(0, len(q) - 1))}
    scored = []
    for d in _KB:
        blob = _norm(d["title"] + "".join(d["tags"]) + d["text"])
        hit = sum(1 for g in grams if g in blob)
        tag_hit = sum(1 for tag in d["tags"] if _norm(tag) in q)
        score = hit + tag_hit * 3
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [{"id": d["id"], "title": d["title"], "text": d["text"]} for _, d in scored[:top_k]]


# ────────────────────────────────── 工具 4：药品查询
def drug_lookup(text):
    t = _norm(text)
    out = []
    for d in _DRUGS:
        names = [d["name"]] + d.get("alias", [])
        if any(_norm(n) in t for n in names):
            out.append(d)
    return out


TOOLS = {
    "check_red_flag": check_red_flag,
    "triage": triage,
    "search_kb": search_kb,
    "drug_lookup": drug_lookup,
}
