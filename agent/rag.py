# -*- coding: utf-8 -*-
"""轻量 RAG：分块 + TF-IDF 向量 + 余弦相似度。

为什么不用 embedding 模型：这个项目的约束是「零依赖、可离线复现」。
TF-IDF 向量空间模型同样是「把文本变成向量、算相似度、取 top-k」的完整 RAG 链路，
只是向量是稀疏的词频权重而非稠密语义向量。

替换点：把 `TfidfIndex.search` 换成 embedding + 向量库（如 FAISS / pgvector）即可，
上层 `search_kb` 的签名不变。
"""
import math
import re
from collections import Counter

# 中文停用词（极简）
STOP = set("的了是在和与或及也就都而及其这那有对我你他她它们一个什么怎么如何")


def tokenize(text):
    """中文按 2-gram，英文数字按词；去掉停用字。"""
    text = (text or "").lower()
    tokens = []
    # 英文/数字
    tokens += re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)
    # 中文 2-gram
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in cjk:
        for i in range(len(seg) - 1):
            pair = seg[i:i + 2]
            if not (set(pair) & STOP):
                tokens.append(pair)
        if len(seg) == 1:
            tokens.append(seg)
    return tokens


def chunk(docs, min_len=24):
    """按句切分成块，块保留原文档 id，便于回溯。"""
    chunks = []
    for d in docs:
        parts = re.split(r"(?<=[。！？；])", d["text"])
        buf = ""
        for p in parts:
            if len(buf) + len(p) < min_len:
                buf += p
            else:
                if buf.strip():
                    chunks.append({"doc_id": d["id"], "title": d["title"], "text": buf.strip()})
                buf = p
        if buf.strip():
            chunks.append({"doc_id": d["id"], "title": d["title"], "text": buf.strip()})
    return chunks


class TfidfIndex:
    def __init__(self, docs):
        self.chunks = chunk(docs)
        self.tfs = []
        df = Counter()
        for c in self.chunks:
            tf = Counter(tokenize(c["title"] + c["text"]))
            self.tfs.append(tf)
            df.update(set(tf.keys()))
        n = max(1, len(self.chunks))
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        self.vecs = [self._vec(tf) for tf in self.tfs]

    def _vec(self, tf):
        v = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    @staticmethod
    def _cos(a, b):
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b.get(t, 0.0) for t, w in a.items())

    def search(self, query, top_k=2, min_score=0.06):
        qv = self._vec(Counter(tokenize(query)))
        scored = []
        for i, cv in enumerate(self.vecs):
            s = self._cos(qv, cv)
            if s >= min_score:
                scored.append((s, i))
        scored.sort(reverse=True)
        out, seen = [], set()
        for s, i in scored:
            c = self.chunks[i]
            if c["doc_id"] in seen:      # 同一文档只取最高分的块
                continue
            seen.add(c["doc_id"])
            out.append({"id": c["doc_id"], "title": c["title"], "text": c["text"],
                        "score": round(s, 3)})
            if len(out) >= top_k:
                break
        return out
