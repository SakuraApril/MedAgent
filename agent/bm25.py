# -*- coding: utf-8 -*-
"""BM25 检索（词法路）

为什么自己写而不是引 rank_bm25：
    零依赖是这个项目的卖点之一；BM25 的公式只有三行，写出来才讲得清
    「为什么需要 k1、b 这两个参数」。

BM25 比 TF-IDF 好在哪（面试必问）：
    · 词频饱和：一个词出现 10 次不该比出现 1 次重要 10 倍 → k1 控制饱和速度
    · 长度归一：长文档天然容易命中更多词 → b 控制惩罚强度
"""
import math
from collections import Counter

K1 = 1.5
B = 0.75


class Bm25Index:
    def __init__(self, chunks, tokenizer):
        self.chunks = chunks
        self.tok = tokenizer
        self.docs = [self.tok(c["text"] + " " + c.get("title", "") + "".join(c.get("tags", []))) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.len = [len(d) for d in self.docs]
        self.avg_len = (sum(self.len) / len(self.len)) if self.len else 0.0

        # 文档频率
        df = Counter()
        for d in self.docs:
            for w in set(d):
                df[w] += 1
        n = len(self.docs)
        # idf：加 0.5 平滑，避免出现负数
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.n = n

    def search(self, query, top_k=5):
        q_terms = self.tok(query)
        if not q_terms:
            return []

        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = self.len[i] or 1
            for w in q_terms:
                f = tf.get(w)
                if not f:
                    continue
                idf = self.idf.get(w, 0.0)
                denom = f + K1 * (1 - B + B * dl / (self.avg_len or 1))
                s += idf * (f * (K1 + 1)) / denom
            if s > 0:
                scores.append((i, s))

        scores.sort(key=lambda x: -x[1])
        return [
            {**self.chunks[i], "score": round(s, 4), "source": "bm25"}
            for i, s in scores[:top_k]
        ]


def rrf_fuse(ranked_lists, k=60, top_k=5):
    """Reciprocal Rank Fusion：只用「名次」融合，不用分数

    为什么不加权求和：两路的分数尺度完全不同（余弦相似度 vs BM25 分数），
    归一化会引入新的超参数；RRF 只看排名，天然免疫尺度问题。

    公式：score(d) = Σ 1 / (k + rank_d)，k 通常取 60。
    """
    fused = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            key = item.get("id") or item["text"][:40]
            entry = fused.setdefault(key, {"item": item, "score": 0.0, "sources": [], "_scores": []})
            entry["score"] += 1.0 / (k + rank)
            entry["sources"].append(item.get("source", "?"))
            entry["_scores"].append(item.get("score", 0.0))

    ordered = sorted(fused.values(), key=lambda e: -e["score"])
    out = []
    for e in ordered[:top_k]:
        item = dict(e["item"])
        item["rrf"] = round(e["score"], 5)
        item["sources"] = e["sources"]
        item["_scores"] = e["_scores"]
        out.append(item)
    return out
