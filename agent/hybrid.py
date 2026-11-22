# -*- coding: utf-8 -*-
"""混合检索：词法路（BM25）+ 字面路（TF-IDF 余弦）→ RRF 融合

为什么要两路：
    TF-IDF 余弦对「整句像不像」敏感，短查询容易因为文档长而被稀释；
    BM25 对「关键词精确命中」敏感，长短文档更公平。
    两路的错法不一样，融合才能互补——这是混合检索的唯一理由。

为什么用 RRF 而不是加权求和：
    余弦相似度落在 0~1，BM25 分数是无上界正数，两路尺度不同；
    加权求和必须先归一化，而归一化本身又引入新超参数。
    RRF 只看名次：score = Σ 1/(k + rank)，k 取 60，天然免疫尺度差异。
"""
from .bm25 import Bm25Index, rrf_fuse
from .rag import TfidfIndex, tokenize

RRF_K = 60


class HybridIndex:
    """对外暴露与 TfidfIndex 相同的 search() 签名，上层无感替换"""

    def __init__(self, docs):
        self.tfidf = TfidfIndex(docs)
        self.chunks = self.tfidf.chunks
        self.bm25 = Bm25Index(self.chunks, tokenize)

    def search(self, query, top_k=2):
        lex = self.tfidf.search(query, top_k=top_k * 3)
        lex = [{**x, "source": "tfidf"} for x in lex]

        bm = self.bm25.search(query, top_k=top_k * 3)
        if bm:
            top = bm[0]["score"] or 1.0
            # 归一化到 0~1，只是为了给置信度模块一个可比的「证据分」
            bm = [{**x, "id": x["doc_id"], "score": round(x["score"] / top, 3), "source": "bm25"} for x in bm]

        if not lex and not bm:
            return []

        fused = rrf_fuse([lex, bm], k=RRF_K, top_k=top_k * 3)

        # 同一文档可能有多块命中，只取融合后最高的那一块
        out, seen = [], set()
        for item in fused:
            doc_id = item.get("id") or item.get("doc_id")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            channel_scores = item.get("_scores", []) or [item.get("score", 0.0)]
            out.append({
                "id": doc_id,
                "title": item["title"],
                "text": item["text"],
                "score": round(max(channel_scores), 3),   # 给置信度用：两路里最高的那个
                "rrf": item.get("rrf"),
                "sources": item.get("sources", []),
            })
            if len(out) >= top_k:
                break
        return out
