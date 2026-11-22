# -*- coding: utf-8 -*-
"""知识库版本管理（SQLite，标准库实现，仍然零依赖）

解决什么问题：
    · 知识库是数据不是代码，改了要能回滚——直接改 JSON 文件没法回滚
    · 新内容需要先验证再上线（draft → active），而不是改完立刻生效
    · 哪些文档从来没人搜到过？靠命中统计，而不是拍脑袋加文档

三张表：
    kb_versions  版本号、状态（draft / active / archived）、变更说明
    kb_docs      某个版本下的文档快照（标题、标签、正文）
    kb_hits      按版本 + 文档统计命中次数，用来找「冷文档」
"""
import json
import os
import sqlite3
import time

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kb.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version     TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL,              -- draft / active / archived
    note        TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_docs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id  INTEGER NOT NULL,
    doc_id      TEXT NOT NULL,
    title       TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]',
    body        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_version ON kb_docs(version_id);

CREATE TABLE IF NOT EXISTS kb_hits (
    version_id  INTEGER NOT NULL,
    doc_id      TEXT NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0,
    last_hit_at TEXT,
    PRIMARY KEY (version_id, doc_id)
);
"""


class KbStore:
    def __init__(self, db_path=DEFAULT_DB):
        self.path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ───────────────────────── 版本
    def versions(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM kb_versions ORDER BY id DESC")]

    def active_version(self):
        row = self.conn.execute(
            "SELECT * FROM kb_versions WHERE status = 'active' LIMIT 1").fetchone()
        return dict(row) if row else None

    def _now(self):
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def create_version(self, version, docs, note=""):
        """新版本一律先落 draft，验证后才 publish"""
        cur = self.conn.execute(
            "INSERT INTO kb_versions (version, status, note, created_at) VALUES (?, 'draft', ?, ?)",
            (version, note, self._now()))
        version_id = cur.lastrowid
        for d in docs:
            self.conn.execute(
                "INSERT INTO kb_docs (version_id, doc_id, title, tags, body) VALUES (?, ?, ?, ?, ?)",
                (version_id, d["id"], d["title"],
                 json.dumps(d.get("tags", []), ensure_ascii=False), d["text"]))
        self.conn.commit()
        return version_id

    def publish(self, version):
        """切流：旧 active → archived，目标版本 → active"""
        row = self.conn.execute("SELECT * FROM kb_versions WHERE version = ?", (version,)).fetchone()
        if not row:
            raise ValueError(f"版本不存在：{version}")
        self.conn.execute("UPDATE kb_versions SET status = 'archived' WHERE status = 'active'")
        self.conn.execute("UPDATE kb_versions SET status = 'active' WHERE version = ?", (version,))
        self.conn.commit()
        return dict(self.conn.execute("SELECT * FROM kb_versions WHERE version = ?", (version,)).fetchone())

    def rollback(self):
        """回滚到上一个 archived 版本"""
        current = self.active_version()
        if not current:
            raise ValueError("当前没有 active 版本")
        prev = self.conn.execute(
            "SELECT * FROM kb_versions WHERE status = 'archived' ORDER BY id DESC LIMIT 1").fetchone()
        if not prev:
            raise ValueError("没有可回滚的历史版本")
        self.conn.execute("UPDATE kb_versions SET status = 'archived' WHERE id = ?", (current["id"],))
        self.conn.execute("UPDATE kb_versions SET status = 'active' WHERE id = ?", (prev["id"],))
        self.conn.commit()
        return {"from": current["version"], "to": prev["version"]}

    # ───────────────────────── 文档
    def docs(self, version=None):
        v = version or (self.active_version() or {}).get("version")
        if not v:
            return []
        row = self.conn.execute("SELECT id FROM kb_versions WHERE version = ?", (v,)).fetchone()
        if not row:
            return []
        return [{"id": r["doc_id"], "title": r["title"],
                 "tags": json.loads(r["tags"]), "text": r["body"]}
                for r in self.conn.execute(
                    "SELECT * FROM kb_docs WHERE version_id = ? ORDER BY id", (row["id"],))]

    # ───────────────────────── 命中统计
    def record_hits(self, doc_ids):
        v = self.active_version()
        if not v or not doc_ids:
            return
        now = self._now()
        for doc_id in doc_ids:
            self.conn.execute("""
                INSERT INTO kb_hits (version_id, doc_id, hits, last_hit_at) VALUES (?, ?, 1, ?)
                ON CONFLICT(version_id, doc_id) DO UPDATE SET hits = hits + 1, last_hit_at = ?
                """, (v["id"], doc_id, now, now))
        self.conn.commit()

    def hit_stats(self):
        """命中排行 + 冷文档（一次都没被检索到过的）"""
        v = self.active_version()
        if not v:
            return {"hot": [], "cold": []}
        rows = self.conn.execute(
            "SELECT doc_id, hits, last_hit_at FROM kb_hits WHERE version_id = ? ORDER BY hits DESC",
            (v["id"],)).fetchall()
        hit_map = {r["doc_id"]: dict(r) for r in rows}
        all_ids = [d["id"] for d in self.docs()]
        hot = [hit_map[i] for i in all_ids if i in hit_map][:5]
        cold = [i for i in all_ids if i not in hit_map]
        return {"hot": hot, "cold": cold}

    def close(self):
        self.conn.close()


def bootstrap(db_path=DEFAULT_DB, kb_json=None):
    """首次运行时把 kb.json 导入成 v1 并启用"""
    store = KbStore(db_path)
    if store.active_version() is None:
        with open(kb_json, encoding="utf-8") as f:
            docs = json.load(f)["docs"]
        store.create_version("v1", docs, note="初始版本，由 kb.json 导入")
        store.publish("v1")
    return store
