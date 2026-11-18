# -*- coding: utf-8 -*-
"""HTTP 服务：静态页 + 对话接口 + 审计查询 + 在线评测。

只用标准库（http.server），不装任何依赖即可运行：
    python app.py            # http://127.0.0.1:8000
"""
import hashlib
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.core import agent                       # noqa: E402
from agent.memory import store                     # noqa: E402
from agent import llm                             # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
AUDIT = os.path.join(ROOT, "audit.log")
START = time.time()

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml"}


def audit(record):
    """审计日志：只落哈希，不落原始症状文本（隐私边界）。"""
    with open(AUDIT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    server_version = "MedAgent/0.1"

    # ── 工具
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if not os.path.isfile(path):
            return self._json({"error": "not found"}, 404)
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(os.path.splitext(path)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def log_message(self, fmt, *args):
        pass  # 静默默认日志，避免刷屏

    # ── 路由
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"))
        if self.path == "/api/health":
            return self._json({
                "ok": True,
                "llm_enabled": llm.is_enabled(),
                "uptime_s": int(time.time() - START),
                "version": "0.1.0",
            })
        if self.path.startswith("/api/memory"):
            sid = self.path.split("session=")[1].split("&")[0] if "session=" in self.path else ""
            if not sid:
                return self._json({"stats": store.stats()})
            m = store.get(sid)
            return self._json({"session_id": sid, "summary": m.summary_line(),
                               "slots": m.slots, "recent": m.turns[-6:]})
        if self.path.startswith("/api/audit"):
            limit = 20
            if "limit=" in self.path:
                try:
                    limit = max(1, min(200, int(self.path.split("limit=")[1].split("&")[0])))
                except Exception:
                    pass
            rows = []
            if os.path.exists(AUDIT):
                with open(AUDIT, encoding="utf-8") as f:
                    rows = [json.loads(x) for x in f.readlines()[-limit:] if x.strip()]
            return self._json({"count": len(rows), "records": rows})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/chat":
            data = self._body()
            msg = (data.get("message") or "").strip()
            sid = (data.get("session_id") or "anon")[:32]
            if not msg:
                return self._json({"error": "message 不能为空"}, 400)
            if len(msg) > 800:
                return self._json({"error": "单条消息请控制在 800 字以内"}, 400)

            res = agent.run(msg, sid)
            audit({
                "ts": int(time.time()),
                "session": sid,
                "q_hash": hashlib.sha256(msg.encode("utf-8")).hexdigest()[:12],
                "q_len": len(msg),
                "flags": res["flags"],
                "steps": res["steps"],
                "latency_ms": res["latency_ms"],
            })
            return self._json(res)

        if self.path == "/api/eval":
            sys.path.insert(0, os.path.join(ROOT, "eval"))
            from run_eval import evaluate
            return self._json(evaluate())

        return self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"MedAgent 已启动： http://127.0.0.1:{port}")
    print(f"LLM 润色： {'开启' if llm.is_enabled() else '关闭（纯规则引擎，可离线演示）'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
