#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MedAgent 的 MCP server（stdio + JSON-RPC 2.0）。

把 tools.py 里的 4 个工具按 Model Context Protocol 暴露出去，
任意 MCP 客户端（Claude Desktop / Cursor / 自研宿主）都能直接调用，
不需要为每个宿主重写一遍工具层。

启动：
    python mcp_server.py

协议要点：
- 一行一条 JSON-RPC 2.0 消息（换行分隔）
- stdout 只写协议消息，日志一律走 stderr，否则会污染协议流
"""
import json
import sys
import traceback

from agent.tools import TOOL_SPECS, dispatch

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "medagent", "version": "0.1.0"}


def _log(msg):
    print(f"[medagent-mcp] {msg}", file=sys.stderr, flush=True)


def _tool_list():
    return [{"name": t["name"], "description": t["description"], "inputSchema": t["parameters"]}
            for t in TOOL_SPECS]


def _tool_result(name, arguments):
    """调用工具并把结果转成 MCP 的 content 结构。"""
    raw = dispatch(name, **(arguments or {}))
    if isinstance(raw, tuple):          # check_red_flag 返回 (bool, list)
        hit, rules = raw
        payload = {"hit": hit, "rules": rules}
    else:
        payload = raw
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
        "isError": False,
    }


def handle(req):
    """处理一条请求，返回响应 dict；通知类消息返回 None。"""
    method = req.get("method")
    rid = req.get("id")
    params = req.get("params") or {}

    # 通知：不需要响应
    if rid is None:
        _log(f"notification: {method}")
        return None

    if method == "initialize":
        client_version = params.get("protocolVersion") or PROTOCOL_VERSION
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": client_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        }}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": _tool_list()}}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name not in {t["name"] for t in TOOL_SPECS}:
            return {"jsonrpc": "2.0", "id": rid,
                     "error": {"code": -32602, "message": f"未知工具: {name}"}}
        try:
            return {"jsonrpc": "2.0", "id": rid, "result": _tool_result(name, args)}
        except Exception as e:                      # 工具内部异常也要回给客户端
            _log(traceback.format_exc())
            return {"jsonrpc": "2.0", "id": rid,
                    "result": {"content": [{"type": "text", "text": f"工具执行失败: {e}"}],
                               "isError": True}}

    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"resources": []}}

    if method == "prompts/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"prompts": []}}

    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}

    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"不支持的方法: {method}"}}


def main():
    _log(f"start, {len(TOOL_SPECS)} tools")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"jsonrpc": "2.0", "id": None,
                              "error": {"code": -32700, "message": f"解析失败: {e}"}},
                             ensure_ascii=False), flush=True)
            continue

        try:
            resp = handle(req)
        except Exception as e:
            _log(traceback.format_exc())
            resp = {"jsonrpc": "2.0", "id": req.get("id"),
                    "error": {"code": -32603, "message": str(e)}}

        if resp is not None:
            print(json.dumps(resp, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
