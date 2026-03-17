#!/usr/bin/env python3
"""
Local ML Worker
Connects to the platform backend and executes commands locally.

Usage:
  python tools/local_worker.py --server https://your-backend --token <PAIR_TOKEN>
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

try:
    import websockets  # type: ignore
except Exception:
    print("Missing dependency: websockets. Install with: pip install websockets")
    sys.exit(1)


def _ws_url(server: str) -> str:
    server = server.strip()
    if server.startswith("ws://") or server.startswith("wss://"):
        base = server
    elif server.startswith("http://"):
        base = "ws://" + server[len("http://"):]
    elif server.startswith("https://"):
        base = "wss://" + server[len("https://"):]
    else:
        base = "ws://" + server
    if "/worker" not in base:
        base = base.rstrip("/") + "/worker"
    return base


def _safe_cwd(root: str | None, cwd: str | None) -> str | None:
    if not cwd:
        return root or None
    path = Path(cwd).resolve()
    if root:
        root_path = Path(root).resolve()
        if path != root_path and root_path not in path.parents:
            raise ValueError("cwd outside allowed root")
    return str(path)


async def _run_command(command: str, cwd: str | None, detach: bool) -> dict:
    env = dict(os.environ)
    if detach:
        await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return {
            "returncode": 0,
            "stdout": "detached",
            "stderr": "",
            "truncated": False,
            "detached": True,
        }
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    max_bytes = 1_000_000
    out = stdout or b""
    err = stderr or b""
    truncated = False
    if len(out) > max_bytes:
        out = out[:max_bytes]
        truncated = True
    if len(err) > max_bytes:
        err = err[:max_bytes]
        truncated = True
    return {
        "returncode": proc.returncode,
        "stdout": out.decode("utf-8", errors="replace"),
        "stderr": err.decode("utf-8", errors="replace"),
        "truncated": truncated,
    }


def _safe_file_path(root: str | None, cwd: str | None, rel_path: str) -> Path:
    rel = (rel_path or "").strip().lstrip("/").lstrip("\\")
    if not rel or ".." in rel.replace("\\", "/").split("/"):
        raise ValueError("invalid relative path")
    base = Path(cwd).resolve() if cwd else (Path(root).resolve() if root else Path.cwd().resolve())
    path = (base / rel).resolve()
    if root:
        root_path = Path(root).resolve()
        if path != root_path and root_path not in path.parents:
            raise ValueError("path outside allowed root")
    return path


async def _write_file(root: str | None, cwd: str | None, rel_path: str, content_b64: str) -> dict:
    path = _safe_file_path(root, cwd, rel_path)
    data = base64.b64decode(content_b64.encode("ascii"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "returncode": 0,
        "stdout": f"wrote {path}",
        "stderr": "",
        "path": str(path),
    }


async def _worker_loop(server: str, token: str, root: str | None) -> None:
    ws_url = _ws_url(server)
    while True:
        try:
            async with websockets.connect(ws_url, max_size=2**20) as ws:
                await ws.send(json.dumps({"type": "pair", "token": token}))
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    if msg.get("type") == "paired":
                        print(f"[worker] paired as {msg.get('email')}")
                        continue
                    if msg.get("type") == "exec":
                        job_id = msg.get("job_id")
                        command = (msg.get("command") or "").strip()
                        cwd = (msg.get("cwd") or "").strip()
                        detach = bool(msg.get("detach"))
                        try:
                            safe_cwd = _safe_cwd(root, cwd)
                        except Exception as exc:
                            payload = {"returncode": 1, "stdout": "", "stderr": str(exc)}
                            await ws.send(json.dumps({"type": "result", "job_id": job_id, "payload": payload}))
                            continue
                        payload = await _run_command(command, safe_cwd, detach)
                        await ws.send(json.dumps({"type": "result", "job_id": job_id, "payload": payload}))
                    elif msg.get("type") == "write_file":
                        job_id = msg.get("job_id")
                        cwd = (msg.get("cwd") or "").strip()
                        rel_path = (msg.get("path") or "").strip()
                        content_b64 = (msg.get("content_b64") or "").strip()
                        try:
                            safe_cwd = _safe_cwd(root, cwd)
                            payload = await _write_file(root, safe_cwd, rel_path, content_b64)
                        except Exception as exc:
                            payload = {"returncode": 1, "stdout": "", "stderr": str(exc)}
                        await ws.send(json.dumps({"type": "result", "job_id": job_id, "payload": payload}))
        except Exception as exc:
            print(f"[worker] connection error: {exc}. retrying in 5s...")
            await asyncio.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local ML Worker")
    parser.add_argument("--server", required=True, help="Backend base URL (http/https)")
    parser.add_argument("--token", required=True, help="Pairing token")
    parser.add_argument("--root", help="Allowed root directory for commands")
    args = parser.parse_args()
    root = args.root
    if root:
        root = str(Path(root).resolve())
    asyncio.run(_worker_loop(args.server, args.token, root))


if __name__ == "__main__":
    main()
