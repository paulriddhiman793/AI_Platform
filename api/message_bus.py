"""
api/message_bus.py — Agent Message Bus

Phase 1: In-memory async pub/sub.
Phase 3: Replace with redis.asyncio.

Channels:
  agent.{id}         — each agent's private inbox
  orchestrator.inbox — all agent reports (forwarded to GUI as team messages)
  p2p.monitor        — copy of every p2p message (GUI activity feed)
  agent.status       — agent idle/working/error changes
  workspace.files    — file write events (GUI file notifications)
  user.output        — direct user-targeted messages
"""
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import Optional


class MockMessageBus:
    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._log: list[dict] = []

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[channel].append(q)
        return q

    async def publish(self, channel: str, message: dict) -> None:
        envelope = {
            "channel":   channel,
            "timestamp": datetime.utcnow().isoformat(),
            "payload":   message,
        }
        self._log.append(envelope)
        preview = str(message.get("content", ""))[:60]
        print(f"[BUS] {message.get('from','?')} → {message.get('to', channel)} | {preview}")
        for q in self._subscribers.get(channel, []):
            await q.put(envelope)

    def get_log(self) -> list[dict]:
        return list(self._log)

    def get_log_for_task(self, task_id: str) -> list[dict]:
        return [e for e in self._log if e["payload"].get("task_id") == task_id]

    def clear_log(self) -> None:
        self._log.clear()


# Global singleton
bus = MockMessageBus()


# ─── Helpers used by BaseAgent ────────────────────────────────────────────────

async def send_to_agent(
    from_agent: str,
    to_agent: str,
    content: str,
    task_id: Optional[str] = None,
) -> None:
    """Send a P2P message to a specific agent's inbox."""
    msg = {
        "from":    from_agent,
        "to":      to_agent,
        "content": content,
        "task_id": task_id,
        "type":    "p2p",
    }
    # Deliver to agent's inbox
    await bus.publish(f"agent.{to_agent}", msg)
    # Also publish to p2p monitor so the GUI activity feed can show it
    await bus.publish("p2p.monitor", msg)


async def broadcast(
    from_agent: str,
    content: str,
    task_id: Optional[str] = None,
) -> None:
    """Broadcast a report to the orchestrator inbox (shown in team chat)."""
    await bus.publish("orchestrator.inbox", {
        "from":    from_agent,
        "to":      "orchestrator",
        "content": content,
        "task_id": task_id,
        "type":    "broadcast",
    })


async def send_to_user(
    from_agent: str,
    content: str,
    task_id: Optional[str] = None,
) -> None:
    """Send a message directly to the user output channel."""
    await bus.publish("user.output", {
        "from":    from_agent,
        "to":      "user",
        "content": content,
        "task_id": task_id,
        "type":    "user_message",
    })


async def publish_status(agent_id: str, status: str) -> None:
    """Publish an agent status change (idle/working/error) to the GUI."""
    await bus.publish("agent.status", {
        "agent_id": agent_id,
        "status":   status,
        "from":     agent_id,
        "to":       "gui",
    })


async def publish_file_event(
    agent_id: str,
    filename: str,
    full_path: str,
    task_id: Optional[str] = None,
) -> None:
    """Notify the GUI that an agent wrote a file to the workspace."""
    await bus.publish("workspace.files", {
        "agent_id": agent_id,
        "filename": filename,
        "path":     full_path,
        "task_id":  task_id,
        "from":     agent_id,
        "to":       "gui",
        "content":  f"File written: {filename}",
    })