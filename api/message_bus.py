"""
message_bus.py — Mock in-memory message bus (replaces Redis for local dev)
In production Phase 3, swap this for a real Redis pub/sub client.
"""
import asyncio
import json
from datetime import datetime
from collections import defaultdict


class MockMessageBus:
    """
    In-memory async message bus that mimics Redis Pub/Sub.
    Agents subscribe to channels and receive messages asynchronously.
    """

    def __init__(self):
        self._channels: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._log: list[dict] = []

    def subscribe(self, channel: str) -> asyncio.Queue:
        """Subscribe to a channel. Returns a queue that receives messages."""
        q = asyncio.Queue()
        self._channels[channel].append(q)
        return q

    async def publish(self, channel: str, message: dict) -> None:
        """Publish a message to a channel. All subscribers receive it."""
        envelope = {
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": message,
        }
        self._log.append(envelope)
        print(f"[BUS] {message.get('from', '?')} → {channel}: {str(message.get('content', ''))[:60]}...")
        for q in self._channels.get(channel, []):
            await q.put(envelope)

    def get_log(self) -> list[dict]:
        return list(self._log)


# Global singleton bus
bus = MockMessageBus()


# ── Convenience helpers used by agents ──────────────────────────────────────

async def send_to_agent(from_agent: str, to_agent: str, content: str, task_id: str = None) -> None:
    """Send a peer-to-peer message from one agent to another."""
    await bus.publish(
        channel=f"agent.{to_agent}",
        message={
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "task_id": task_id,
            "type": "p2p",
        },
    )


async def broadcast(from_agent: str, content: str, task_id: str = None) -> None:
    """Broadcast a message to the orchestrator / user-facing channel."""
    await bus.publish(
        channel="orchestrator.inbox",
        message={
            "from": from_agent,
            "to": "orchestrator",
            "content": content,
            "task_id": task_id,
            "type": "broadcast",
        },
    )
