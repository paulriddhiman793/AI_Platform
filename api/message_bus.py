"""
message_bus.py — Agent Message Bus

Phase 1: In-memory async pub/sub (no external dependencies).
Phase 3: Replace MockMessageBus with redis.asyncio pub/sub.

Redis replacement (Phase 3):
─────────────────────────────────────────────────────────────────────
import redis.asyncio as aioredis

redis_client = aioredis.from_url("redis://localhost:6379")

async def publish(channel: str, message: dict) -> None:
    await redis_client.publish(channel, json.dumps(message))

async def subscribe(channel: str) -> asyncio.Queue:
    q = asyncio.Queue()
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)
    async def reader():
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                await q.put({"channel": channel,
                             "payload": json.loads(msg["data"])})
    asyncio.create_task(reader())
    return q
─────────────────────────────────────────────────────────────────────
"""
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import Optional


class MockMessageBus:
    """In-memory async message bus that mimics Redis Pub/Sub."""

    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._log: list[dict] = []

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[channel].append(q)
        return q

    async def publish(self, channel: str, message: dict) -> None:
        envelope = {
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": message,
        }
        self._log.append(envelope)
        preview = str(message.get("content", ""))[:70]
        print(f"[BUS] {message.get('from','?')} → {message.get('to', channel)} | {preview}")
        for q in self._subscribers.get(channel, []):
            await q.put(envelope)

    def get_log(self) -> list[dict]:
        return list(self._log)

    def get_log_for_task(self, task_id: str) -> list[dict]:
        return [e for e in self._log if e["payload"].get("task_id") == task_id]

    def clear_log(self) -> None:
        self._log.clear()


bus = MockMessageBus()


async def send_to_agent(from_agent: str, to_agent: str, content: str,
                         task_id: Optional[str] = None) -> None:
    await bus.publish(
        channel=f"agent.{to_agent}",
        message={"from": from_agent, "to": to_agent,
                 "content": content, "task_id": task_id, "type": "p2p"},
    )


async def broadcast(from_agent: str, content: str,
                    task_id: Optional[str] = None) -> None:
    await bus.publish(
        channel="orchestrator.inbox",
        message={"from": from_agent, "to": "orchestrator",
                 "content": content, "task_id": task_id, "type": "broadcast"},
    )


async def send_to_user(from_agent: str, content: str,
                       task_id: Optional[str] = None) -> None:
    await bus.publish(
        channel="user.output",
        message={"from": from_agent, "to": "user",
                 "content": content, "task_id": task_id, "type": "user_message"},
    )