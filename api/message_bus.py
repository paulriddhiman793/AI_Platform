"""
Redis-backed message bus for multi-instance deployments.
Falls back to in-memory if Redis is unavailable (dev mode).
"""
import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import redis.asyncio as redis
from redis.exceptions import RedisError

from api.config import settings


class InMemoryMessageBus:
    """Fallback in-memory message bus for development."""

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

        if channel != "p2p.monitor":
            preview = str(message.get("content", ""))[:60]
            print(f"[BUS] {message.get('from', '?')} -> {message.get('to', channel)} | {preview}")

        for q in self._subscribers.get(channel, []):
            await q.put(envelope)

    def subscriber_count(self, channel: str) -> int:
        return len(self._subscribers.get(channel, []))

    def get_log(self) -> list[dict]:
        return list(self._log)

    def clear_log(self) -> None:
        self._log.clear()


class RedisMessageBus:
    """Redis-backed message bus for production."""

    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._redis_subscribed_channels: set[str] = set()
        self._redis_pending_channels: set[str] = set()
        self._listen_task: Optional[asyncio.Task] = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Redis. Returns True on success."""
        try:
            self._redis = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            self._pubsub = self._redis.pubsub()
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._connected = True
            print(f"[BUS] Connected to Redis at {self._redis_url}")
            return True
        except RedisError as e:
            print(f"[BUS] Failed to connect to Redis: {e}. Falling back to in-memory bus.")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        if self._listen_task:
            self._listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listen_task
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        self._connected = False

    async def _listen_loop(self) -> None:
        """Listen for messages on subscribed channels and forward to local queues."""
        while self._connected:
            try:
                if not self._pubsub:
                    await asyncio.sleep(0.5)
                    continue
                if self._redis_pending_channels:
                    to_sub = list(self._redis_pending_channels)
                    self._redis_pending_channels.clear()
                    try:
                        await self._pubsub.subscribe(*to_sub)
                        self._redis_subscribed_channels.update(to_sub)
                    except Exception as e:
                        print(f"[BUS] Batch subscribe error: {e}")
                        self._redis_pending_channels.update(to_sub)

                message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                if message and message.get("type") == "message":
                    channel = message["channel"]
                    try:
                        data = json.loads(message["data"])
                    except Exception:
                        continue
                    # Skip if published by this exact process instance since publish() already delivered locally
                    if data.get("publisher_pid") == os.getpid() and data.get("instance_id") == id(self):
                        continue
                    for q in self._subscribers.get(channel, []):
                        await q.put(data)
                else:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[BUS] Listen loop error (reconnecting/retrying): {e}")
                await asyncio.sleep(1.0)

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[channel].append(q)
        if self._connected and self._pubsub and channel not in self._redis_subscribed_channels:
            self._redis_pending_channels.add(channel)
        return q

    async def publish(self, channel: str, message: dict) -> None:
        envelope = {
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": message,
            "publisher_pid": os.getpid(),
            "instance_id": id(self),
        }

        if channel != "p2p.monitor":
            preview = str(message.get("content", ""))[:60]
            print(f"[BUS] {message.get('from', '?')} -> {message.get('to', channel)} | {preview}")

        # Always deliver immediately to local subscribers in the exact same process
        for q in self._subscribers.get(channel, []):
            await q.put(envelope)

        # Publish to Redis for remote subscribers across other instances
        if self._connected and self._redis:
            try:
                await self._redis.publish(channel, json.dumps(envelope))
            except Exception as exc:
                print(f"[BUS] Redis publish error on {channel}: {exc}")

    def subscriber_count(self, channel: str) -> int:
        return len(self._subscribers.get(channel, []))

    @property
    def is_connected(self) -> bool:
        return self._connected


class ProxyMessageBus:
    """Wrapper that delegates all calls to the active underlying bus implementation (InMemory or Redis)."""

    def __init__(self):
        self._impl: InMemoryMessageBus | RedisMessageBus = InMemoryMessageBus()

    def set_impl(self, impl: InMemoryMessageBus | RedisMessageBus) -> None:
        old_impl = self._impl
        self._impl = impl
        # Transfer any existing channel subscribers from old_impl to new impl
        if hasattr(old_impl, "_subscribers"):
            for channel, queues in old_impl._subscribers.items():
                for q in queues:
                    self._impl._subscribers[channel].append(q)
                    if isinstance(self._impl, RedisMessageBus) and self._impl._connected and self._impl._pubsub and channel not in self._impl._redis_subscribed_channels:
                        self._impl._redis_pending_channels.add(channel)

    def subscribe(self, channel: str) -> asyncio.Queue:
        return self._impl.subscribe(channel)

    async def publish(self, channel: str, message: dict) -> None:
        await self._impl.publish(channel, message)

    def subscriber_count(self, channel: str) -> int:
        return self._impl.subscriber_count(channel)

    @property
    def is_connected(self) -> bool:
        if isinstance(self._impl, RedisMessageBus):
            return self._impl.is_connected
        return True

    async def disconnect(self) -> None:
        if isinstance(self._impl, RedisMessageBus):
            await self._impl.disconnect()


# Global bus instance - initialized in main.py
bus: ProxyMessageBus = ProxyMessageBus()
_recent_p2p: dict[tuple[str, str, str], float] = {}


async def init_message_bus() -> None:
    """Initialize the message bus (Redis or in-memory fallback)."""
    redis_bus = RedisMessageBus(settings.redis_url)
    if await redis_bus.connect():
        bus.set_impl(redis_bus)
        print(f"[MESSAGE BUS] ✅ Active message bus switched to Redis at {settings.redis_url}")
        return
    # Fallback to in-memory
    bus.set_impl(InMemoryMessageBus())
    print("[BUS] Using in-memory message bus (development mode or Redis unavailable)")


async def close_message_bus() -> None:
    """Close the message bus connections."""
    await bus.disconnect()


def _dedup_p2p(from_agent: str, to_agent: str, content: str, window_s: float = 1.5) -> bool:
    """Return True if duplicate within window; otherwise record and return False."""
    key = (from_agent, to_agent, content)
    now_ts = time.time()
    last_ts = _recent_p2p.get(key, 0.0)
    if now_ts - last_ts < window_s:
        return True
    _recent_p2p[key] = now_ts
    if len(_recent_p2p) > 500:
        _recent_p2p.pop(next(iter(_recent_p2p)))
    return False


async def send_to_agent(
    from_agent: str,
    to_agent: str,
    content: str,
    task_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Send a p2p message to an agent inbox and copy it to p2p.monitor for GUI activity feed.
    """
    if _dedup_p2p(from_agent, to_agent, content):
        return
    msg = {
        "from": from_agent,
        "to": to_agent,
        "content": content,
        "task_id": task_id,
        "type": "p2p",
    }
    if extra:
        msg.update(extra)
    await bus.publish(f"agent.{to_agent}", msg)
    await bus.publish("p2p.monitor", msg)


async def broadcast(
    from_agent: str,
    content: str,
    task_id: Optional[str] = None,
) -> None:
    """Broadcast a result to orchestrator inbox (shown in team/user flow by server)."""
    await bus.publish("orchestrator.inbox", {
        "from": from_agent,
        "to": "orchestrator",
        "content": content,
        "task_id": task_id,
        "type": "broadcast",
    })


async def send_to_user(
    from_agent: str,
    content: str,
    task_id: Optional[str] = None,
) -> None:
    await bus.publish("user.output", {
        "from": from_agent,
        "to": "user",
        "content": content,
        "task_id": task_id,
        "type": "user_message",
    })


async def publish_status(agent_id: str, status: str) -> None:
    await bus.publish("agent.status", {
        "agent_id": agent_id,
        "status": status,
        "from": agent_id,
        "to": "gui",
    })


async def publish_file_event(
    agent_id: str,
    filename: str,
    full_path: str,
    task_id: Optional[str] = None,
) -> None:
    await bus.publish("workspace.files", {
        "agent_id": agent_id,
        "filename": filename,
        "path": full_path,
        "task_id": task_id,
        "from": agent_id,
        "to": "gui",
        "content": f"File written: {filename}",
    })