"""
base_agent.py — Shared base class for all 6 platform agents.

Every agent inherits from BaseAgent and overrides:
  - AGENT_ID: unique string identifier
  - SYSTEM_PROMPT: role-specific instructions for the LLM
  - get_tools(): list of LangChain/LangGraph tools this agent can call

Phase 1: LLM calls are mocked with proxy responses.
Phase 2+: Replace _call_llm() with real Claude API calls.
"""
import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from api.message_bus import bus, send_to_agent, broadcast


class BaseAgent(ABC):
    AGENT_ID: str = "base"
    SYSTEM_PROMPT: str = "You are a helpful AI agent."

    def __init__(self):
        self.task_queue: asyncio.Queue = bus.subscribe(f"agent.{self.AGENT_ID}")
        self.memory: list[dict] = []  # Phase 9: replace with ChromaDB
        self.status: str = "idle"     # idle | working | error

    # ── Core loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main event loop. Listens for tasks and processes them."""
        print(f"[{self.AGENT_ID}] Agent online, listening...")
        while True:
            envelope = await self.task_queue.get()
            await self._handle(envelope)

    async def _handle(self, envelope: dict) -> None:
        """Dispatch incoming message to handler."""
        self.status = "working"
        try:
            payload = envelope["payload"]
            response = await self.handle_task(payload)
            if response:
                await broadcast(self.AGENT_ID, response, task_id=payload.get("task_id"))
        except Exception as e:
            print(f"[{self.AGENT_ID}] ERROR: {e}")
            self.status = "error"
            await broadcast(self.AGENT_ID, f"Error in {self.AGENT_ID}: {str(e)}")
        finally:
            self.status = "idle"

    # ── Override in subclasses ────────────────────────────────────────────────

    @abstractmethod
    async def handle_task(self, payload: dict) -> str:
        """Process an incoming task. Return the response string."""
        ...

    # ── Communication helpers ─────────────────────────────────────────────────

    async def message(self, to_agent: str, content: str, task_id: str = None) -> None:
        """Send a P2P message to another agent."""
        await send_to_agent(self.AGENT_ID, to_agent, content, task_id)

    async def report(self, content: str, task_id: str = None) -> None:
        """Report findings/completion back to the orchestrator."""
        await broadcast(self.AGENT_ID, content, task_id)

    # ── LLM proxy (Phase 1 — mock) ────────────────────────────────────────────

    async def _call_llm(self, user_message: str) -> str:
        """
        PHASE 1 MOCK: Returns a scripted proxy response.
        Phase 2+: Replace with:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )
            return msg.content[0].text
        """
        await asyncio.sleep(0.5)  # Simulate latency
        return f"[MOCK] {self.AGENT_ID} processed: {user_message[:80]}"

    # ── Memory (Phase 9 stub) ─────────────────────────────────────────────────

    def remember(self, key: str, value: str) -> None:
        """Store a memory. Phase 9: replace with ChromaDB upsert."""
        self.memory.append({"key": key, "value": value, "ts": datetime.utcnow().isoformat()})

    def recall(self, key: str) -> str | None:
        """Retrieve a memory. Phase 9: replace with ChromaDB similarity search."""
        for m in reversed(self.memory):
            if m["key"] == key:
                return m["value"]
        return None
