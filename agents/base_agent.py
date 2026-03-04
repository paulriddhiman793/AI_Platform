"""
base_agent.py — Shared base class for all 6 platform agents.

Every agent inherits from BaseAgent and overrides:
  - AGENT_ID       : unique string identifier
  - SYSTEM_PROMPT  : role-specific instructions for the LLM
  - handle_task()  : processes an incoming task payload
  - get_tools()    : returns list of LangChain tools (Phase 2+)

Phase 1: LLM calls are mocked. Real logic is hardcoded in handle_task().
Phase 2: Replace _call_llm() with real Claude API (claude-sonnet-4-6).
Phase 3: Replace MockMessageBus with real Redis pub/sub.
Phase 9: Replace in-memory memory with ChromaDB.
"""
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime

from api.message_bus import bus, send_to_agent, broadcast
from tools.workspace import workspace


class BaseAgent(ABC):
    AGENT_ID: str = "base"
    SYSTEM_PROMPT: str = "You are a helpful AI agent."

    def __init__(self):
        # Subscribe to this agent's private channel on the message bus
        self.inbox: asyncio.Queue = bus.subscribe(f"agent.{self.AGENT_ID}")

        # In-memory key-value store (Phase 9: replace with ChromaDB)
        self._memory: dict[str, list[dict]] = {}

        # Current status
        self.status: str = "idle"  # idle | working | error

        # Dedup: ignore messages with the same (task_id + content) seen recently
        self._seen: set[str] = set()

    # ─── Main event loop ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start listening for tasks. Call this in asyncio.gather() at startup."""
        print(f"[{self.AGENT_ID}] Online and listening on agent.{self.AGENT_ID}")
        while True:
            envelope = await self.inbox.get()
            asyncio.create_task(self._dispatch(envelope))

    async def _dispatch(self, envelope: dict) -> None:
        """Dispatch an incoming message envelope to handle_task()."""
        payload = envelope["payload"]

        # ── Deduplication guard ───────────────────────────────────────────────
        # If the bus delivers the same message twice (e.g. two WS connections
        # both forwarding the same user instruction) we drop the duplicate.
        dedup_key = f"{payload.get('task_id','')}:{payload.get('content','')}"
        if dedup_key in self._seen:
            print(f"[{self.AGENT_ID}] Duplicate message dropped: {dedup_key[:60]}")
            return
        self._seen.add(dedup_key)
        # Keep the set from growing unbounded
        if len(self._seen) > 200:
            self._seen.pop()

        self.status = "working"
        try:
            print(f"[{self.AGENT_ID}] Received task from {payload.get('from', '?')}: "
                  f"{str(payload.get('content', ''))[:60]}...")
            result = await self.handle_task(payload)
            if result:
                await self.report(result, task_id=payload.get("task_id"))
        except Exception as e:
            print(f"[{self.AGENT_ID}] ERROR in handle_task: {e}")
            self.status = "error"
            await self.report(
                f"Error in {self.AGENT_ID}: {str(e)}. "
                "Please check logs or provide clarification.",
                task_id=envelope["payload"].get("task_id")
            )
        finally:
            if self.status != "error":
                self.status = "idle"

    # ─── Override in subclasses ───────────────────────────────────────────────

    @abstractmethod
    async def handle_task(self, payload: dict) -> str | None:
        """
        Process an incoming task. Return a string to automatically broadcast
        it as a report, or return None if you're sending messages manually.

        payload keys:
          - from      : sender agent id
          - to        : this agent's id
          - content   : the instruction or message text
          - task_id   : shared task identifier (use for threading)
          - type      : "p2p" | "broadcast"
        """
        ...

    def get_tools(self) -> list:
        """
        Return a list of LangChain tools this agent can use.
        Phase 2+: override this in each subclass.
        """
        return []

    # ─── Communication helpers ────────────────────────────────────────────────

    async def message(self, to_agent: str, content: str, task_id: str = None) -> None:
        """Send a peer-to-peer message directly to another agent."""
        print(f"[{self.AGENT_ID}] → [{to_agent}]: {content[:60]}...")
        await send_to_agent(
            from_agent=self.AGENT_ID,
            to_agent=to_agent,
            content=content,
            task_id=task_id
        )

    async def report(self, content: str, task_id: str = None) -> None:
        """Broadcast a result or update to the orchestrator / user channel."""
        print(f"[{self.AGENT_ID}] REPORT: {content[:60]}...")
        await broadcast(
            from_agent=self.AGENT_ID,
            content=content,
            task_id=task_id
        )

    # ─── LLM integration (Phase 1 = mock, Phase 2 = real) ────────────────────

    async def _call_llm(self, user_message: str, system_prompt: str = None) -> str:
        """
        Call the LLM with a user message and optional system override.

        PHASE 1 (current): Returns a mock response after a short delay.

        PHASE 2 — replace body with:
        ─────────────────────────────────────────────────────────────
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_prompt or self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.content[0].text
        ─────────────────────────────────────────────────────────────

        PHASE 2 with tool use:
        ─────────────────────────────────────────────────────────────
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_prompt or self.SYSTEM_PROMPT,
            tools=self.get_tools(),
            messages=[{"role": "user", "content": user_message}]
        )
        # Handle tool_use blocks in response.content
        ─────────────────────────────────────────────────────────────
        """
        await asyncio.sleep(0.3)  # Simulate network latency
        return f"[MOCK LLM] {self.AGENT_ID} processed: {user_message[:80]}"

    async def _call_llm_with_history(self, history: list[dict]) -> str:
        """
        Multi-turn LLM call with conversation history.

        history format: [{"role": "user"|"assistant", "content": "..."}]

        PHASE 2 — replace body with:
        ─────────────────────────────────────────────────────────────
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=self.SYSTEM_PROMPT,
            messages=history
        )
        return response.content[0].text
        ─────────────────────────────────────────────────────────────
        """
        await asyncio.sleep(0.3)
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"[MOCK LLM] {self.AGENT_ID} responded to: {last_user[:60]}"

    # ─── Self-healing loop (used by ML Engineer + others) ────────────────────

    async def _run_with_healing(
        self,
        code: str,
        max_retries: int = 5,
        task_id: str = None
    ) -> dict:
        """
        Execute code with an autonomous fix loop.

        PHASE 1: Always succeeds (mock).
        PHASE 2: Replace execute_in_sandbox() with real E2B call.

        Returns: {"success": bool, "output": str, "attempts": int}
        """
        for attempt in range(1, max_retries + 1):
            result = await self._execute_in_sandbox(code)

            if result["success"]:
                print(f"[{self.AGENT_ID}] Code succeeded on attempt {attempt}")
                return {"success": True, "output": result["output"], "attempts": attempt}

            print(f"[{self.AGENT_ID}] Attempt {attempt} failed: {result['error'][:60]}")
            await self.report(
                f"Attempt {attempt}/{max_retries} failed: {result['error'][:120]}. "
                "Diagnosing and applying fix...",
                task_id
            )

            # Ask LLM to fix the code
            fix_prompt = (
                f"This Python code failed with the following error:\n\n"
                f"ERROR:\n{result['error']}\n\n"
                f"CODE:\n{code}\n\n"
                "Please provide the corrected code only, no explanation."
            )
            code = await self._call_llm(fix_prompt)

        # All retries exhausted
        await self.report(
            f"Could not auto-fix after {max_retries} attempts. "
            "Escalating to user with full context.",
            task_id
        )
        return {"success": False, "output": None, "attempts": max_retries}

    async def _execute_in_sandbox(self, code: str) -> dict:
        """
        Execute code in a sandboxed environment.

        PHASE 1: Always returns success (mock).
        PHASE 2 — replace with:
        ─────────────────────────────────────────────────────────────
        from e2b_code_interpreter import Sandbox
        with Sandbox() as sandbox:
            result = sandbox.run_code(code)
            if result.error:
                return {"success": False, "output": None, "error": str(result.error)}
            return {"success": True, "output": str(result.text), "error": None}
        ─────────────────────────────────────────────────────────────
        """
        await asyncio.sleep(0.2)
        return {"success": True, "output": "Execution successful (mock)", "error": None}

    # ─── Workspace file access ───────────────────────────────────────────────

    def read_file(self, agent_id: str, filename: str) -> str | None:
        """
        Read a file written by any agent in the current project.
        Returns file content as string, or None if not found.
        """
        try:
            return workspace.read(agent_id, filename)
        except Exception:
            return None

    def list_workspace_files(self, agent_id: str = None) -> list[str]:
        """List all files in the workspace (or for a specific agent folder)."""
        try:
            return workspace.list_files(agent_id)
        except Exception:
            return []

    def workspace_context(self, *file_specs: tuple[str, str]) -> str:
        """
        Build a context string from one or more workspace files.

        Usage:
            ctx = self.workspace_context(
                ("ml_engineer", "pipeline.py"),
                ("data_scientist", "eda_report.md"),
                ("sast", "scan_report_ml.md"),
            )

        Returns a formatted string with each file's content, or a note
        if the file doesn't exist yet. Pass this to handle_task logic so
        every response is grounded in the actual files on disk.
        """
        if not workspace.is_initialized:
            return "[workspace not initialized]"

        parts = []
        for agent_id, filename in file_specs:
            content = self.read_file(agent_id, filename)
            if content:
                parts.append(
                    f"=== {agent_id}/{filename} ===\n{content}\n"
                )
            else:
                parts.append(
                    f"=== {agent_id}/{filename} ===\n[file not yet written]\n"
                )
        return "\n".join(parts) if parts else "[no files found]"

    # ─── Memory (Phase 9: replace with ChromaDB) ─────────────────────────────

    def remember(self, key: str, value: str, namespace: str = "default") -> None:
        """
        Store a memory entry.
        Phase 9: replace with ChromaDB upsert.
        """
        if namespace not in self._memory:
            self._memory[namespace] = []
        self._memory[namespace].append({
            "key": key,
            "value": value,
            "ts": datetime.utcnow().isoformat()
        })

    def recall(self, key: str, namespace: str = "default") -> str | None:
        """
        Retrieve the most recent memory for a key.
        Phase 9: replace with ChromaDB similarity search.
        """
        entries = self._memory.get(namespace, [])
        for entry in reversed(entries):
            if entry["key"] == key:
                return entry["value"]
        return None

    def recall_all(self, namespace: str = "default") -> list[dict]:
        """Return all memories in a namespace."""
        return list(self._memory.get(namespace, []))