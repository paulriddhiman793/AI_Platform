"""
main.py — Platform entry point

Starts all 6 agents as concurrent asyncio tasks and wires up the
message bus so they can communicate.

Usage:
    python -m api.main

Phase 1: Runs a simple CLI loop for testing agent-to-agent messaging.
Phase 2: Replace CLI loop with FastAPI + WebSocket server.
Phase 6: GUI connects via WebSocket instead of CLI.

Dependencies (install before running):
    pip install anthropic  # for Phase 2 LLM calls
"""
import asyncio
import uuid

from api.message_bus import bus, send_to_agent

from agents.orchestrator       import OrchestratorAgent
from agents.ml_engineer        import MLEngineerAgent
from agents.data_scientist     import DataScientistAgent
from agents.data_analyst       import DataAnalystAgent
from agents.frontend_fullstack import FrontendAgent
from agents.sast               import SASTAgent
from agents.runtime_security   import RuntimeSecurityAgent


async def cli_loop(orchestrator_agent: OrchestratorAgent) -> None:
    """
    Simple CLI to send instructions to the Orchestrator and see responses.
    Phase 6: replaced by the React GUI + FastAPI WebSocket.
    """
    # Subscribe to the user output channel to see all agent responses
    output_q = bus.subscribe("orchestrator.inbox")
    user_q   = bus.subscribe("user.output")

    print("\n" + "="*60)
    print("  AI Engineering Platform — CLI Mode")
    print("  Type an instruction and press Enter.")
    print("  Type 'quit' to exit.")
    print("="*60 + "\n")

    async def print_responses():
        while True:
            try:
                envelope = output_q.get_nowait()
                payload  = envelope["payload"]
                agent    = payload.get("from", "?").replace("_", " ").upper()
                content  = payload.get("content", "")
                print(f"\n[{agent}]\n{content}\n")
            except asyncio.QueueEmpty:
                pass

            try:
                envelope = user_q.get_nowait()
                payload  = envelope["payload"]
                agent    = payload.get("from", "?").replace("_", " ").upper()
                content  = payload.get("content", "")
                print(f"\n[{agent} → YOU]\n{content}\n")
            except asyncio.QueueEmpty:
                pass

            await asyncio.sleep(0.1)

    # Start the response printer
    asyncio.create_task(print_responses())

    loop = asyncio.get_event_loop()

    while True:
        try:
            user_input = await loop.run_in_executor(None, input, "You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Platform shutting down.")
            break

        if not user_input.strip():
            continue

        task_id = str(uuid.uuid4())[:8]

        # Route through the orchestrator
        await send_to_agent(
            from_agent="user",
            to_agent="orchestrator",
            content=user_input.strip(),
            task_id=task_id,
        )

        # Give agents time to process
        await asyncio.sleep(8)


async def main() -> None:
    # Instantiate all agents
    agents = [
        OrchestratorAgent(),
        MLEngineerAgent(),
        DataScientistAgent(),
        DataAnalystAgent(),
        FrontendAgent(),
        SASTAgent(),
        RuntimeSecurityAgent(),
    ]

    print("Starting all agents...")

    # Run all agents + CLI concurrently
    orchestrator = agents[0]
    await asyncio.gather(
        *[agent.run() for agent in agents],
        cli_loop(orchestrator),
    )


if __name__ == "__main__":
    asyncio.run(main())