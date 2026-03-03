"""
main.py — Platform entry point

Asks the user for:
  1. Output directory (e.g. D:/Downloads)
  2. Project name (e.g. "churn prediction model")

Then creates the project folder structure and starts all agents.

Usage:
    python -m api.main
"""
import asyncio
import uuid
from pathlib import Path

from api.message_bus import bus, send_to_agent
from tools.workspace import workspace

from agents.orchestrator       import OrchestratorAgent
from agents.ml_engineer        import MLEngineerAgent
from agents.data_scientist     import DataScientistAgent
from agents.data_analyst       import DataAnalystAgent
from agents.frontend_fullstack import FrontendAgent
from agents.sast               import SASTAgent
from agents.runtime_security   import RuntimeSecurityAgent


# ─── Startup prompt ───────────────────────────────────────────────────────────

def get_startup_config() -> tuple[str, str]:
    """Ask user for output path and project name before starting agents."""

    print("\n" + "="*60)
    print("  AI Engineering Platform")
    print("="*60)

    # Output path
    while True:
        output_path = input(
            "\nWhere should agents save their code?\n"
            "Enter full path (e.g. D:\\Downloads or /Users/you/projects): "
        ).strip()

        if not output_path:
            print("  ❌ Path cannot be empty.")
            continue

        path = Path(output_path)
        if not path.exists():
            create = input(f"  Path does not exist. Create it? (y/n): ").strip().lower()
            if create == "y":
                path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created: {path}")
                break
            else:
                continue
        else:
            print(f"  ✅ Path exists: {path}")
            break

    # Project name
    while True:
        project_name = input(
            "\nProject name (e.g. 'churn prediction', 'fraud detection dashboard'): "
        ).strip()
        if project_name:
            break
        print("  ❌ Project name cannot be empty.")

    return output_path, project_name


# ─── CLI response loop ────────────────────────────────────────────────────────

async def cli_loop() -> None:
    output_q = bus.subscribe("orchestrator.inbox")
    user_q   = bus.subscribe("user.output")

    print("\n" + "="*60)
    print("  CLI Mode — type an instruction and press Enter")
    print("  Type 'files' to see all files written so far")
    print("  Type 'quit' to exit")
    print("="*60 + "\n")

    async def drain_queues():
        while True:
            try:
                while True:
                    envelope = output_q.get_nowait()
                    payload  = envelope["payload"]
                    agent    = payload.get("from", "?").replace("_", " ").upper()
                    content  = payload.get("content", "")
                    print(f"\n[{agent}]\n{content}\n")
            except asyncio.QueueEmpty:
                pass

            try:
                while True:
                    envelope = user_q.get_nowait()
                    payload  = envelope["payload"]
                    agent    = payload.get("from", "?").replace("_", " ").upper()
                    content  = payload.get("content", "")
                    print(f"\n[{agent} → YOU]\n{content}\n")
            except asyncio.QueueEmpty:
                pass

            await asyncio.sleep(0.1)

    asyncio.create_task(drain_queues())
    loop = asyncio.get_event_loop()

    while True:
        try:
            user_input = await loop.run_in_executor(None, input, "You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        cmd = user_input.strip().lower()

        if cmd in ("quit", "exit", "q"):
            print(f"\nProject saved at:\n  {workspace.project_root}")
            print("Platform shutting down.")
            break

        if cmd == "files":
            workspace.print_tree()
            continue

        if not user_input.strip():
            continue

        task_id = str(uuid.uuid4())[:8]
        await send_to_agent(
            from_agent="user",
            to_agent="orchestrator",
            content=user_input.strip(),
            task_id=task_id,
        )
        await asyncio.sleep(10)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    output_path, project_name = get_startup_config()

    workspace.configure(output_path)
    project_root = workspace.new_project(project_name)

    print(f"\n✅ Project directory created:")
    print(f"   {project_root}\n")
    print("Starting all agents...")

    agents = [
        OrchestratorAgent(),
        MLEngineerAgent(),
        DataScientistAgent(),
        DataAnalystAgent(),
        FrontendAgent(),
        SASTAgent(),
        RuntimeSecurityAgent(),
    ]

    await asyncio.gather(
        *[agent.run() for agent in agents],
        cli_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())