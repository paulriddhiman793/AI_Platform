"""
api/main.py — Platform entry point

Starts:
  1. All 7 Python agents (async tasks on the message bus)
  2. FastAPI + WebSocket server on http://localhost:8000

Usage:
    pip install fastapi uvicorn websockets
    python -m api.main

Then open the React GUI:
    cd gui && npm run dev
    → http://localhost:5173
"""
import asyncio
import sys
from pathlib import Path

import uvicorn

from tools.workspace import workspace
from api.message_bus import bus, publish_status

from agents.orchestrator       import OrchestratorAgent
from agents.ml_engineer        import MLEngineerAgent
from agents.data_scientist     import DataScientistAgent
from agents.data_analyst       import DataAnalystAgent
from agents.frontend_fullstack import FrontendAgent
from agents.sast               import SASTAgent
from agents.runtime_security   import RuntimeSecurityAgent


# ─── Startup config ───────────────────────────────────────────────────────────

def get_startup_config() -> tuple[str, str]:
    print("\n" + "="*60)
    print("  AI Engineering Platform — Backend Server")
    print("="*60)

    # Output path
    while True:
        output_path = input(
            "\nWhere should agents save project files?\n"
            "Enter full path (e.g. D:\\Downloads or /Users/you/projects): "
        ).strip()
        if not output_path:
            print("  ❌ Path cannot be empty.")
            continue
        path = Path(output_path)
        if not path.exists():
            ans = input(f"  Path does not exist. Create it? (y/n): ").strip().lower()
            if ans == "y":
                path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created: {path}")
                break
        else:
            print(f"  ✅ Found: {path}")
            break

    # Project name
    while True:
        project_name = input(
            "\nProject name (e.g. 'churn prediction', 'fraud detection'): "
        ).strip()
        if project_name:
            break
        print("  ❌ Project name cannot be empty.")

    return output_path, project_name


# ─── Run all agents as background tasks ──────────────────────────────────────

async def run_agents(agents: list) -> None:
    """Start all agents concurrently."""
    await asyncio.gather(*[agent.run() for agent in agents])


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    # 1. Get config
    output_path, project_name = get_startup_config()

    # 2. Set up workspace
    workspace.configure(output_path)
    project_root = workspace.new_project(project_name)

    print(f"\n✅ Project directory created:")
    print(f"   {project_root}")

    # 3. Instantiate all agents
    agents = [
        OrchestratorAgent(),
        MLEngineerAgent(),
        DataScientistAgent(),
        DataAnalystAgent(),
        FrontendAgent(),
        SASTAgent(),
        RuntimeSecurityAgent(),
    ]

    # Publish initial idle status for all agents
    for agent in agents:
        await publish_status(agent.AGENT_ID, "idle")

    print("\nStarting all agents...")
    for a in agents:
        print(f"  ✅ {a.AGENT_ID}")

    # 4. Start FastAPI server config
    # Import here so workspace is configured before server starts
    from api.server import app

    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",   # suppress uvicorn noise
    )
    server = uvicorn.Server(config)

    print(f"\n🚀 WebSocket server running at ws://localhost:8000/ws")
    print(f"   Open the React GUI: cd gui && npm run dev")
    print(f"   Then visit: http://localhost:5173\n")
    print("="*60)

    # 5. Run agents + server concurrently
    await asyncio.gather(
        run_agents(agents),
        server.serve(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nPlatform stopped.")