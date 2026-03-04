"""
api/main.py — Platform entry point

Key fix: agents are instantiated ONCE here and listeners are started
explicitly BEFORE uvicorn starts — not via FastAPI's @on_event("startup")
which can fire multiple times.
"""
import asyncio
from pathlib import Path

import uvicorn

from tools.workspace import workspace
from api.message_bus import bus, publish_status
from api.server import app, start_listeners_once

from agents.orchestrator       import OrchestratorAgent
from agents.ml_engineer        import MLEngineerAgent
from agents.data_scientist     import DataScientistAgent
from agents.data_analyst       import DataAnalystAgent
from agents.frontend_fullstack import FrontendAgent
from agents.sast               import SASTAgent
from agents.runtime_security   import RuntimeSecurityAgent


def get_startup_config() -> tuple[str, str]:
    print("\n" + "="*60)
    print("  AI Engineering Platform")
    print("="*60)

    while True:
        output_path = input(
            "\nWhere should agents save project files?\n"
            "Path (e.g. D:\\Downloads): "
        ).strip()
        if not output_path:
            print("  ❌ Cannot be empty.")
            continue
        path = Path(output_path)
        if not path.exists():
            ans = input("  Path doesn't exist. Create it? (y/n): ").strip().lower()
            if ans == "y":
                path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created: {path}")
                break
        else:
            print(f"  ✅ Found: {path}")
            break

    while True:
        project_name = input("\nProject name (e.g. 'fraud detection'): ").strip()
        if project_name:
            break
        print("  ❌ Cannot be empty.")

    return output_path, project_name


async def main() -> None:
    # 1. Config
    output_path, project_name = get_startup_config()

    # 2. Workspace
    workspace.configure(output_path)
    project_root = workspace.new_project(project_name)
    print(f"\n✅ Project: {project_root}")

    # 3. Instantiate agents ONCE
    agents = [
        OrchestratorAgent(),
        MLEngineerAgent(),
        DataScientistAgent(),
        DataAnalystAgent(),
        FrontendAgent(),
        SASTAgent(),
        RuntimeSecurityAgent(),
    ]

    print("\nAgents ready:")
    for a in agents:
        print(f"  ✅ {a.AGENT_ID}")

    # 4. Publish initial status
    for agent in agents:
        await publish_status(agent.AGENT_ID, "idle")

    # 5. Start uvicorn in the background
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",
        # CRITICAL: disable reload — reload spawns a second process
        # which creates a second set of agents and bus subscribers
        reload=False,
    )
    server = uvicorn.Server(config)

    print(f"\n🚀 Server: ws://localhost:8000/ws")
    print(f"   GUI:    cd gui && npm run dev → http://localhost:5173")
    print("="*60 + "\n")

    # 6. Run everything together
    # start_listeners_once() is called inside the running event loop
    # so tasks can be created properly
    async def run_all():
        # Start bus→GUI listeners (guarded — only runs once)
        start_listeners_once()
        # Start all agents + uvicorn concurrently
        await asyncio.gather(
            *[agent.run() for agent in agents],
            server.serve(),
        )

    await run_all()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPlatform stopped.")