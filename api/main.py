"""
api/main.py - Platform entry point

Key fix: agents are instantiated ONCE here and listeners are started
explicitly BEFORE uvicorn starts - not via FastAPI startup hooks that can
fire multiple times in reload-style configurations.
"""
import asyncio
import os
import re
import atexit
import sys
from datetime import datetime
from pathlib import Path

import uvicorn

from tools.workspace import workspace
from api.message_bus import publish_status
from api.server import app, start_listeners_once

from agents.orchestrator import OrchestratorAgent
from agents.ml_engineer import MLEngineerAgent
from agents.data_scientist import DataScientistAgent
from agents.data_analyst import DataAnalystAgent
from agents.github_agent import GitHubAgent


GITHUB_HTTP_RE = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+(?:\.git)?/?$")
GITHUB_SSH_RE = re.compile(r"^git@github\.com:[\w.-]+/[\w.-]+(?:\.git)?$")
GITHUB_SSH_URL_RE = re.compile(r"^ssh://git@github\.com/[\w.-]+/[\w.-]+(?:\.git)?/?$")


if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _display_host() -> str:
    public_base = (os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if public_base:
        return public_base
    host = (os.getenv("HOST") or "0.0.0.0").strip()
    port = int(os.getenv("PORT", "8000"))
    if host in ("0.0.0.0", "::"):
        host = "localhost"
    scheme = "https" if port == 443 else "http"
    return f"{scheme}://{host}:{port}"


def _load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env into os.environ if not already set."""
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        print(f"  Warning: could not parse {env_path}: {exc}")


def _is_valid_github_repo_url(url: str) -> bool:
    if not url:
        return False
    return bool(
        GITHUB_HTTP_RE.match(url)
        or GITHUB_SSH_RE.match(url)
        or GITHUB_SSH_URL_RE.match(url)
    )


def _upsert_env_value(env_path: Path, key: str, value: str) -> None:
    """Create or update a KEY=value line in .env."""
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        existing_key = line.split("=", 1)[0].strip()
        if existing_key == key:
            lines[idx] = f"{key}={value}"
            updated = True
            break

    if not updated:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _configure_github_repo_url() -> None:
    env_path = Path(".env")
    _load_env_file(env_path)

    repo_url = (os.getenv("GITHUB_REPO_URL") or "").strip()
    if repo_url and _is_valid_github_repo_url(repo_url):
        print(f"  GitHub remote from env: {repo_url}")
        return

    if repo_url:
        print(f"  Invalid GITHUB_REPO_URL in env: {repo_url}")

    while True:
        entered = input(
            "\nGitHub repo URL for automated pushes\n"
            "(https://github.com/<owner>/<repo>.git or git@github.com:<owner>/<repo>.git): "
        ).strip()
        if not entered:
            print("  Cannot be empty.")
            continue
        if not _is_valid_github_repo_url(entered):
            print("  Invalid GitHub URL format. Please try again.")
            continue

        os.environ["GITHUB_REPO_URL"] = entered
        try:
            _upsert_env_value(env_path, "GITHUB_REPO_URL", entered)
            print(f"  Saved GITHUB_REPO_URL to {env_path}.")
        except Exception as exc:
            print(f"  Warning: could not write {env_path}: {exc}")
        break


def get_startup_config() -> tuple[str, str]:
    print("\n" + "=" * 60)
    print("  AI Engineering Platform")
    print("=" * 60)

    # Platform-only storage (no user-supplied path)
    output_path = (os.getenv("PLATFORM_STORAGE_ROOT") or "").strip()
    if not output_path:
        output_path = str(Path(__file__).resolve().parent.parent / "platform_projects")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"\nStorage root: {output_path}")

    while True:
        project_name = input("\nProject name (e.g. 'fraud detection'): ").strip()
        if project_name:
            break
        print("  Cannot be empty.")

    _configure_github_repo_url()
    return output_path, project_name


async def main() -> None:
    lock_path = Path(".agent_tmp") / "server.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    force_start = (os.getenv("AI_PLATFORM_FORCE_START") or "").strip().lower() in ("1", "true", "yes")
    if lock_path.exists() and not force_start:
        print(f"[INIT] Server lock found at {lock_path}. Another instance may be running. Set AI_PLATFORM_FORCE_START=1 to override.")
        return

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(f"pid={os.getpid()}\n")
            fh.write(f"started={datetime.utcnow().isoformat()}Z\n")
    except FileExistsError:
        if not force_start:
            print(f"[INIT] Server lock exists at {lock_path}. Aborting start.")
            return
        lock_path.write_text(f"pid={os.getpid()}\nstarted={datetime.utcnow().isoformat()}Z\n", encoding="utf-8")

    def _cleanup_lock() -> None:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass

    atexit.register(_cleanup_lock)

    # 1. Config (CLI init is optional; GUI can initialize the project)
    cli_init = (os.getenv("AI_PLATFORM_CLI_INIT") or "").strip().lower() in ("1", "true", "yes")
    if cli_init:
        output_path, project_name = get_startup_config()
        workspace.configure(output_path)
        project_root = workspace.new_project(project_name)
        print(f"\nProject: {project_root}")
    else:
        # Configure platform storage root for GUI-based projects
        output_path = (os.getenv("PLATFORM_STORAGE_ROOT") or "").strip()
        if not output_path:
            output_path = str(Path(__file__).resolve().parent.parent / "platform_projects")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        workspace.configure(output_path)
        print("\n[INIT] CLI init disabled. Waiting for GUI to initialize project.")

    # 3. Instantiate agents ONCE
    agents = [
        OrchestratorAgent(),
        MLEngineerAgent(),
        DataScientistAgent(),
        DataAnalystAgent(),
        GitHubAgent(),
    ]

    print("\nAgents ready:")
    for agent in agents:
        print(f"  {agent.AGENT_ID}")

    # 4. Publish initial status
    for agent in agents:
        await publish_status(agent.AGENT_ID, "idle")

    # 5. Start uvicorn in the background
    port = int(os.getenv("PORT", "8000"))
    host = (os.getenv("HOST") or "0.0.0.0").strip()
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        # disable reload - reload spawns a second process
        reload=False,
    )
    server = uvicorn.Server(config)

    base_url = _display_host()
    ws_url = re.sub(r"^http", "ws", base_url, count=1).rstrip("/") + "/ws"
    print(f"\nServer: {ws_url}")
    print("GUI:    cd gui && npm run dev")
    print("=" * 60 + "\n")

    # 6. Run everything together
    async def run_all():
        # Start bus->GUI listeners (guarded - only runs once)
        start_listeners_once()
        # Start all agents + uvicorn concurrently
        await asyncio.gather(
            *[agent.run() for agent in agents],
            server.serve(),
        )

    try:
        await run_all()
    finally:
        _cleanup_lock()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPlatform stopped.")
