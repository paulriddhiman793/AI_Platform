"""
tools/workspace.py — Project Workspace Manager

Manages the project directory on disk and notifies the GUI
every time a file is written, edited, or deleted.
"""
import asyncio
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class WorkspaceManager:
    def __init__(self):
        self._output_path: Optional[Path] = None
        self._project_name: Optional[str] = None
        self._project_root: Optional[Path] = None
        self._initialized: bool = False

    # ─── Setup ───────────────────────────────────────────────────────────────

    def configure(self, output_path: str) -> None:
        path = Path(output_path)
        if not path.exists():
            raise ValueError(f"Output path does not exist: {output_path}")
        self._output_path = path
        print(f"[WORKSPACE] Output path: {self._output_path}")

    def new_project(self, project_name: str) -> Path:
        if not self._output_path:
            raise RuntimeError("Call workspace.configure(path) first.")

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in project_name.lower().replace(" ", "_")
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._project_name = project_name
        self._project_root = self._output_path / f"{safe_name}_{timestamp}"

        for subdir in ["ml_engineer", "data_scientist", "data_analyst",
                       "frontend", "sast", "runtime_security", "github", "shared"]:
            (self._project_root / subdir).mkdir(parents=True, exist_ok=True)

        # Write project info file
        (self._project_root / ".project_info").write_text(
            f"Project: {project_name}\n"
            f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Location: {self._project_root}\n",
            encoding="utf-8"
        )

        self._initialized = True
        print(f"[WORKSPACE] Project created: {self._project_root}")
        return self._project_root

    # ─── File Operations ─────────────────────────────────────────────────────

    def write(self, agent_id: str, filename: str, content: str,
              task_id: str = None) -> Path:
        self._check_initialized()
        file_path = self._resolve(agent_id, filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        print(f"[WORKSPACE] ✍  {agent_id}/{filename} ({len(content)} chars)")

        # Fire event to GUI (non-blocking)
        self._fire_file_event(agent_id, filename, str(file_path), task_id)

        return file_path

    def write_bytes(self, agent_id: str, filename: str, data: bytes,
                    task_id: str = None) -> Path:
        self._check_initialized()
        file_path = self._resolve(agent_id, filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        print(f"[WORKSPACE] binary write {agent_id}/{filename} ({len(data)} bytes)")
        self._fire_file_event(agent_id, filename, str(file_path), task_id)
        return file_path

    def read(self, agent_id: str, filename: str) -> str:
        self._check_initialized()
        file_path = self._resolve(agent_id, filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Not found: {agent_id}/{filename}")
        return file_path.read_text(encoding="utf-8")

    def edit(self, agent_id: str, filename: str, old_str: str, new_str: str,
             task_id: str = None) -> Path:
        content = self.read(agent_id, filename)
        if old_str not in content:
            raise ValueError(f"String not found in {agent_id}/{filename}: {old_str[:40]}")
        return self.write(agent_id, filename, content.replace(old_str, new_str, 1), task_id)

    def append(self, agent_id: str, filename: str, content: str,
               task_id: str = None) -> Path:
        self._check_initialized()
        file_path = self._resolve(agent_id, filename)
        existing = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        return self.write(agent_id, filename, existing + "\n" + content, task_id)

    def exists(self, agent_id: str, filename: str) -> bool:
        self._check_initialized()
        return self._resolve(agent_id, filename).exists()

    def delete(self, agent_id: str, filename: str) -> None:
        self._check_initialized()
        fp = self._resolve(agent_id, filename)
        if fp.exists():
            fp.unlink()
            print(f"[WORKSPACE] 🗑  {agent_id}/{filename}")

    def copy_to_shared(self, from_agent: str, filename: str,
                       task_id: str = None) -> Path:
        self._check_initialized()
        src = self._resolve(from_agent, filename)
        dst = self._resolve("shared", filename)
        shutil.copy2(src, dst)
        self._fire_file_event("shared", filename, str(dst), task_id)
        return dst

    def list_files(self, agent_id: str = None) -> list[str]:
        self._check_initialized()
        base = self._project_root / agent_id if agent_id else self._project_root
        if not base.exists():
            return []
        files = []
        for f in base.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                files.append(str(f.relative_to(self._project_root)))
        return sorted(files)

    def print_tree(self) -> None:
        self._check_initialized()
        print(f"\n[WORKSPACE] {self._project_name}")
        print(f"[WORKSPACE] {self._project_root}\n")
        for f in self.list_files():
            depth = f.count(os.sep)
            print(f"  {'  ' * depth}{Path(f).name}")
        print()

    def get_summary(self) -> str:
        if not self._initialized:
            return "No active project."
        files = self.list_files()
        return (
            f"Project: {self._project_name}\n"
            f"Location: {self._project_root}\n"
            f"Files ({len(files)}):\n" +
            "\n".join(f"  {f}" for f in files)
        )

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def project_root(self) -> Optional[Path]:
        return self._project_root

    @property
    def project_name(self) -> Optional[str]:
        return self._project_name

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ─── Internal ────────────────────────────────────────────────────────────

    def _resolve(self, agent_id: str, filename: str) -> Path:
        return self._project_root / agent_id / filename

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("No active project. Call workspace.new_project() first.")

    def _fire_file_event(self, agent_id: str, filename: str,
                         full_path: str, task_id: str = None) -> None:
        """
        Non-blocking: publish a file-written event to the message bus
        so the server can forward it to the GUI.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                from api.message_bus import publish_file_event
                asyncio.create_task(
                    publish_file_event(agent_id, filename, full_path, task_id)
                )
        except Exception:
            pass  # Don't crash if event loop not available


# Global singleton
workspace = WorkspaceManager()
