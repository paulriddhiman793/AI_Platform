"""
tools/workspace.py — Project Workspace Manager

Every new project gets its own timestamped directory at the user-specified
output path. Agents call these functions to read, write, and edit files.

Usage in any agent:
    from tools.workspace import workspace
    workspace.write("ml_engineer", "pipeline.py", code)
    workspace.read("ml_engineer", "pipeline.py")
    workspace.edit("ml_engineer", "pipeline.py", old_str, new_str)
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class WorkspaceManager:
    """
    Manages the project workspace on the user's local machine.

    Directory structure created per project:
    <output_path>/
    └── <project_name>_<timestamp>/
        ├── .project_info          ← project metadata
        ├── ml_engineer/           ← ML Engineer writes here
        │   ├── pipeline.py
        │   ├── train.py
        │   └── deploy.py
        ├── data_scientist/        ← Data Scientist writes here
        │   ├── eda_report.md
        │   └── feature_analysis.py
        ├── data_analyst/          ← Data Analyst writes here
        │   └── monitoring_report.xlsx
        ├── frontend/              ← Frontend Agent writes here
        │   ├── Dashboard.jsx
        │   └── components/
        ├── sast/                  ← SAST writes scan reports here
        │   └── audit_report.md
        ├── runtime_security/      ← Runtime Security writes here
        │   └── pentest_report.md
        └── shared/                ← All agents can read/write here
            ├── requirements.txt
            └── README.md
    """

    def __init__(self):
        self._output_path: Optional[Path] = None
        self._project_name: Optional[str] = None
        self._project_root: Optional[Path] = None
        self._initialized: bool = False

    # ─── Setup ───────────────────────────────────────────────────────────────

    def configure(self, output_path: str) -> None:
        """
        Set the base output directory where projects will be created.
        Called once at platform startup with the user-specified path.

        Example:
            workspace.configure("D:/Downloads")
            workspace.configure("/Users/john/projects")
        """
        path = Path(output_path)
        if not path.exists():
            raise ValueError(
                f"Output path does not exist: {output_path}\n"
                f"Please create the directory first or choose an existing path."
            )
        self._output_path = path
        print(f"[WORKSPACE] Output path set to: {self._output_path}")

    def new_project(self, project_name: str) -> Path:
        """
        Create a new project directory with full agent subdirectory structure.
        Called by the Orchestrator when a new task session begins.

        Returns the project root path.
        """
        if not self._output_path:
            raise RuntimeError(
                "Workspace not configured. Call workspace.configure(path) first."
            )

        # Sanitize project name for use as directory name
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in project_name.lower().replace(" ", "_")
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{safe_name}_{timestamp}"

        self._project_name = project_name
        self._project_root = self._output_path / dir_name

        # Create all agent subdirectories
        subdirs = [
            "ml_engineer",
            "data_scientist",
            "data_analyst",
            "frontend",
            "sast",
            "runtime_security",
            "shared",
        ]
        for subdir in subdirs:
            (self._project_root / subdir).mkdir(parents=True, exist_ok=True)

        # Write project metadata file
        info = (
            f"Project: {project_name}\n"
            f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Directory: {self._project_root}\n"
            f"\nAgent Directories:\n"
            + "\n".join(f"  {s}/" for s in subdirs)
        )
        (self._project_root / ".project_info").write_text(info, encoding="utf-8")

        self._initialized = True
        print(f"[WORKSPACE] New project created: {self._project_root}")
        return self._project_root

    # ─── File Operations ─────────────────────────────────────────────────────

    def write(self, agent_id: str, filename: str, content: str) -> Path:
        """
        Write a file to an agent's directory.
        Creates subdirectories as needed.

        Args:
            agent_id:  e.g. "ml_engineer", "frontend", "shared"
            filename:  e.g. "pipeline.py", "components/Chart.jsx"
            content:   the full file content as a string

        Returns the full path of the written file.
        """
        self._check_initialized()
        file_path = self._resolve(agent_id, filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        print(f"[WORKSPACE] ✍  {agent_id}/{filename} ({len(content)} chars)")
        return file_path

    def read(self, agent_id: str, filename: str) -> str:
        """
        Read a file from an agent's directory.
        Agents use this to read files written by other agents.
        """
        self._check_initialized()
        file_path = self._resolve(agent_id, filename)
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {agent_id}/{filename}\n"
                f"Full path: {file_path}"
            )
        content = file_path.read_text(encoding="utf-8")
        print(f"[WORKSPACE] 📖  {agent_id}/{filename} ({len(content)} chars)")
        return content

    def edit(self, agent_id: str, filename: str, old_str: str, new_str: str) -> Path:
        """
        Edit a file by replacing a specific string.
        Used by agents to apply targeted fixes without rewriting the whole file.

        Raises ValueError if old_str is not found in the file.
        """
        self._check_initialized()
        content = self.read(agent_id, filename)
        if old_str not in content:
            raise ValueError(
                f"String not found in {agent_id}/{filename}:\n"
                f"  Looking for: {old_str[:60]}..."
            )
        updated = content.replace(old_str, new_str, 1)
        file_path = self.write(agent_id, filename, updated)
        print(f"[WORKSPACE] ✏️  Edited {agent_id}/{filename}")
        return file_path

    def append(self, agent_id: str, filename: str, content: str) -> Path:
        """Append content to an existing file (e.g. adding log entries)."""
        self._check_initialized()
        file_path = self._resolve(agent_id, filename)
        if file_path.exists():
            existing = file_path.read_text(encoding="utf-8")
            content = existing + "\n" + content
        return self.write(agent_id, filename, content)

    def exists(self, agent_id: str, filename: str) -> bool:
        """Check if a file exists in an agent's directory."""
        self._check_initialized()
        return self._resolve(agent_id, filename).exists()

    def list_files(self, agent_id: str = None) -> list[str]:
        """
        List all files in an agent's directory, or the entire project if
        agent_id is None.
        """
        self._check_initialized()
        base = self._project_root / agent_id if agent_id else self._project_root
        if not base.exists():
            return []
        files = []
        for f in base.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                files.append(str(f.relative_to(self._project_root)))
        return sorted(files)

    def delete(self, agent_id: str, filename: str) -> None:
        """Delete a file from an agent's directory."""
        self._check_initialized()
        file_path = self._resolve(agent_id, filename)
        if file_path.exists():
            file_path.unlink()
            print(f"[WORKSPACE] 🗑  Deleted {agent_id}/{filename}")

    def copy_to_shared(self, from_agent: str, filename: str) -> Path:
        """
        Copy a file from an agent's directory to shared/.
        Use this when one agent needs another agent to read their output.
        """
        self._check_initialized()
        src = self._resolve(from_agent, filename)
        dst = self._resolve("shared", filename)
        shutil.copy2(src, dst)
        print(f"[WORKSPACE] 📋  Copied {from_agent}/{filename} → shared/{filename}")
        return dst

    # ─── Project info ─────────────────────────────────────────────────────────

    @property
    def project_root(self) -> Optional[Path]:
        return self._project_root

    @property
    def project_name(self) -> Optional[str]:
        return self._project_name

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def print_tree(self) -> None:
        """Print the full project directory tree to the console."""
        self._check_initialized()
        print(f"\n[WORKSPACE] Project: {self._project_name}")
        print(f"[WORKSPACE] Location: {self._project_root}\n")
        files = self.list_files()
        if not files:
            print("  (empty)")
        for f in files:
            depth = f.count(os.sep)
            indent = "  " * depth
            print(f"  {indent}{Path(f).name}")
        print()

    def get_summary(self) -> str:
        """Return a short text summary of the project workspace."""
        if not self._initialized:
            return "No active project."
        files = self.list_files()
        return (
            f"Project: {self._project_name}\n"
            f"Location: {self._project_root}\n"
            f"Files: {len(files)}\n"
            + "\n".join(f"  {f}" for f in files)
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _resolve(self, agent_id: str, filename: str) -> Path:
        """Resolve a filename to its full path inside the project."""
        return self._project_root / agent_id / filename

    def _check_initialized(self) -> None:
        if not self._initialized or not self._project_root:
            raise RuntimeError(
                "No active project. Call workspace.new_project(name) first.\n"
                "Or call workspace.configure(path) then workspace.new_project(name)."
            )


# ── Global singleton — import this everywhere ─────────────────────────────────
workspace = WorkspaceManager()