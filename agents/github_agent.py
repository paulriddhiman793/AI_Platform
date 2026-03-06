"""
github_agent.py - GitHub automation agent

Workflow:
1) Push full project to main when work is ready
2) Push each agent directory to its own branch
3) Merge all agent branches back into main and push
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from agents.base_agent import BaseAgent
from tools.workspace import workspace


SYSTEM_PROMPT = """You are a GitHub automation agent.
You manage repository setup, pushes, branch pushes per agent, and merges to main.
"""

AGENT_FOLDERS = [
    "ml_engineer",
    "data_scientist",
    "data_analyst",
    "frontend",
    "sast",
    "runtime_security",
    "shared",
]


class GitHubAgent(BaseAgent):
    AGENT_ID = "github"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "orchestrator")
        c = content.lower()

        if not workspace.is_initialized or not workspace.project_root:
            await self.report("GitHub sync skipped: workspace is not initialized.", task_id)
            return None

        if from_agent == "orchestrator" or any(
            kw in c for kw in ["github", "git", "push", "branch", "merge", "repo"]
        ):
            await self._sync_repository(task_id)
            return None

        await self.report(
            "GitHub agent ready. Ask me to push, create branches, or merge to main.",
            task_id,
        )
        return None

    async def _sync_repository(self, task_id: str | None) -> None:
        root = workspace.project_root
        assert root is not None

        await self.report(f"GitHub sync started for: {root}", task_id)

        ok, msg = self._ensure_repo(root)
        if not ok:
            await self.report(msg, task_id)
            return

        remote_url = os.getenv("GITHUB_REPO_URL", "").strip()
        if remote_url:
            ok, msg = self._ensure_remote(root, remote_url)
            if not ok:
                await self.report(msg, task_id)
                return
        elif not self._has_remote(root):
            await self.report(
                "No git remote configured. Set GITHUB_REPO_URL env var or add origin manually.",
                task_id,
            )
            return

        pushed_branches: list[str] = []

        ok, msg = self._commit_all(root, "chore: initial project snapshot from agents")
        if not ok:
            await self.report(msg, task_id)
            return

        ok, msg = self._push_branch(root, "main")
        if not ok:
            await self.report(msg, task_id)
            return

        await self.report("Main branch pushed. Creating per-agent branches.", task_id)

        for folder in AGENT_FOLDERS:
            folder_path = root / folder
            if not folder_path.exists():
                continue

            branch = f"agent/{folder}"
            ok, msg = self._create_agent_branch_push(root, folder, branch)
            if ok:
                pushed_branches.append(branch)
                await self.report(f"Pushed {folder} to branch `{branch}`.", task_id)
            else:
                await self.report(f"Branch push skipped for {folder}: {msg}", task_id)

        if not pushed_branches:
            await self.report("No agent branches were pushed.", task_id)
            return

        ok, msg = self._merge_branches_to_main(root, pushed_branches)
        if not ok:
            await self.report(msg, task_id)
            return

        workspace.write(
            "github",
            "sync_report.md",
            self._build_report(root, pushed_branches),
            task_id,
        )

        await self.report(
            "GitHub sync complete: main pushed, per-agent branches pushed, all merged to main.",
            task_id,
        )

    def _run_git(self, root: Path, args: list[str]) -> tuple[bool, str]:
        try:
            cp = subprocess.run(
                ["git", *args],
                cwd=str(root),
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return False, "git executable not found on PATH."
        output = (cp.stdout or cp.stderr or "").strip()
        return cp.returncode == 0, output

    def _ensure_repo(self, root: Path) -> tuple[bool, str]:
        git_dir = root / ".git"
        if not git_dir.exists():
            ok, out = self._run_git(root, ["init", "-b", "main"])
            if not ok:
                ok2, out2 = self._run_git(root, ["init"])
                if not ok2:
                    return False, f"Failed to initialize git repo: {out or out2}"
                self._run_git(root, ["checkout", "-B", "main"])
        else:
            self._run_git(root, ["checkout", "-B", "main"])
        return True, "Repository ready."

    def _has_remote(self, root: Path) -> bool:
        ok, out = self._run_git(root, ["remote"])
        if not ok:
            return False
        remotes = [r.strip() for r in out.splitlines() if r.strip()]
        return "origin" in remotes

    def _ensure_remote(self, root: Path, remote_url: str) -> tuple[bool, str]:
        if self._has_remote(root):
            ok, out = self._run_git(root, ["remote", "set-url", "origin", remote_url])
            if not ok:
                return False, f"Failed to update origin remote: {out}"
            return True, "Origin remote updated."
        ok, out = self._run_git(root, ["remote", "add", "origin", remote_url])
        if not ok:
            return False, f"Failed to add origin remote: {out}"
        return True, "Origin remote added."

    def _commit_all(self, root: Path, message: str) -> tuple[bool, str]:
        ok, out = self._run_git(root, ["add", "-A"])
        if not ok:
            return False, f"git add failed: {out}"

        ok, out = self._run_git(root, ["diff", "--cached", "--quiet"])
        if ok:
            return True, "Nothing new to commit."

        ok, out = self._run_git(root, ["commit", "-m", message])
        if not ok:
            return False, f"git commit failed: {out}"
        return True, "Commit created."

    def _push_branch(self, root: Path, branch: str) -> tuple[bool, str]:
        ok, out = self._run_git(root, ["push", "-u", "origin", branch])
        if not ok:
            return False, f"Push failed for {branch}: {out}"
        return True, out

    def _create_agent_branch_push(self, root: Path, folder: str, branch: str) -> tuple[bool, str]:
        ok, out = self._run_git(root, ["checkout", "-B", branch, "main"])
        if not ok:
            return False, f"checkout failed: {out}"

        ok, out = self._run_git(root, ["add", folder])
        if not ok:
            return False, f"git add failed: {out}"

        self._run_git(root, ["commit", "-m", f"feat({folder}): update agent files"])

        ok, out = self._push_branch(root, branch)
        if not ok:
            return False, out
        return True, "Branch pushed."

    def _merge_branches_to_main(self, root: Path, branches: list[str]) -> tuple[bool, str]:
        ok, out = self._run_git(root, ["checkout", "main"])
        if not ok:
            return False, f"Could not checkout main: {out}"

        for branch in branches:
            ok, out = self._run_git(root, ["merge", "--no-ff", "--no-edit", branch])
            if not ok:
                return False, f"Merge failed for {branch}: {out}"

        ok, out = self._push_branch(root, "main")
        if not ok:
            return False, out

        return True, "Merged and pushed main."

    def _build_report(self, root: Path, branches: list[str]) -> str:
        branch_lines = "\n".join(f"- {b}" for b in branches)
        return (
            "# GitHub Sync Report\n\n"
            f"- Project root: `{root}`\n"
            "- Main push: completed\n"
            "- Agent branch pushes: completed\n"
            "- Merge to main: completed\n\n"
            "## Branches Pushed\n"
            f"{branch_lines}\n"
        )

    def get_tools(self):
        return []
