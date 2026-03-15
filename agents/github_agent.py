"""
github_agent.py - GitHub automation agent

Workflow:
1) Push full project to main when work is ready
2) Push each agent directory to its own branch
3) Merge all agent branches back into main and push
"""
from __future__ import annotations

import base64
import asyncio
import os
import subprocess
from pathlib import Path
import json
import requests

from agents.base_agent import BaseAgent
from tools.workspace import workspace


SYSTEM_PROMPT = """You are a GitHub automation agent.
You manage repository setup, pushes, branch pushes per agent, and merges to main.
"""

AGENT_FOLDERS = [
    "ml_engineer",
    "data_scientist",
    "data_analyst",
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

        ok, msg = await self._ensure_repo(root)
        if not ok:
            await self.report(msg, task_id)
            return

        cfg = self._load_github_config(root)
        remote_url = (os.getenv("GITHUB_REPO_URL") or "").strip()
        if not remote_url:
            remote_url = (cfg.get("repo_url") or "").strip()

        if not remote_url and not await self._has_remote(root):
            created, err = await self._create_repo_if_needed(cfg)
            if not created:
                await self.report(
                    err
                    or "No git remote configured. Use the GUI 'Connect GitHub' first or set GITHUB_REPO_URL.",
                    task_id,
                )
                return
            remote_url = (cfg.get("repo_url") or "").strip()

        if remote_url:
            ok, msg = await self._ensure_remote(root, remote_url)
            if not ok:
                await self.report(msg, task_id)
                return

        pushed_branches: list[str] = []

        ok, msg = await self._commit_all(root, "chore: initial project snapshot from agents")
        if not ok:
            await self.report(msg, task_id)
            return

        ok, msg = await self._push_branch(root, "main", cfg)
        if not ok:
            await self.report(msg, task_id)
            return

        await self.report("Main branch pushed. Creating per-agent branches.", task_id)

        for folder in AGENT_FOLDERS:
            folder_path = root / folder
            if not folder_path.exists():
                continue

            branch = f"agent/{folder}"
            ok, msg = await self._create_agent_branch_push(root, folder, branch, cfg)
            if ok:
                pushed_branches.append(branch)
                await self.report(f"Pushed {folder} to branch `{branch}`.", task_id)
            else:
                await self.report(f"Branch push skipped for {folder}: {msg}", task_id)

        if not pushed_branches:
            await self.report("No agent branches were pushed.", task_id)
            return

        ok, msg = await self._merge_branches_to_main(root, pushed_branches, cfg)
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

    async def _run_git(self, root: Path, args: list[str], cfg: dict | None = None) -> tuple[bool, str]:
        def _run() -> tuple[bool, str]:
            try:
                cmd = ["git", *args]
                token = (cfg or {}).get("token") if cfg else None
                if token and any(a in args for a in ["push", "fetch", "pull", "ls-remote"]):
                    auth = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("utf-8")
                    cmd = ["git", "-c", f"http.extraHeader=AUTHORIZATION: basic {auth}", *args]
                cp = subprocess.run(
                    cmd,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError:
                return False, "git executable not found on PATH."
            output = (cp.stdout or cp.stderr or "").strip()
            return cp.returncode == 0, output
        return await asyncio.to_thread(_run)

    async def _ensure_repo(self, root: Path) -> tuple[bool, str]:
        git_dir = root / ".git"
        if not git_dir.exists():
            ok, out = await self._run_git(root, ["init", "-b", "main"])
            if not ok:
                ok2, out2 = await self._run_git(root, ["init"])
                if not ok2:
                    return False, f"Failed to initialize git repo: {out or out2}"
                await self._run_git(root, ["checkout", "-B", "main"])
        else:
            await self._run_git(root, ["checkout", "-B", "main"])
        return True, "Repository ready."

    async def _has_remote(self, root: Path) -> bool:
        ok, out = await self._run_git(root, ["remote"])
        if not ok:
            return False
        remotes = [r.strip() for r in out.splitlines() if r.strip()]
        return "origin" in remotes

    async def _ensure_remote(self, root: Path, remote_url: str) -> tuple[bool, str]:
        if await self._has_remote(root):
            ok, out = await self._run_git(root, ["remote", "set-url", "origin", remote_url])
            if not ok:
                return False, f"Failed to update origin remote: {out}"
            return True, "Origin remote updated."
        ok, out = await self._run_git(root, ["remote", "add", "origin", remote_url])
        if not ok:
            return False, f"Failed to add origin remote: {out}"
        return True, "Origin remote added."

    async def _commit_all(self, root: Path, message: str) -> tuple[bool, str]:
        ok, out = await self._run_git(root, ["add", "-A"])
        if not ok:
            return False, f"git add failed: {out}"

        ok, out = await self._run_git(root, ["diff", "--cached", "--quiet"])
        if ok:
            return True, "Nothing new to commit."

        ok, out = await self._run_git(root, ["commit", "-m", message])
        if not ok:
            return False, f"git commit failed: {out}"
        return True, "Commit created."

    async def _push_branch(self, root: Path, branch: str, cfg: dict | None = None) -> tuple[bool, str]:
        ok, out = await self._run_git(root, ["push", "-u", "origin", branch], cfg)
        if not ok:
            return False, f"Push failed for {branch}: {out}"
        return True, out

    async def _create_agent_branch_push(self, root: Path, folder: str, branch: str, cfg: dict | None = None) -> tuple[bool, str]:
        ok, out = await self._run_git(root, ["checkout", "-B", branch, "main"])
        if not ok:
            return False, f"checkout failed: {out}"

        ok, out = await self._run_git(root, ["add", folder])
        if not ok:
            return False, f"git add failed: {out}"

        await self._run_git(root, ["commit", "-m", f"feat({folder}): update agent files"])

        ok, out = await self._push_branch(root, branch, cfg)
        if not ok:
            return False, out
        return True, "Branch pushed."

    async def _merge_branches_to_main(self, root: Path, branches: list[str], cfg: dict | None = None) -> tuple[bool, str]:
        ok, out = await self._run_git(root, ["checkout", "main"])
        if not ok:
            return False, f"Could not checkout main: {out}"

        for branch in branches:
            ok, out = await self._run_git(root, ["merge", "--no-ff", "--no-edit", branch])
            if not ok:
                return False, f"Merge failed for {branch}: {out}"

        ok, out = await self._push_branch(root, "main", cfg)
        if not ok:
            return False, out

        return True, "Merged and pushed main."

    def _load_github_config(self, root: Path) -> dict:
        cfg_path = root / "shared" / "github_config.json"
        if not cfg_path.exists():
            return {}
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    async def _create_repo_if_needed(self, cfg: dict) -> tuple[bool, str | None]:
        def _run() -> tuple[bool, str | None]:
            token = (cfg.get("token") or "").strip()
            repo = (cfg.get("repo") or "").strip()
            owner = (cfg.get("owner") or "").strip()
            visibility = (cfg.get("visibility") or "private").strip().lower()
            if not token or not repo:
                return False, "GitHub connect is missing token or repo name."

            is_private = visibility != "public"
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            }
            if owner:
                url = f"https://api.github.com/orgs/{owner}/repos"
                body = {"name": repo, "private": is_private}
                lookup_url = f"https://api.github.com/repos/{owner}/{repo}"
            else:
                url = "https://api.github.com/user/repos"
                body = {"name": repo, "private": is_private}
                lookup_url = None
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=30)
                data = {}
                try:
                    data = resp.json() if resp.content else {}
                except Exception:
                    data = {}
                if resp.status_code in (200, 201):
                    cfg["repo_url"] = data.get("clone_url") or f"https://github.com/{data.get('full_name', '')}.git"
                    if not cfg.get("owner"):
                        cfg["owner"] = (data.get("owner") or {}).get("login", "")
                elif resp.status_code == 422:
                    # Repo may already exist. Try to fetch repo info.
                    try:
                        resolved_owner = owner
                        if not resolved_owner:
                            me = requests.get("https://api.github.com/user", headers=headers, timeout=30)
                            if me.status_code == 200:
                                resolved_owner = (me.json() or {}).get("login", "")
                        lookup = f"https://api.github.com/repos/{resolved_owner}/{repo}" if resolved_owner else None
                        if not lookup:
                            return False, "GitHub repo creation failed (422). Owner could not be resolved."
                        get_resp = requests.get(lookup, headers=headers, timeout=30)
                        if get_resp.status_code == 200:
                            data = get_resp.json()
                            cfg["repo_url"] = data.get("clone_url") or f"https://github.com/{data.get('full_name', '')}.git"
                        else:
                            return False, f"GitHub repo creation failed (422). Also could not read repo: {get_resp.text}"
                    except Exception as e:
                        return False, f"GitHub repo creation failed (422). Repo lookup failed: {e}"
                else:
                    msg = data.get("message") if isinstance(data, dict) else None
                    return False, f"GitHub repo creation failed ({resp.status_code}). {msg or resp.text}"
                try:
                    root = workspace.project_root
                    if root:
                        cfg_path = root / "shared" / "github_config.json"
                        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                except Exception:
                    pass
                return True, None
            except Exception as e:
                return False, f"GitHub repo creation failed: {e}"
        return await asyncio.to_thread(_run)

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
