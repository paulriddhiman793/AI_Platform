"""
orchestrator.py - Supervisor orchestrator agent
"""
from agents.base_agent import BaseAgent
from tools.workspace import workspace


ROUTING_TABLE = {
    # ML / model work
    "model": ["ml_engineer", "data_scientist"],
    "train": ["ml_engineer", "data_scientist"],
    "predict": ["ml_engineer"],
    "churn": ["ml_engineer", "data_scientist"],
    "accuracy": ["ml_engineer", "data_scientist"],
    "pipeline": ["ml_engineer"],
    "deploy": ["ml_engineer"],
    "retrain": ["ml_engineer", "data_scientist"],

    # Data work
    "analyse": ["data_scientist", "data_analyst"],
    "analyze": ["data_scientist", "data_analyst"],
    "eda": ["data_scientist"],
    "analysis": ["data_scientist", "data_analyst"],
    "dataset": ["data_scientist"],
    "feature": ["data_scientist"],
    "experiment": ["data_scientist"],
    "drift": ["data_scientist", "data_analyst", "ml_engineer"],

    # Reporting
    "report": ["data_analyst"],
    "dashboard": ["data_analyst"],
    "metrics": ["data_analyst"],
    "kpi": ["data_analyst"],
    "monitoring": ["data_analyst", "ml_engineer"],

    # Frontend / Security agents removed

    # GitHub workflow
    "github": ["github"],
    "git": ["github"],
    "branch": ["github"],
    "merge": ["github"],
    "repository": ["github"],
    "repo": ["github"],
    "push": ["github"],
}


BUILD_FLOW_KEYWORDS = (
    "model", "train", "predict", "pipeline", "accuracy",
    "churn", "fraud", "classif", "retrain",
)


class OrchestratorAgent(BaseAgent):
    AGENT_ID = "orchestrator"
    SYSTEM_PROMPT = "You are the Orchestrator of a multi-agent AI engineering platform."

    def __init__(self):
        super().__init__()
        self._active_tasks: dict[str, dict] = {}

    def route(self, user_message: str) -> list[str]:
        msg = user_message.lower()
        assigned: list[str] = []
        seen = set()

        for keyword, agents in ROUTING_TABLE.items():
            if keyword in msg:
                for agent in agents:
                    if agent not in seen:
                        assigned.append(agent)
                        seen.add(agent)

        return assigned if assigned else ["ml_engineer"]

    def _is_build_flow(self, message: str) -> bool:
        msg = message.lower()
        if any(k in msg for k in ("security", "audit", "scan", "owasp", "pentest")):
            return False
        if any(k in msg for k in ("github", "repo", "branch", "merge", "push")):
            return False
        return any(k in msg for k in BUILD_FLOW_KEYWORDS)

    @staticmethod
    def _is_training_process_request(message: str) -> bool:
        msg = message.lower()
        return any(
            k in msg
            for k in (
                "training process",
                "training proceeded",
                "how training",
                "model transparency",
                "training explanation",
            )
        )

    def _analysis_ready(self) -> bool:
        if not workspace.project_root:
            return False

        root = workspace.project_root
        da_findings = root / "data_analyst" / "findings.txt"
        ds_dir = root / "data_scientist"
        ds_suggestions = list(ds_dir.glob("*_feature_suggestions.md")) if ds_dir.exists() else []
        data_dir = root / "shared" / "datasets"
        engineered_csvs = (
            list(data_dir.glob("*_engineered.csv")) if data_dir.exists() else []
        )

        if not da_findings.exists():
            return False

        if not ds_suggestions and not engineered_csvs:
            return False

        if data_dir.exists():
            raw_csvs = [p for p in data_dir.glob("*.csv") if not p.name.endswith("_engineered.csv")]
            latest_raw_mtime = max((p.stat().st_mtime for p in raw_csvs), default=None)
            if latest_raw_mtime:
                if da_findings.stat().st_mtime < latest_raw_mtime:
                    return False
                if ds_suggestions and max(p.stat().st_mtime for p in ds_suggestions) < latest_raw_mtime:
                    return False
                if engineered_csvs and max(p.stat().st_mtime for p in engineered_csvs) < latest_raw_mtime:
                    return False

        return True

    @staticmethod
    def _has_issue_signal(text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ("critical", "high", "blocked", "issue", "found", "vulnerability"))

    @staticmethod
    def _has_pass_signal(text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ("approved", "pass", "clean", "no high", "no vulnerability"))

    async def _dispatch_pipeline_phase(self, task: dict, task_id: str | None) -> None:
        phase = task["phase"]
        prompt = task["user_message"]
        extra = {}
        if task.get("auth_token"):
            extra["auth_token"] = task.get("auth_token")
        if task.get("worker_project_path"):
            extra["worker_project_path"] = task.get("worker_project_path")
        if task.get("target_col"):
            extra["target_col"] = task.get("target_col")

        if phase == "data_review":
            task["expected"] = {"data_scientist", "data_analyst"}
            task["reported"] = set()
            await self.message(
                "data_scientist",
                f"{prompt}\nAnalyse data now. Start with EDA and feature recommendations. Write reports first.",
                task_id,
                extra=extra,
            )
            await self.message(
                "data_analyst",
                f"{prompt}\nAnalyse data now. Validate data quality/history and write monitoring report first.",
                task_id,
                extra=extra,
            )
            await self.report(
                "Workflow started: Data Scientist + Data Analyst first. ML Engineer waits for reports.",
                task_id,
            )
            return

        if phase == "ml_build":
            task["expected"] = {"ml_engineer"}
            task["reported"] = set()
            await self.message(
                "ml_engineer",
                f"{prompt}\nUse hybrid RAG from shared/output.txt and reports to train the model.",
                task_id,
                extra=extra,
            )
            await self.report("Phase 2: ML Engineer training with RAG context.", task_id)
            return

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "user")
        auth_token = (payload.get("auth_token") or "").strip()
        worker_project_path = (payload.get("worker_project_path") or "").strip()
        target_col = (payload.get("target_col") or "").strip()

        if from_agent != "user" and task_id not in self._active_tasks:
            return None

        if from_agent != "user" and task_id in self._active_tasks:
            task = self._active_tasks[task_id]

            if task.get("flow") == "pipeline":
                if from_agent in task["expected"]:
                    task["reported"].add(from_agent)
                    task["results"].append({"from": from_agent, "content": content})

                phase = task["phase"]

                if phase == "data_review" and task["expected"].issubset(task["reported"]):
                    task["phase"] = "ml_build"
                    await self._dispatch_pipeline_phase(task, task_id)
                    return None

                if phase == "ml_build" and "ml_engineer" in task["reported"]:
                    summary = self._summarise(task)
                    await self.report(f"Task complete. {summary}", task_id)
                    del self._active_tasks[task_id]
                    return None

                return None

            # Generic non-pipeline path
            task["results"].append({"from": from_agent, "content": content})
            assigned = set(task["agents_assigned"])
            reported = {r["from"] for r in task["results"]}
            if assigned.issubset(reported):
                summary = self._summarise(task)
                del self._active_tasks[task_id]
                await self.report(f"Task complete. {summary}", task_id)
            return None

        # New user instruction
        if self._is_training_process_request(content):
            if task_id:
                self._active_tasks[task_id] = {
                    "flow": "generic",
                    "user_message": content,
                    "agents_assigned": ["ml_engineer"],
                    "results": [],
                    "auth_token": auth_token,
                    "worker_project_path": worker_project_path,
                    "target_col": target_col,
                }
            await self.report(
                "Training-process request received. Assigning: ML Engineer.",
                task_id,
            )
            extra = {"worker_project_path": worker_project_path}
            if auth_token:
                extra["auth_token"] = auth_token
            if target_col:
                extra["target_col"] = target_col
            await self.message("ml_engineer", content, task_id, extra=extra)
            return None

        if "deploy" in content.lower() and ("local" in content.lower() or "locally" in content.lower()):
            if task_id:
                self._active_tasks[task_id] = {
                    "flow": "generic",
                    "user_message": content,
                    "agents_assigned": ["ml_engineer"],
                    "results": [],
                    "auth_token": auth_token,
                    "worker_project_path": worker_project_path,
                    "target_col": target_col,
                }
            await self.report("Local deploy request received. Assigning: ML Engineer.", task_id)
            extra = {"worker_project_path": worker_project_path}
            if auth_token:
                extra["auth_token"] = auth_token
            if target_col:
                extra["target_col"] = target_col
            await self.message("ml_engineer", content, task_id, extra=extra)
            return None

        if self._is_build_flow(content):
            if task_id:
                phase = "ml_build" if self._analysis_ready() else "data_review"
                self._active_tasks[task_id] = {
                    "flow": "pipeline",
                    "phase": phase,
                    "user_message": content,
                    "agents_assigned": ["data_scientist", "data_analyst", "ml_engineer"],
                    "results": [],
                    "auth_token": auth_token,
                    "worker_project_path": worker_project_path,
                    "target_col": target_col,
                }
                if phase == "ml_build":
                    await self.report(
                        "Analysis outputs detected. Skipping data review and starting ML training.",
                        task_id,
                    )
                await self._dispatch_pipeline_phase(self._active_tasks[task_id], task_id)
            return None

        agents = self.route(content)
        if task_id:
            self._active_tasks[task_id] = {
                "flow": "generic",
                "user_message": content,
                "agents_assigned": agents,
                "results": [],
                "auth_token": auth_token,
                "worker_project_path": worker_project_path,
                "target_col": target_col,
            }

        agent_names = ", ".join(a.replace("_", " ").title() for a in agents)
        await self.report(f"Task received. Assigning: {agent_names}.", task_id)
        extra = {"worker_project_path": worker_project_path}
        if auth_token:
            extra["auth_token"] = auth_token
        if target_col:
            extra["target_col"] = target_col
        for agent_id in agents:
            await self.message(agent_id, content, task_id, extra=extra)
        return None

    def _summarise(self, task: dict) -> str:
        lines = []
        for result in task.get("results", []):
            agent = result["from"].replace("_", " ").title()
            first_sentence = result["content"].split(".")[0]
            lines.append(f"{agent}: {first_sentence}")
        return " | ".join(lines) if lines else "No agent reports."

    def get_tools(self):
        return []
