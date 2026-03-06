"""
orchestrator.py - Supervisor orchestrator agent
"""
from agents.base_agent import BaseAgent


ROUTING_TABLE = {
    # ML / model work
    "model": ["ml_engineer", "data_scientist"],
    "train": ["ml_engineer", "data_scientist"],
    "predict": ["ml_engineer"],
    "churn": ["ml_engineer", "data_scientist"],
    "accuracy": ["ml_engineer", "data_scientist"],
    "pipeline": ["ml_engineer"],
    "deploy": ["ml_engineer", "sast", "runtime_security"],
    "retrain": ["ml_engineer", "data_scientist"],

    # Data work
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

    # Frontend
    "frontend": ["ml_engineer", "data_scientist", "frontend"],

    # Security
    "security": ["sast", "runtime_security"],
    "scan": ["sast"],
    "audit": ["sast", "runtime_security"],
    "vulnerability": ["sast"],
    "pentest": ["runtime_security"],
    "owasp": ["runtime_security", "sast"],
    "threat": ["runtime_security"],

    # GitHub workflow
    "github": ["github"],
    "git": ["github"],
    "branch": ["github"],
    "merge": ["github"],
    "repository": ["github"],
    "repo": ["github"],
    "push": ["github"],
}


class OrchestratorAgent(BaseAgent):
    AGENT_ID = "orchestrator"
    SYSTEM_PROMPT = """You are the Orchestrator of a multi-agent AI engineering platform."""

    def __init__(self):
        super().__init__()
        self._active_tasks: dict[str, dict] = {}

    def route(self, user_message: str) -> list[str]:
        msg = user_message.lower()
        assigned = []
        seen = set()

        for keyword, agents in ROUTING_TABLE.items():
            if keyword in msg:
                for agent in agents:
                    if agent not in seen:
                        assigned.append(agent)
                        seen.add(agent)

        return assigned if assigned else ["ml_engineer"]

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "user")

        # Ignore out-of-thread agent chatter so it does not re-route as user work.
        if from_agent != "user" and task_id not in self._active_tasks:
            return None

        # Agent result path
        if from_agent != "user" and task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task["results"].append({"from": from_agent, "content": content})

            assigned = set(task["agents_assigned"])
            reported = {r["from"] for r in task["results"]}

            if assigned.issubset(reported):
                summary = self._summarise(task)
                del self._active_tasks[task_id]
                await self.report(f"Task complete. {summary}", task_id)
                await self.message(
                    "github",
                    "All assigned agents completed. Perform repository sync: "
                    "push main, push per-agent branches, then merge branches to main.",
                    task_id,
                )
            return None

        # New user instruction path
        agents = self.route(content)
        if task_id:
            self._active_tasks[task_id] = {
                "user_message": content,
                "agents_assigned": agents,
                "results": [],
            }

        agent_names = ", ".join(a.replace("_", " ").title() for a in agents)
        await self.report(f"Task received. Assigning: {agent_names}.", task_id)
        for agent_id in agents:
            await self.message(agent_id, content, task_id)
        return None

    def _summarise(self, task: dict) -> str:
        lines = []
        for result in task["results"]:
            agent = result["from"].replace("_", " ").title()
            first_sentence = result["content"].split(".")[0]
            lines.append(f"{agent}: {first_sentence}")
        return " | ".join(lines)

    def get_tools(self):
        return []
