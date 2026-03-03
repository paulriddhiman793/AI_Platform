"""
orchestrator.py — Supervisor Orchestrator Agent

The Orchestrator is the brain of the platform. It:
  1. Receives all user instructions
  2. Parses intent and breaks instructions into subtasks
  3. Routes each subtask to the correct agent(s)
  4. Coordinates multi-agent sequences (e.g. SAST before deploy)
  5. Collects results and summarises back to the user

Pattern: LangGraph Supervisor — maintains a task graph and decides
which agent node to activate at each step.

Phase 1: Intent routing is keyword-based (no LLM call needed).
Phase 2: Replace route() with a real Claude API intent classification call.
Phase 3: Replace with a proper LangGraph StateGraph supervisor.
"""
import asyncio
from agents.base_agent import BaseAgent
from api.message_bus import bus


# Routing table: keyword → list of agents to assign
ROUTING_TABLE = {
    # ML / model work
    "model":      ["ml_engineer", "data_scientist"],
    "train":      ["ml_engineer", "data_scientist"],
    "predict":    ["ml_engineer"],
    "churn":      ["ml_engineer", "data_scientist"],
    "accuracy":   ["ml_engineer", "data_scientist"],
    "pipeline":   ["ml_engineer"],
    "deploy":     ["ml_engineer", "sast", "runtime_security"],
    "retrain":    ["ml_engineer", "data_scientist"],

    # Data work
    "eda":        ["data_scientist"],
    "analysis":   ["data_scientist", "data_analyst"],
    "dataset":    ["data_scientist"],
    "feature":    ["data_scientist"],
    "experiment": ["data_scientist"],
    "drift":      ["data_scientist", "data_analyst", "ml_engineer"],

    # Reporting
    "report":     ["data_analyst"],
    "dashboard":  ["frontend", "data_analyst"],
    "metrics":    ["data_analyst"],
    "kpi":        ["data_analyst"],
    "monitoring": ["data_analyst", "ml_engineer"],

    # Frontend
    "ui":         ["frontend"],
    "interface":  ["frontend"],
    "component":  ["frontend"],
    "react":      ["frontend"],

    # Security
    "security":   ["sast", "runtime_security"],
    "scan":       ["sast"],
    "audit":      ["sast", "runtime_security"],
    "vulnerability": ["sast"],
    "pentest":    ["runtime_security"],
    "owasp":      ["runtime_security", "sast"],
    "threat":     ["runtime_security"],
}

# Execution order constraints:
# When these agents are both assigned, enforce this sequence.
SEQUENCE_RULES = [
    # SAST must run before deployment
    ({"sast", "ml_engineer"}, "sast_before_deploy"),
    # Runtime Security runs after SAST
    ({"sast", "runtime_security"}, "sast_before_runtime"),
    # Data Scientist runs before ML Engineer for EDA tasks
    ({"data_scientist", "ml_engineer"}, "ds_before_ml"),
]


class OrchestratorAgent(BaseAgent):
    AGENT_ID = "orchestrator"
    SYSTEM_PROMPT = """You are the Orchestrator of a multi-agent AI engineering platform.

Your job:
1. Parse the user's natural language instruction.
2. Identify which agents are needed and in what order.
3. Break the instruction into discrete subtasks, one per agent.
4. Route subtasks to agents in the correct sequence.
5. Monitor completion and summarise results to the user.

Routing rules:
- ML work: ml_engineer (always) + data_scientist (for EDA first)
- Security: sast (static scan) always before deploy; runtime_security for live testing
- Dashboards: frontend + data_analyst together
- Data quality: data_scientist + data_analyst
- Deployments: ml_engineer → sast → runtime_security → deploy

Always confirm receipt to the user before dispatching agents.
Always summarise the final outcome once all agents report back.
"""

    def __init__(self):
        super().__init__()
        # Track in-progress tasks: task_id → {agents_assigned, results_received, user_message}
        self._active_tasks: dict[str, dict] = {}

    def route(self, user_message: str) -> list[str]:
        """
        Determine which agents to assign based on message content.

        Phase 1: Keyword matching.
        Phase 2: Replace with LLM intent classification:
        ─────────────────────────────────────────────────────
        prompt = f"Given this user instruction, list which agents to assign: {user_message}"
        response = await self._call_llm(prompt)
        return parse_agent_list(response)
        ─────────────────────────────────────────────────────
        """
        msg = user_message.lower()
        assigned = []
        seen = set()

        for keyword, agents in ROUTING_TABLE.items():
            if keyword in msg:
                for agent in agents:
                    if agent not in seen:
                        assigned.append(agent)
                        seen.add(agent)

        return assigned if assigned else ["ml_engineer"]  # Safe default

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "user")

        # ── Handle incoming agent results ──────────────────────────────────
        if from_agent != "user" and task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task["results"].append({"from": from_agent, "content": content})

            # Check if all assigned agents have reported back
            assigned = set(task["agents_assigned"])
            reported = {r["from"] for r in task["results"]}

            if assigned.issubset(reported):
                # All done — summarise to user
                summary = self._summarise(task)
                del self._active_tasks[task_id]
                await self.report(f"✅ Task complete. {summary}", task_id)
            return None

        # ── New instruction from user ──────────────────────────────────────
        agents = self.route(content)

        # Store task state
        if task_id:
            self._active_tasks[task_id] = {
                "user_message": content,
                "agents_assigned": agents,
                "results": [],
            }

        # Acknowledge to user
        agent_names = ", ".join(a.replace("_", " ").title() for a in agents)
        await self.report(
            f"Task received. Assigning: {agent_names}.",
            task_id
        )

        # Dispatch to each agent
        for agent_id in agents:
            await self.message(agent_id, content, task_id)

        return None

    def _summarise(self, task: dict) -> str:
        """
        Build a plain-English summary from all agent results.
        Phase 2: Replace with LLM summarisation call.
        """
        lines = []
        for result in task["results"]:
            agent = result["from"].replace("_", " ").title()
            # Take first sentence of each result
            first_sentence = result["content"].split(".")[0]
            lines.append(f"{agent}: {first_sentence}")
        return " | ".join(lines)

    def get_tools(self):
        return []  # Orchestrator uses agent messaging, not external tools