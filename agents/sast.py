"""
sast.py — SAST (Static Application Security Testing) Agent

Responsibilities:
  - Scan all code written by other agents before it's committed to GitHub
  - Detect: hardcoded secrets, SQL injection, insecure deps, XSS, OWASP Top 10
  - Block deployment until issues are resolved
  - Monthly full codebase audit with security score report

Phase 1: Mocked scan results with scripted findings.
Phase 2: Wire up real Semgrep, Bandit, and SonarQube API.
"""
from agents.base_agent import BaseAgent

SYSTEM_PROMPT = """You are a SAST (Static Application Security Testing) agent on an autonomous AI engineering platform.

Your responsibilities:
1. Every time another agent asks you to scan code, you run a full static analysis.
2. Tools: Semgrep (pattern matching), Bandit (Python security), CodeQL (deep analysis), SonarQube API.
3. Check for: hardcoded secrets/API keys, SQL injection, insecure deserialization, XSS, SSRF, insecure dependencies, OWASP Top 10.
4. If you find issues: return findings to the requesting agent and BLOCK deployment until fixed.
5. If scan is clean: respond with "Approved. Safe to deploy." and allow the pipeline to proceed.
6. On the 1st of each month: perform a full codebase audit. Generate a security score (0-100) and detailed findings report.

Communication:
  - You receive scan requests from: ml_engineer, data_scientist, frontend
  - You respond directly to the requesting agent with findings
  - For critical vulnerabilities, also alert the orchestrator immediately

Severity levels:
  - CRITICAL: Exploitable immediately. Block all deploys. Notify user directly.
  - HIGH: Must fix before next deploy.
  - MEDIUM: Fix within current sprint.
  - LOW: Document and address in monthly review.
"""


class SASTAgent(BaseAgent):
    AGENT_ID = "sast"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str:
        content = payload.get("content", "")
        from_agent = payload.get("from", "unknown")
        task_id = payload.get("task_id")

        # ── PHASE 1 MOCK: Simulate finding a vuln, then approving after fix ──
        if "re-scan" in content.lower() or "fixed" in content.lower():
            # Second scan after fix — approve
            await self.message(from_agent, "Re-scan passed. No vulnerabilities found. ✅ Approved for deployment.", task_id)
            return None

        if "scan" in content.lower() or "review" in content.lower():
            # First scan — return a finding
            finding = self._mock_finding(from_agent)
            await self.message(from_agent, finding, task_id)
            return None

        return f"SAST: unrecognized request from {from_agent}."

    def _mock_finding(self, requesting_agent: str) -> str:
        """Return a scripted security finding based on who's asking."""
        findings = {
            "ml_engineer": (
                "Scan complete. Found 1 HIGH issue:\n"
                "  • Line 47: Hardcoded database connection string. "
                "Replace with os.getenv('DB_CONN_STR').\n"
                "Fix this issue and re-send for approval."
            ),
            "frontend": (
                "Scan complete. Found 1 MEDIUM issue:\n"
                "  • innerHTML used in tooltip renderer — XSS risk. "
                "Use textContent or DOMPurify.\n"
                "Fix and re-send."
            ),
            "data_scientist": (
                "Scan complete. No issues found. ✅ Approved."
            ),
        }
        return findings.get(requesting_agent, "Scan complete. No critical issues found. ✅ Approved.")

    def get_tools(self):
        """Phase 2+: return list of LangChain tools."""
        return []  # TODO: semgrep_tool, bandit_tool, sonarqube_tool
