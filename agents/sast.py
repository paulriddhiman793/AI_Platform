"""
sast.py — SAST (Static Application Security Testing) Agent

Responsibilities:
  - Scan all code written by other agents before any GitHub commit
  - Detect: hardcoded secrets, SQL injection, XSS, insecure deserialization,
    SSRF, path traversal, insecure dependencies, OWASP Top 10
  - BLOCK deployment until all HIGH/CRITICAL issues are resolved
  - Monthly full codebase audit with security score report
  - Cross-reference findings with Runtime Security for confirmation

Phase 1: Mocked scan results with realistic scripted findings.
Phase 2: Wire up Semgrep, Bandit, and SonarQube API.
"""
import asyncio
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a SAST (Static Application Security Testing) agent.

Your responsibilities:
1. Scan all code before it reaches GitHub. No exceptions.
2. Tools: Semgrep (pattern matching), Bandit (Python security), SonarQube (code quality).
3. Severity levels:
   - CRITICAL: Exploitable immediately. Block ALL deploys. Alert user directly.
   - HIGH: Must fix before next deploy. Return findings to requesting agent.
   - MEDIUM: Fix within current sprint. Document in security report.
   - LOW: Document and address in monthly review.
4. For CRITICAL/HIGH: return specific findings (file, line, issue, fix recommendation).
5. On clean scan: respond with "✅ Approved. Safe to deploy." exactly.
6. Monthly audit: full codebase scan, generate security score 0-100, detailed report.

What to check every scan:
  - Hardcoded secrets: API keys, passwords, connection strings, tokens
  - SQL injection: string concatenation in queries, unsanitized inputs
  - XSS: innerHTML, dangerouslySetInnerHTML, eval(), unescaped user content
  - Path traversal: unsanitized file paths, os.path without abspath/realpath
  - Insecure deserialization: pickle.loads, yaml.load (use safe_load)
  - Dependency vulnerabilities: known CVEs in requirements.txt
  - SSRF: unvalidated URLs in requests.get/post
  - Insecure randomness: random module for security tokens (use secrets)
  - Debug endpoints left accessible in production code

Communication:
  - Return findings directly to requesting agent
  - For CRITICAL findings, also alert orchestrator immediately
  - For re-scans after fixes: confirm ✅ or return remaining issues
"""

# Mock scan results per requesting agent
MOCK_FINDINGS = {
    "ml_engineer": {
        "first_scan": {
            "severity": "HIGH",
            "finding": (
                "Scan complete. Found 1 HIGH issue:\n"
                "  File: pipeline/deploy.py, Line 47\n"
                "  Issue: Hardcoded database connection string\n"
                "    Found: DB_CONN = 'postgresql://admin:pass123@prod-db:5432/ml'\n"
                "  Fix: Replace with os.getenv('DB_CONN_STR')\n"
                "  ⛔ Blocked: deployment not approved until this is fixed."
            ),
            "rescan": "✅ Re-scan passed. No vulnerabilities found. Approved for deployment.",
        },
        "deploy": {
            "first_scan": (
                "Scan complete. Found 1 MEDIUM issue:\n"
                "  File: requirements.txt, Line 14\n"
                "  Issue: requests==2.28.0 has known CVE-2023-32681 (SSRF via Proxy-Authorization header)\n"
                "  Fix: Upgrade to requests>=2.31.0\n"
                "  ⚠ Medium severity — fix recommended before production deploy."
            ),
            "rescan": "✅ Re-scan passed. Dependency updated. No vulnerabilities found. Approved.",
        }
    },
    "frontend": {
        "first_scan": {
            "severity": "MEDIUM",
            "finding": (
                "Scan complete. Found 1 MEDIUM issue:\n"
                "  File: components/Tooltip.jsx, Line 34\n"
                "  Issue: XSS risk — innerHTML used with user-controlled content\n"
                "    Found: element.innerHTML = props.label\n"
                "  Fix: Use element.textContent = props.label, or wrap with DOMPurify.sanitize()\n"
                "  ⚠ Medium severity — fix before deploy."
            ),
            "rescan": "✅ Re-scan passed. XSS risk resolved. DOMPurify correctly applied. Approved.",
        },
        "dashboard": {
            "first_scan": "✅ Scan complete. No vulnerabilities found. Dashboard code approved for deployment.",
        }
    },
    "data_scientist": {
        "first_scan": {
            "severity": "NONE",
            "finding": "✅ Scan complete. No issues found. Code approved.",
            "rescan": "✅ Re-scan passed. Clean.",
        }
    },
    "runtime_security": {
        "cross_ref": (
            "Cross-referencing static findings with runtime report:\n"
            "  Path traversal in tools/code_execution.py:88 — confirmed exploitable at runtime.\n"
            "  Upgrading severity from HIGH → CRITICAL.\n"
            "  Writing patch: strict path whitelisting with os.path.realpath() + allowlist check.\n"
            "  Staging patch in security/critical-path-fix branch."
        )
    }
}


class SASTAgent(BaseAgent):
    AGENT_ID = "sast"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    # Track which agents have already had a first scan this session
    _scan_state: dict = {}

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "orchestrator")
        c = content.lower()

        # ── PHASE 1 MOCK RESPONSES ──────────────────────────────────────────

        await asyncio.sleep(0.2)  # Simulate scan time

        # Re-scan after fix
        if any(kw in c for kw in ["re-scan", "rescan", "fixed", "applied", "updated", "re-sending"]):
            rescan_result = (
                MOCK_FINDINGS.get(from_agent, {})
                .get("first_scan", {})
                .get("rescan", "✅ Re-scan passed. No issues found. Approved for deployment.")
            )
            await self.message(from_agent, rescan_result, task_id)
            return None

        # Runtime Security sending cross-reference request
        if from_agent == "runtime_security" and any(kw in c for kw in ["cross", "static", "test", "surface"]):
            await self.message(
                "runtime_security",
                MOCK_FINDINGS["runtime_security"]["cross_ref"],
                task_id
            )
            return None

        # Full codebase audit
        if any(kw in c for kw in ["full", "audit", "codebase", "monthly", "entire"]):
            await self.report("Full codebase audit started. Scanning all agent files...", task_id)
            await asyncio.sleep(0.4)
            await self.report(
                "Full audit complete:\n"
                "  Files scanned: 14 Python files, 8 JSX components, 3 config files\n"
                "  CRITICAL: 0\n"
                "  HIGH: 1 — path traversal in tools/code_execution.py:88\n"
                "  MEDIUM: 3 — outdated deps (numpy, requests, pillow)\n"
                "  LOW: 2 — unused imports, missing type hints\n"
                "  Security score: 87/100\n"
                "Looping in Runtime Security to validate HIGH finding at runtime.",
                task_id
            )
            await self.message(
                "runtime_security",
                "Static audit found a HIGH: path traversal in tools/code_execution.py:88. "
                "Can you specifically test this surface to confirm exploitability?",
                task_id
            )
            return None

        # Standard scan from any agent
        if any(kw in c for kw in ["scan", "review", "check", "security"]):
            findings = MOCK_FINDINGS.get(from_agent, {})

            # Determine which scan template to use
            if "deploy" in c and from_agent == "ml_engineer":
                result = findings.get("deploy", {}).get("first_scan",
                    "✅ Scan complete. No issues found. Approved for deployment.")
            elif "dashboard" in c and from_agent == "frontend":
                result = findings.get("dashboard", {}).get("first_scan",
                    "✅ Scan complete. No issues found. Approved.")
            else:
                first_scan = findings.get("first_scan", {})
                if isinstance(first_scan, dict):
                    result = first_scan.get("finding", "✅ Scan complete. No issues found. Approved.")
                else:
                    result = "✅ Scan complete. No issues found. Approved."

            await self.message(from_agent, result, task_id)
            return None

        # Status check
        await self.report(
            "SAST Agent ready.\n"
            "  Last full audit: 3 days ago. Score: 91/100.\n"
            "  Active findings: 0 critical, 0 high.\n"
            "  Monthly audit: scheduled for the 1st.\n"
            "  All deploy scans: up to date.",
            task_id
        )
        return None

    def get_tools(self):
        return []
        # Phase 2+:
        # from tools.security_tools import semgrep_tool, bandit_tool, sonarqube_tool
        # return [semgrep_tool, bandit_tool, sonarqube_tool]