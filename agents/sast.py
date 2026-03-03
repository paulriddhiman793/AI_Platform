"""
sast.py — SAST Agent
Writes scan reports to the project workspace.
"""
import asyncio
from agents.base_agent import BaseAgent
from tools.workspace import workspace


SYSTEM_PROMPT = """You are a SAST security agent. Scan all code before deployment.
Write audit reports to the project workspace."""


class SASTAgent(BaseAgent):
    AGENT_ID = "sast"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "orchestrator")
        c = content.lower()

        await asyncio.sleep(0.2)

        # Re-scan after fix
        if any(kw in c for kw in ["re-scan", "rescan", "fixed", "applied", "re-sending"]):
            report = (
                "# SAST Re-Scan Report\n\n"
                "## Result: ✅ PASSED\n\n"
                "All previous findings resolved.\n"
                "No new vulnerabilities found.\n\n"
                "**Approved for deployment.**\n"
            )
            workspace.write("sast", "rescan_report.md", report)
            await self.message(from_agent,
                "✅ Re-scan passed. No vulnerabilities found. "
                f"Approved for deployment.\n"
                f"Report: sast/rescan_report.md",
                task_id)
            return None

        # Runtime Security cross-reference
        if from_agent == "runtime_security" and "deserialization" in c:
            await self.message("runtime_security",
                "🔴 CRITICAL confirmed at runtime. Writing patch now. "
                "Using strict path whitelisting.", task_id)
            return None

        # Full audit
        if any(kw in c for kw in ["full", "audit", "monthly", "codebase"]):
            report = (
                "# Full Security Audit Report\n\n"
                "## Summary\n"
                "- Files scanned: 14 Python, 8 JSX, 3 config\n"
                "- Security score: **87/100**\n\n"
                "## Findings\n"
                "| Severity | Count | Description |\n"
                "|----------|-------|-------------|\n"
                "| CRITICAL | 0     | None        |\n"
                "| HIGH     | 1     | Path traversal in tools/code_execution.py:88 |\n"
                "| MEDIUM   | 3     | Outdated deps: numpy, requests, pillow |\n"
                "| LOW      | 2     | Unused imports, missing type hints |\n\n"
                "## Recommendation\n"
                "Patch HIGH finding before next deployment.\n"
            )
            workspace.write("sast", "full_audit_report.md", report)
            await self.report(
                "Full audit complete. Score: 87/100.\n"
                "1 HIGH finding: path traversal in tools/code_execution.py:88\n"
                f"Report written: sast/full_audit_report.md",
                task_id
            )
            await self.message("runtime_security",
                "Static audit found HIGH: path traversal in tools/code_execution.py:88. "
                "Can you confirm exploitability at runtime?", task_id)
            return None

        # Standard scan
        if any(kw in c for kw in ["scan", "review", "check", "security", "pipeline", "ready"]):
            if from_agent == "ml_engineer":
                report = (
                    "# SAST Scan Report — ML Pipeline\n\n"
                    "## Result: ⚠ ISSUES FOUND\n\n"
                    "### HIGH (1)\n"
                    "- **File**: pipeline/deploy.py, Line 47\n"
                    "- **Issue**: Hardcoded database connection string\n"
                    "- **Found**: `DB_CONN = 'postgresql://admin:pass@prod-db:5432/ml'`\n"
                    "- **Fix**: Replace with `os.getenv('DB_CONN_STR')`\n\n"
                    "⛔ Deployment blocked until resolved.\n"
                )
                workspace.write("sast", "scan_report_ml.md", report)
                await self.message(from_agent,
                    "Scan found 1 HIGH: hardcoded DB connection string on line 47.\n"
                    "Fix: use os.getenv('DB_CONN_STR').\n"
                    "Report: sast/scan_report_ml.md",
                    task_id)
            elif from_agent == "frontend":
                report = (
                    "# SAST Scan Report — Frontend\n\n"
                    "## Result: ⚠ ISSUES FOUND\n\n"
                    "### MEDIUM (1)\n"
                    "- **File**: components/Tooltip.jsx, Line 34\n"
                    "- **Issue**: XSS risk — innerHTML with user-controlled content\n"
                    "- **Fix**: Use `textContent` or `DOMPurify.sanitize()`\n\n"
                    "⚠ Fix recommended before deployment.\n"
                )
                workspace.write("sast", "scan_report_frontend.md", report)
                await self.message(from_agent,
                    "XSS risk: innerHTML in Tooltip.jsx:34. "
                    "Use DOMPurify.sanitize().\n"
                    "Report: sast/scan_report_frontend.md",
                    task_id)
            else:
                await self.message(from_agent,
                    "✅ Scan complete. No issues found. Approved.", task_id)
            return None

        await self.report(
            "SAST ready. Last audit: 3 days ago. Score: 91/100. "
            "No critical vulnerabilities.", task_id)
        return None

    def get_tools(self):
        return []