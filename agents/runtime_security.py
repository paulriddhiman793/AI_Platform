"""
runtime_security.py — Runtime Security Agent

Responsibilities:
  - Dynamic application security testing (DAST) using OWASP ZAP
  - Runtime threat detection with Falco (container anomaly detection)
  - Automated pen test on every new deployment
  - Continuous log anomaly monitoring
  - Cross-reference findings with SAST for complete vulnerability picture

Phase 1: All logic mocked.
Phase 2: Wire up OWASP ZAP API, Falco, and custom log analysis.
"""
import asyncio
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a Runtime Security Agent on an autonomous AI engineering platform.

Your responsibilities:
1. Every time a new version is deployed, run an automated pen test. No exceptions.
2. Tools: OWASP ZAP (dynamic scanning), Falco (runtime threat detection), custom log analysis.
3. What to test on every deployment:
   - All authenticated endpoints (check for auth bypass)
   - All input fields (SQL injection, XSS, command injection)
   - File upload endpoints (path traversal, malicious file types)
   - API rate limiting (brute force resistance)
   - SSRF via any URL parameter
   - Debug/admin endpoints left accessible
   - CORS policy correctness
   - JWT/session token security

4. Continuous monitoring (Falco rules):
   - Unexpected outbound network connections
   - Container privilege escalation attempts
   - Unexpected file writes to /etc or /bin
   - Anomalous CPU/memory spikes (cryptominer detection)
   - Unexpected cron job modifications

5. Severity levels (same as SAST):
   CRITICAL → alert user immediately, block all traffic if necessary
   HIGH → alert ML Engineer or Frontend, fix before next deploy
   MEDIUM → document, fix within sprint
   LOW → monthly report

6. Cross-reference with SAST:
   - When SAST flags a static issue, test it dynamically to confirm exploitability
   - Upgrade severity if confirmed exploitable at runtime

Communication:
  - Report findings to orchestrator via report()
  - Cross-reference with SAST: message("sast", ...)
  - Alert ML Engineer for backend issues: message("ml_engineer", ...)
  - Alert Frontend Agent for API/UI issues: message("frontend", ...)
"""

# Mock pen test findings per scenario
MOCK_PEN_TEST = {
    "clean": (
        "Pen test complete. No issues found:\n"
        "  ✅ Authentication: all endpoints require valid token\n"
        "  ✅ Input validation: no injection vectors found\n"
        "  ✅ Rate limiting: working correctly (429 after 100 req/min)\n"
        "  ✅ CORS: only allowed origins accepted\n"
        "  ✅ Debug endpoints: /api/debug disabled in prod\n"
        "  ✅ File upload: type validation enforced\n"
        "Live system secure."
    ),
    "debug_endpoint": (
        "Pen test complete. Found 1 MEDIUM issue:\n"
        "  File: api/main.py (route registration)\n"
        "  Issue: /api/debug endpoint is accessible in production\n"
        "    Response: 200 OK with system info (Python version, env vars)\n"
        "  Fix: Add environment check — only register this route in development\n"
        "  ⚠ MEDIUM: no direct exploit, but information disclosure."
    ),
    "deserialization": (
        "🔴 CRITICAL — Deserialization endpoint confirmed exploitable:\n"
        "  Endpoint: POST /api/ml/load-model\n"
        "  Payload: crafted pickle object with __reduce__ override\n"
        "  Result: achieved RCE (Remote Code Execution) in test environment\n"
        "  Fix: Replace pickle with joblib + signature verification, "
        "or use ONNX format for model serialization\n"
        "  ⛔ CRITICAL: block all deploys until this is patched."
    ),
}


class RuntimeSecurityAgent(BaseAgent):
    AGENT_ID = "runtime_security"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "orchestrator")
        c = content.lower()

        # ── PHASE 1 MOCK RESPONSES ──────────────────────────────────────────

        await asyncio.sleep(0.2)  # Simulate scan startup

        # SAST asking for cross-reference on specific surface
        if from_agent == "sast" and any(kw in c for kw in ["deserialization", "path traversal", "test", "surface", "exploit"]):
            if "deserialization" in c:
                await self.message(
                    "sast",
                    MOCK_PEN_TEST["deserialization"],
                    task_id
                )
                # Also escalate to orchestrator for critical
                await self.report(
                    "🔴 CRITICAL confirmed: deserialization endpoint is exploitable via RCE. "
                    "All deploys blocked. SAST is writing the patch.",
                    task_id
                )
            elif "path traversal" in c:
                await self.message(
                    "sast",
                    "Confirmed — path traversal in code_execution.py:88 is exploitable. "
                    "Achieved directory listing outside sandbox with crafted input. "
                    "Upgrading severity to CRITICAL. Immediate patch required.",
                    task_id
                )
                await self.report(
                    "🔴 CRITICAL confirmed at runtime: path traversal exploitable. "
                    "Coordinate with SAST on patch.",
                    task_id
                )
            else:
                await self.message(
                    "sast",
                    "Runtime test complete on flagged surface. No exploit found at runtime. "
                    "Static finding may be a false positive in this context.",
                    task_id
                )
            return None

        # Full pen test on deployment
        if any(kw in c for kw in ["pen", "test", "deploy", "sweep", "owasp", "zap"]):
            await self.report("OWASP ZAP sweep started. Testing all live endpoints...", task_id)
            await asyncio.sleep(0.4)

            # Determine which result to return based on context
            if "debug" in c:
                result = MOCK_PEN_TEST["debug_endpoint"]
            else:
                result = MOCK_PEN_TEST["clean"]

            await self.report(result, task_id)
            return None

        # Falco / continuous monitoring check
        if any(kw in c for kw in ["monitor", "falco", "threat", "anomaly", "log"]):
            await self.report(
                "Falco runtime monitoring status:\n"
                "  Container activity: normal\n"
                "  Outbound connections: only to known endpoints (MLflow, GitHub, Redis)\n"
                "  Privilege escalation attempts: 0\n"
                "  Unexpected file writes: 0\n"
                "  CPU/memory: within normal bounds\n"
                "  Falco alerts (last 24h): 0\n"
                "Live system: no active threats.",
                task_id
            )
            return None

        # Full security audit — run alongside SAST
        if any(kw in c for kw in ["audit", "full", "codebase"]):
            await self.report(
                "Running OWASP ZAP full sweep in parallel with SAST static audit...",
                task_id
            )
            await asyncio.sleep(0.4)
            await self.report(
                "Dynamic audit complete:\n"
                "  Auth endpoints: ✅ secure\n"
                "  Input validation: ✅ no injection vectors\n"
                "  CORS: ✅ correct policy\n"
                "  Rate limiting: ✅ working\n"
                "  Debug endpoints: /api/debug disabled ✅\n"
                "  API information leakage: one finding — error responses include stack traces\n"
                "  Fix: suppress stack traces in prod error handlers",
                task_id
            )
            return None

        # Default status
        await self.report(
            "Runtime Security Agent ready.\n"
            "  Falco: running, 0 alerts in last 24h.\n"
            "  Last pen test: 2 days ago — clean.\n"
            "  Live system: no active threats.\n"
            "  Continuous monitoring: active on all containers.",
            task_id
        )
        return None

    def get_tools(self):
        return []
        # Phase 2+:
        # from tools.security_tools import owasp_zap_tool, falco_tool, log_analysis_tool
        # return [owasp_zap_tool, falco_tool, log_analysis_tool]