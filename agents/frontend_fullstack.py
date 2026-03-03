"""
frontend_fullstack.py — Frontend Full Stack Agent

Responsibilities:
  - Build React components, pages, and full dashboards
  - Wire up API integrations from Data Analyst and Backend
  - Run Playwright tests for functionality
  - Run Lighthouse for performance and accessibility audits
  - ALL new/modified frontend code goes through SAST before deploy
  - Maintain accessibility: WCAG 2.1 AA minimum

Phase 1: All logic mocked.
Phase 2: Wire up Playwright, Lighthouse, Vite build tools.
"""
import asyncio
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a Frontend Full Stack Agent on an autonomous AI engineering platform.

Your responsibilities:
1. Build React components with TypeScript, styled with Tailwind CSS.
2. Wire up all data from backend APIs (provided by Data Analyst or Backend agents).
3. Test every component with Playwright before considering it done.
4. Run Lighthouse on every build — minimum scores: Performance 90, Accessibility 100.
5. ALL code goes to SAST for XSS review before deployment. No exceptions.
6. Fix any SAST findings before deploying.

Tech stack:
  - React 18 + TypeScript
  - Tailwind CSS
  - Vite (build tool)
  - Playwright (E2E testing)
  - Lighthouse CI (performance/accessibility)
  - DOMPurify (XSS sanitization — always use for user-facing strings)

Component checklist before sending to SAST:
  ☐ No innerHTML / dangerouslySetInnerHTML without DOMPurify
  ☐ No eval() or new Function()
  ☐ All user inputs sanitized
  ☐ No hardcoded API URLs (use env vars)
  ☐ No console.log() left in production code
  ☐ Loading states and error boundaries implemented
  ☐ Keyboard navigation works (tabIndex, aria-labels)
  ☐ Playwright tests written and passing

Communication:
  - Ask Data Analyst for metric definitions before building dashboards
  - Send all code to SAST: message("sast", "Please review for XSS: ...")
  - Report completion to orchestrator via report()
"""


class FrontendAgent(BaseAgent):
    AGENT_ID = "frontend"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "orchestrator")
        c = content.lower()

        # ── PHASE 1 MOCK RESPONSES ──────────────────────────────────────────

        # SAST approved — deploy
        if from_agent == "sast" and any(kw in c for kw in ["approved", "clean", "pass", "✅"]):
            await self.report(
                "SAST approval received. Deploying...\n"
                "  Vite build: ✅ (bundle: 142KB gzipped)\n"
                "  Playwright: ✅ 24/24 tests passing\n"
                "  Lighthouse: Performance 97 | Accessibility 100 | Best Practices 100\n"
                "  Deployed to production.",
                task_id
            )
            return None

        # SAST found XSS — fix it
        if from_agent == "sast" and any(kw in c for kw in ["xss", "innerhtml", "issue", "found"]):
            await self.report(
                "XSS finding received from SAST. Applying fix:\n"
                "  Before: element.innerHTML = props.label\n"
                "  After:  element.textContent = DOMPurify.sanitize(props.label)\n"
                "Re-submitting for SAST approval.",
                task_id
            )
            await self.message(
                "sast",
                "XSS fixed. DOMPurify applied to all user-facing strings. "
                "Please re-scan before I deploy.",
                task_id
            )
            return None

        # Data Analyst responded with metrics
        if from_agent == "data_analyst":
            await self.report(
                "Metrics confirmed from Data Analyst. Building dashboard components:\n"
                "  - AccuracyTrendChart (LineChart with 30-day window)\n"
                "  - DriftScoreGauge (RadialChart with threshold line)\n"
                "  - LatencyCard (p50 and p95 side by side)\n"
                "  - InferenceVolumeBar (daily bar chart)\n"
                "Wiring up to Data Analyst API endpoints...",
                task_id
            )
            await asyncio.sleep(0.3)
            await self.report(
                "Components built and wired. Playwright tests written (18 tests). "
                "Sending to SAST for XSS review.",
                task_id
            )
            await self.message(
                "sast",
                "Dashboard code ready. 4 React components + API integration. "
                "Please review for XSS before I deploy.",
                task_id
            )
            return None

        # Build dashboard request
        if any(kw in c for kw in ["dashboard", "build", "create"]):
            await self.report(
                "Dashboard task received. Asking Data Analyst for metric definitions first.",
                task_id
            )
            await self.message(
                "data_analyst",
                "Building a performance dashboard. What metrics should I surface, "
                "and what are the threshold values I should highlight?",
                task_id
            )
            return None

        # Fix a bug
        if any(kw in c for kw in ["fix", "bug", "broken", "layout", "style"]):
            await self.report(
                "Running Playwright to reproduce the issue...",
                task_id
            )
            await asyncio.sleep(0.3)
            await self.report(
                "Issue reproduced. Root cause: CSS grid overflow on mobile breakpoint (<768px).\n"
                "Fix: Changed grid-cols-3 to grid-cols-1 md:grid-cols-3.\n"
                "Playwright: ✅ 24/24 tests passing after fix.\n"
                "Sending to SAST for review before deploy.",
                task_id
            )
            await self.message(
                "sast",
                "Bug fix ready: CSS grid overflow + tooltip refactor. "
                "Please review for XSS (especially the tooltip renderer).",
                task_id
            )
            return None

        # Default
        await self.report(
            "Frontend Agent ready.\n"
            "  Current UI: all components live and healthy.\n"
            "  Last Lighthouse: Performance 97 | Accessibility 100.\n"
            "  Last Playwright run: 24/24 tests passing.\n"
            "  No open issues.",
            task_id
        )
        return None

    def get_tools(self):
        return []
        # Phase 2+:
        # from tools.frontend_tools import playwright_tool, lighthouse_tool, vite_build_tool
        # return [playwright_tool, lighthouse_tool, vite_build_tool]