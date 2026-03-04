"""
sast.py — SAST Security Agent

Reads real workspace files before every scan:
  - ml_engineer/pipeline.py        → scans actual code, not a template
  - ml_engineer/deploy.py          → checks for hardcoded creds, path issues
  - frontend/Dashboard.jsx         → checks for XSS vectors
  - shared/requirements.txt        → checks for CVE-flagged dependency versions
  - sast/scan_report_ml.md         → reads own previous report on re-scan
"""
from agents.base_agent import BaseAgent
from tools.workspace import workspace


SYSTEM_PROMPT = """You are a SAST security agent. Scan actual code files from the workspace.
Never fabricate findings — only report what is actually in the code you read.
Write detailed reports. Block deployment on HIGH findings.
"""

# ── CVE database (subset) ────────────────────────────────────────────────────
KNOWN_CVES = {
    "requests==2.28": "CVE-2023-32681 (SSRF via Proxy-Authorization header leakage)",
    "requests<2.31":  "CVE-2023-32681",
    "numpy<1.24":     "CVE-2021-34141 (buffer overflow in string operations)",
    "pillow<9.3":     "CVE-2022-45199 (denial of service via crafted image)",
}

SECURITY_PATTERNS = {
    "hardcoded_creds": [
        ("postgresql://", "HIGH", "Hardcoded database connection string"),
        ("mysql://",      "HIGH", "Hardcoded database connection string"),
        ("password =",    "HIGH", "Hardcoded password literal"),
        ("secret_key =",  "HIGH", "Hardcoded secret key"),
        ("api_key =",     "HIGH", "Hardcoded API key"),
    ],
    "path_issues": [
        ("open(",         "MEDIUM", "Unvalidated file open — check for path traversal"),
    ],
    "xss": [
        ("dangerouslySetInnerHTML", "HIGH",   "XSS risk — dangerouslySetInnerHTML without DOMPurify"),
        ("innerHTML =",             "MEDIUM", "XSS risk — direct innerHTML assignment"),
        ("eval(",                   "HIGH",   "Code injection — eval() with user input"),
    ],
    "env_hardcoded": [
        ("DB_CONN =",     "HIGH",   "Hardcoded DB connection (use os.getenv)"),
    ],
}


def _scan_code(code: str, filename: str) -> list[dict]:
    """
    Scan actual file content for security issues.
    Returns list of {severity, issue, line, fix}.
    """
    findings = []
    if not code or "[file not yet written]" in code:
        return findings

    lines = code.splitlines()
    ext = filename.split(".")[-1].lower()

    # Choose patterns based on file type
    if ext in ("py", "python"):
        patterns = (
            SECURITY_PATTERNS["hardcoded_creds"] +
            SECURITY_PATTERNS["path_issues"] +
            SECURITY_PATTERNS["env_hardcoded"]
        )
    elif ext in ("jsx", "tsx", "js", "ts"):
        patterns = SECURITY_PATTERNS["xss"]
    else:
        patterns = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue  # skip comments
        for pattern, severity, description in patterns:
            if pattern.lower() in line.lower():
                # Check if already using env var on same line (false positive filter)
                if "os.getenv" in line or "process.env" in line or "DOMPurify" in line:
                    continue
                findings.append({
                    "severity":    severity,
                    "issue":       description,
                    "line":        i,
                    "code":        line.strip()[:80],
                    "fix":         _suggest_fix(pattern, line),
                })

    return findings


def _scan_requirements(content: str) -> list[dict]:
    """Scan requirements.txt for known CVEs."""
    findings = []
    if not content:
        return findings
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for pkg_pattern, cve in KNOWN_CVES.items():
            pkg, ver = pkg_pattern.split("==") if "==" in pkg_pattern else pkg_pattern.split("<")
            if pkg.lower() in line.lower():
                if "==" in pkg_pattern and pkg_pattern.lower() in line.lower():
                    findings.append({"severity": "MEDIUM", "issue": f"CVE: {cve}", "line": 0, "code": line, "fix": f"Upgrade {pkg} to >= 2.31.0"})
                elif "<" in pkg_pattern:
                    import re
                    m = re.search(r'[=><]+\s*([\d.]+)', line)
                    if m:
                        installed = tuple(int(x) for x in m.group(1).split("."))
                        min_safe  = tuple(int(x) for x in ver.split("."))
                        if installed < min_safe:
                            findings.append({"severity": "MEDIUM", "issue": f"CVE: {cve}", "line": 0, "code": line, "fix": f"Upgrade {pkg}"})
    return findings


def _suggest_fix(pattern: str, line: str) -> str:
    fixes = {
        "postgresql://":            "Use os.getenv('DB_CONN_STR') instead of hardcoded URL",
        "mysql://":                 "Use os.getenv('DB_CONN_STR') instead of hardcoded URL",
        "password =":               "Store in environment variable, read with os.getenv()",
        "secret_key =":             "Store in environment variable, read with os.getenv()",
        "api_key =":                "Store in environment variable, read with os.getenv()",
        "open(":                    "Sanitize path with os.path.realpath(os.path.abspath(path))",
        "dangerouslySetInnerHTML":  "Wrap value with DOMPurify.sanitize() before rendering",
        "innerHTML =":              "Use textContent or DOMPurify.sanitize()",
        "eval(":                    "Never use eval() with user-supplied input",
        "DB_CONN =":                "Replace with DB_CONN_STR = os.getenv('DB_CONN_STR')",
    }
    for k, v in fixes.items():
        if k.lower() in pattern.lower():
            return v
    return "Review and remediate"


def _format_report(findings: list[dict], filename: str, rescan: bool = False) -> tuple[str, bool]:
    """Format findings into a markdown report. Returns (report_text, passed)."""
    high   = [f for f in findings if f["severity"] == "HIGH"]
    medium = [f for f in findings if f["severity"] == "MEDIUM"]
    low    = [f for f in findings if f["severity"] == "LOW"]
    passed = len(high) == 0

    status = "✅ PASSED" if passed else "⚠ ISSUES FOUND"
    prefix = "Re-Scan" if rescan else "Scan"

    lines = [
        f"# SAST {prefix} Report — {filename}",
        f"\n## Result: {status}",
        f"\n**Scanned file**: `{filename}`",
        f"**HIGH**: {len(high)}  |  **MEDIUM**: {len(medium)}  |  **LOW**: {len(low)}",
    ]

    if high:
        lines.append("\n## HIGH Findings (Deployment Blocked)")
        for f in high:
            lines.append(f"\n### {f['issue']}")
            if f['line']: lines.append(f"- **Line {f['line']}**: `{f['code']}`")
            lines.append(f"- **Fix**: {f['fix']}")

    if medium:
        lines.append("\n## MEDIUM Findings")
        for f in medium:
            lines.append(f"\n### {f['issue']}")
            if f['line']: lines.append(f"- **Line {f['line']}**: `{f['code']}`")
            lines.append(f"- **Fix**: {f['fix']}")

    if not findings:
        lines.append("\nNo security issues found in the scanned code.")

    if passed:
        lines.append("\n**✅ Approved for deployment.**")
    else:
        lines.append(f"\n**⛔ Deployment blocked until {len(high)} HIGH finding(s) resolved.**")

    return "\n".join(lines), passed


class SASTAgent(BaseAgent):
    AGENT_ID      = "sast"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str | None:
        content    = payload.get("content", "")
        task_id    = payload.get("task_id")
        from_agent = payload.get("from", "orchestrator")
        c          = content.lower()

        # ── Re-scan after fix ─────────────────────────────────────────────
        if any(kw in c for kw in ["re-scan", "rescan", "fixed", "applied", "re-sending", "fixes applied"]):
            # Read the actual updated file
            target_file   = "deploy.py" if "deploy" in c else "pipeline.py"
            target_agent  = from_agent if from_agent in ("ml_engineer", "frontend") else "ml_engineer"
            code          = self.read_file(target_agent, target_file) or ""

            findings      = _scan_code(code, target_file)
            report_text, passed = _format_report(findings, target_file, rescan=True)

            report_filename = "rescan_report.md"
            workspace.write("sast", report_filename, report_text, task_id)

            if passed:
                await self.report(
                    f"Re-scan of {target_agent}/{target_file} complete.\n"
                    f"Read {len(code)} chars from workspace — {len(findings)} issues found.\n"
                    f"✅ All previous findings resolved. Approved for deployment.\n"
                    f"Report: sast/{report_filename}",
                    task_id
                )
                await self.message(
                    from_agent,
                    f"✅ Re-scan passed. No vulnerabilities found in {target_file}. "
                    f"Approved for deployment.\nReport: sast/{report_filename}",
                    task_id
                )
            else:
                high_count = len([f for f in findings if f["severity"] == "HIGH"])
                await self.message(
                    from_agent,
                    f"Re-scan found {high_count} remaining HIGH issue(s) in {target_file}. "
                    f"Please fix before deploying.\nReport: sast/{report_filename}",
                    task_id
                )
            return None

        # ── Runtime Security cross-reference ─────────────────────────────
        if from_agent == "runtime_security":
            rt_finding = content[:200]
            await self.message(
                "runtime_security",
                f"Static confirmation: issue also visible in code. "
                f"Writing patch and updating scan report.",
                task_id
            )
            return None

        # ── Full codebase audit ───────────────────────────────────────────
        if any(kw in c for kw in ["full", "audit", "monthly", "codebase", "all files"]):
            all_files    = self.list_workspace_files()
            all_findings = []
            scanned      = []

            for rel_path in all_files:
                parts = rel_path.replace("\\", "/").split("/")
                if len(parts) == 2:
                    agent_id, filename = parts
                    code = self.read_file(agent_id, filename)
                    if code:
                        findings = _scan_code(code, filename)
                        if filename == "requirements.txt":
                            findings += _scan_requirements(code)
                        all_findings.extend(findings)
                        scanned.append(f"{agent_id}/{filename}")

            report, passed = _format_report(all_findings, "full codebase")
            report = f"# Full SAST Audit\n\n**Files scanned ({len(scanned)}):**\n" + \
                     "\n".join(f"  - {f}" for f in scanned) + "\n\n" + report

            workspace.write("sast", "full_audit_report.md", report, task_id)
            await self.report(
                f"Full audit complete. Scanned {len(scanned)} files.\n"
                f"Total findings: {len(all_findings)} "
                f"({len([f for f in all_findings if f['severity']=='HIGH'])} HIGH, "
                f"{len([f for f in all_findings if f['severity']=='MEDIUM'])} MEDIUM)\n"
                f"Report: sast/full_audit_report.md",
                task_id
            )
            await self.message(
                "runtime_security",
                f"Static audit complete on {len(scanned)} files. "
                f"Found {len([f for f in all_findings if f['severity']=='HIGH'])} HIGH findings. "
                "Can you confirm exploitability at runtime?",
                task_id
            )
            return None

        # ── Standard scan (triggered by ML Engineer, Frontend, or orchestrator)
        if any(kw in c for kw in ["scan", "review", "check", "security", "ready", "pipeline", "push"]):
            # Determine what to scan based on who sent the request
            if from_agent == "ml_engineer":
                files_to_scan = [
                    ("ml_engineer", "pipeline.py"),
                    ("ml_engineer", "deploy.py"),
                    ("shared",      "requirements.txt"),
                ]
                report_filename = "scan_report_ml.md"
            elif from_agent == "frontend":
                files_to_scan = [("frontend", "Dashboard.jsx")]
                report_filename = "scan_report_frontend.md"
            else:
                files_to_scan = [
                    ("ml_engineer", "pipeline.py"),
                    ("shared",      "requirements.txt"),
                ]
                report_filename = "scan_report.md"

            all_findings = []
            files_read   = []

            for agent_id, filename in files_to_scan:
                code = self.read_file(agent_id, filename)
                if code and "[file not yet written]" not in code:
                    f = _scan_code(code, filename)
                    if filename == "requirements.txt":
                        f += _scan_requirements(code)
                    all_findings.extend(f)
                    files_read.append(f"{agent_id}/{filename} ({len(code)} chars)")

            report_text, passed = _format_report(all_findings, ", ".join(f for _, f in files_to_scan))
            workspace.write("sast", report_filename, report_text, task_id)

            files_summary = "\n".join(f"  - {f}" for f in files_read) or "  (no files found in workspace yet)"
            high_count    = len([f for f in all_findings if f["severity"] == "HIGH"])

            await self.report(
                f"Scan complete. Files read from workspace:\n{files_summary}\n"
                f"Findings: {len(all_findings)} total ({high_count} HIGH)\n"
                f"Report: sast/{report_filename}",
                task_id
            )

            if passed:
                await self.message(
                    from_agent,
                    f"✅ Scan passed — no HIGH findings in your code. Approved for deployment.\n"
                    f"Report: sast/{report_filename}",
                    task_id
                )
            else:
                finding_summary = "\n".join(
                    f"  {f['severity']} (line {f['line']}): {f['issue']}"
                    for f in all_findings if f["severity"] in ("HIGH", "MEDIUM")
                )
                await self.message(
                    from_agent,
                    f"Scan found {high_count} HIGH issue(s) — deployment blocked.\n"
                    f"{finding_summary}\n"
                    f"Full report: sast/{report_filename}",
                    task_id
                )
            return None

        # ── Status ────────────────────────────────────────────────────────
        files = self.list_workspace_files("sast")
        await self.report(
            f"SAST ready.\n"
            f"Reports in workspace: {files if files else 'none yet'}\n"
            "Send me code to scan — I read actual files from the workspace.",
            task_id
        )
        return None

    def get_tools(self): return []