#!/usr/bin/env python3
"""
Phase 4 state report: summarizes runs, tasks, and artifacts.
"""

import json
from pathlib import Path
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "shared" / "state"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return items


def main() -> int:
    runs = _read_jsonl(STATE_DIR / "runs.jsonl")
    tasks = _read_jsonl(STATE_DIR / "tasks.jsonl")
    artifacts = _read_jsonl(STATE_DIR / "artifacts.jsonl")

    out = [
        "PHASE 4 STATE REPORT",
        "=" * 60,
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        f"Runs logged: {len(runs)}",
        f"Task events: {len(tasks)}",
        f"Artifacts: {len(artifacts)}",
        "",
    ]

    if runs:
        out.append("Latest runs:")
        for r in runs[-5:]:
            out.append(f"- {r.get('agent_id')} {r.get('event')} task={r.get('task_id')}")
        out.append("")

    if artifacts:
        out.append("Latest artifacts:")
        for a in artifacts[-5:]:
            out.append(f"- {a.get('agent_id')}/{a.get('filename')} task={a.get('task_id')}")
        out.append("")

    report_path = REPO_ROOT / "phase4_state_report.txt"
    report_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

