#!/usr/bin/env python3
"""
Phase 3 execution checks (no LLMs).

Runs hardcoded pipelines in isolated subprocesses, validates outputs against
contracts, and writes a consolidated report for CI-style validation.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tools.phase3_contracts import DA_REQUIRED_KEYS, ML_REQUIRED_KEYS


CODE_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_latest_dataset(target_root: Path, raw_only: bool) -> Path | None:
    ds_dir = target_root / "shared" / "datasets"
    if not ds_dir.exists():
        return None
    candidates = []
    for ext in ("*.csv", "*.xlsx", "*.xls", "*.parquet", "*.json", "*.tsv"):
        candidates.extend(ds_dir.glob(ext))
    if not candidates:
        return None
    if raw_only:
        raw = [p for p in candidates if "engineered" not in p.name.lower()]
        if raw:
            return sorted(raw, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _find_latest_engineered(target_root: Path) -> Path | None:
    ds_dir = target_root / "shared" / "datasets"
    if not ds_dir.exists():
        return None
    candidates = list(ds_dir.glob("*_engineered.csv"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _run_subprocess(cmd: list[str], cwd: Path, timeout_s: int, env_extra: dict | None = None) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT)
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": f"TIMEOUT after {timeout_s}s",
        }


def _extract_block(text: str, begin: str, end: str) -> dict | None:
    if not text or begin not in text or end not in text:
        return None
    blob = text.split(begin, 1)[1].split(end, 1)[0].strip()
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


def _validate_required(payload: dict | None, required: list[str]) -> list[str]:
    if not payload:
        return required[:]
    return [k for k in required if k not in payload]


def _build_da_script(dataset_path: Path) -> str:
    tpl = (CODE_ROOT / "tools" / "da_pipeline_template.py").read_text(encoding="utf-8")
    return tpl.replace("__DATA_PATH__", str(dataset_path).replace("\\", "\\\\"))


def _build_fe_runner(dataset_path: Path) -> str:
    ds = str(dataset_path).replace("\\", "\\\\")
    return f'''import os
import pandas as pd
from fe_1 import run, identify_target_column

df = pd.read_csv(r"{ds}")
target_col = identify_target_column(df, r"{ds}")
run(csv_path=r"{ds}", target_col=target_col)
print("ENGINEERED_BASE:", os.path.splitext(os.path.basename(r"{ds}"))[0])
'''


def _build_ml_script(dataset_path: Path) -> str:
    from agents.ml_engineer import MLEngineerAgent
    agent = MLEngineerAgent()
    return agent._build_ml_optuna_code(Path(dataset_path))


def run_phase3_checks(raw_path: Path | None, eng_path: Path | None, use_sample: bool, target_root: Path) -> dict:
    report: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(CODE_ROOT),
        "project_root": str(target_root),
        "raw_dataset": str(raw_path) if raw_path else None,
        "engineered_dataset": str(eng_path) if eng_path else None,
        "raw_sha256": _sha256(raw_path) if raw_path else None,
        "engineered_sha256": _sha256(eng_path) if eng_path else None,
        "checks": {},
    }

    work_root = target_root / ".agent_tmp" / "phase3"
    work_root.mkdir(parents=True, exist_ok=True)

    # Optional sampling to keep runtime small
    if raw_path and use_sample:
        sample_path = work_root / "sample_raw.csv"
        import pandas as pd
        df = pd.read_csv(raw_path)
        df.head(200).to_csv(sample_path, index=False)
        raw_path = sample_path
        report["raw_dataset"] = str(raw_path)
        report["raw_sha256"] = _sha256(raw_path)

    if eng_path and use_sample:
        # Only sample engineered if it exists; otherwise DS step may create one.
        eng_sample = work_root / "sample_engineered.csv"
        import pandas as pd
        df = pd.read_csv(eng_path)
        df.head(200).to_csv(eng_sample, index=False)
        eng_path = eng_sample
        report["engineered_dataset"] = str(eng_path)
        report["engineered_sha256"] = _sha256(eng_path)

    # 1) Data Analyst
    if raw_path:
        da_dir = work_root / "data_analyst"
        da_dir.mkdir(parents=True, exist_ok=True)
        da_script = da_dir / "da_run.py"
        da_script.write_text(_build_da_script(raw_path), encoding="utf-8")
        da_res = _run_subprocess([sys.executable, str(da_script)], da_dir, 300)
        payload = _extract_block(da_res["stdout"], "__DA_RESULTS_BEGIN__", "__DA_RESULTS_END__")
        missing = _validate_required(payload, DA_REQUIRED_KEYS)
        report["checks"]["data_analyst"] = {
            "returncode": da_res["returncode"],
            "missing_keys": missing,
            "has_report_txt": (da_dir / "da_report.txt").exists(),
        }
    else:
        report["checks"]["data_analyst"] = {"skipped": True}

    # 2) Data Scientist (feature engineering)
    if raw_path:
        ds_dir = work_root / "data_scientist"
        ds_dir.mkdir(parents=True, exist_ok=True)
        fe_script = ds_dir / "fe_runner.py"
        fe_script.write_text(_build_fe_runner(raw_path), encoding="utf-8")
        ds_res = _run_subprocess([sys.executable, str(fe_script)], ds_dir, 600)
        base = raw_path.stem
        engineered_out = raw_path.parent / f"{base}_engineered.csv"
        report["checks"]["data_scientist"] = {
            "returncode": ds_res["returncode"],
            "engineered_exists": engineered_out.exists(),
            "engineered_path": str(engineered_out) if engineered_out.exists() else None,
        }
        if engineered_out.exists():
            eng_path = engineered_out
            report["engineered_dataset"] = str(eng_path)
            report["engineered_sha256"] = _sha256(eng_path)
    else:
        report["checks"]["data_scientist"] = {"skipped": True}

    # 3) ML Engineer (raw + engineered)
    ml_env = {
        "ML_OPTUNA_TRIALS": "2",
        "ML_N_JOBS": "1",
    }
    if raw_path:
        ml_dir = work_root / "ml_engineer_raw"
        ml_dir.mkdir(parents=True, exist_ok=True)
        ml_script = ml_dir / "ml_run_raw.py"
        ml_script.write_text(_build_ml_script(raw_path), encoding="utf-8")
        ml_res = _run_subprocess([sys.executable, str(ml_script)], ml_dir, 1200, env_extra=ml_env)
        payload = _extract_block(ml_res["stdout"], "__ML_RESULTS_BEGIN__", "__ML_RESULTS_END__")
        missing = _validate_required(payload, ML_REQUIRED_KEYS)
        report["checks"]["ml_engineer_raw"] = {
            "returncode": ml_res["returncode"],
            "missing_keys": missing,
            "has_results_json": (ml_dir / "ml_results.json").exists(),
        }

    if eng_path:
        ml_dir = work_root / "ml_engineer_engineered"
        ml_dir.mkdir(parents=True, exist_ok=True)
        ml_script = ml_dir / "ml_run_engineered.py"
        ml_script.write_text(_build_ml_script(eng_path), encoding="utf-8")
        ml_res = _run_subprocess([sys.executable, str(ml_script)], ml_dir, 1200, env_extra=ml_env)
        payload = _extract_block(ml_res["stdout"], "__ML_RESULTS_BEGIN__", "__ML_RESULTS_END__")
        missing = _validate_required(payload, ML_REQUIRED_KEYS)
        report["checks"]["ml_engineer_engineered"] = {
            "returncode": ml_res["returncode"],
            "missing_keys": missing,
            "has_results_json": (ml_dir / "ml_results.json").exists(),
        }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 execution checks")
    parser.add_argument("--raw", help="Path to raw dataset (overrides auto-detect)")
    parser.add_argument("--engineered", help="Path to engineered dataset (overrides auto-detect)")
    parser.add_argument("--project-root", help="Target project root for outputs (defaults to repo root)")
    parser.add_argument("--no-sample", action="store_true", help="Do not downsample datasets")
    args = parser.parse_args()

    target_root = Path(args.project_root) if args.project_root else CODE_ROOT
    raw_path = Path(args.raw) if args.raw else _find_latest_dataset(target_root, raw_only=True)
    eng_path = Path(args.engineered) if args.engineered else _find_latest_engineered(target_root)

    report = run_phase3_checks(raw_path, eng_path, use_sample=not args.no_sample, target_root=target_root)

    out_json = target_root / "phase3_check_report.json"
    out_txt = target_root / "phase3_check_report.txt"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "PHASE 3 CHECK REPORT",
        "=" * 60,
        f"Timestamp: {report['timestamp']}",
        f"Raw dataset: {report.get('raw_dataset')}",
        f"Engineered dataset: {report.get('engineered_dataset')}",
        "",
        "Checks:",
    ]
    for name, res in report["checks"].items():
        lines.append(f"- {name}: {res}")
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_json} and {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
