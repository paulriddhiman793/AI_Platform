// agents.js — Agent definitions, scenarios, and direct response flows
// Each step can carry a `fileWrite` field:
//   { agent: "ml_engineer", filename: "pipeline.py", content: "..." }
// The frontend fires this to the backend which writes it to disk.

export const AGENTS = {
  orchestrator: {
    id: "orchestrator", name: "Orchestrator", shortName: "ORCH", icon: "🧠",
    color: "#6366f1", bgColor: "#1e1b4b",
    role: "Routes tasks, coordinates the team, speaks to you",
    status: "idle",
  },
  ml_engineer: {
    id: "ml_engineer", name: "ML Engineer", shortName: "ML", icon: "🤖",
    color: "#10b981", bgColor: "#022c22",
    role: "Builds, trains, deploys ML models. Self-healing debug loop.",
    status: "idle",
  },
  data_scientist: {
    id: "data_scientist", name: "Data Scientist", shortName: "DS", icon: "📊",
    color: "#3b82f6", bgColor: "#0c1a3a",
    role: "EDA, hypothesis testing, feature engineering, experiments.",
    status: "idle",
  },
  data_analyst: {
    id: "data_analyst", name: "Data Analyst", shortName: "DA", icon: "📈",
    color: "#f59e0b", bgColor: "#1c1000",
    role: "Business insights, dashboards, twice-daily health reports.",
    status: "idle",
  },
  frontend: {
    id: "frontend", name: "Frontend Agent", shortName: "FE", icon: "💻",
    color: "#ec4899", bgColor: "#2d0a1a",
    role: "React components, API integration, UX, Playwright tests.",
    status: "idle",
  },
  sast: {
    id: "sast", name: "SAST Agent", shortName: "SAST", icon: "🔒",
    color: "#ef4444", bgColor: "#2d0a0a",
    role: "Static analysis, vulnerability scanning, security hardening.",
    status: "idle",
  },
  runtime_security: {
    id: "runtime_security", name: "Runtime Security", shortName: "RT-SEC", icon: "🛡",
    color: "#8b5cf6", bgColor: "#1a0a2d",
    role: "Live threat detection, pen testing, anomaly monitoring.",
    status: "idle",
  },
  github: {
    id: "github", name: "GitHub Agent", shortName: "GH", icon: "GH",
    color: "#93c5fd", bgColor: "#0b1022",
    role: "Pushes repo, per-agent branches, and merges to main.",
    status: "idle",
  },
};

// ─── File content templates ───────────────────────────────────────────────────

const FILES = {

  // ── ML Engineer files ───────────────────────────────────────────────────────

  pipeline_v1: `"""
pipeline.py — Churn Prediction Pipeline  [v1 — initial build]
Written by ML Engineer Agent
"""
import os, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

DATA_PATH  = "data/churn_dataset.csv"
MODEL_PATH = "model/churn_model.joblib"

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
    return df

def preprocess(df):
    # v1: basic preprocessing — EDA recommendations not yet applied
    X = df.drop("churn", axis=1)
    y = df["churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def train(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1":       round(f1_score(y_test, y_pred), 4),
        "auc":      round(roc_auc_score(y_test, y_proba), 4),
    }

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess(df)
    model = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    joblib.dump(model, MODEL_PATH)
    print("v1 pipeline complete:", metrics)
`,

  pipeline_v2_eda: `"""
pipeline.py — Churn Prediction Pipeline  [v2 — EDA recommendations applied]
Written by ML Engineer Agent · Reviewed by Data Scientist
Changes from v1:
  - Dropped: promo_clicks (41% nulls), referral_code, tenure (collinear)
  - Log-transformed: account_age, session_duration
  - One-hot encoded: region, device_type
  - Added class_weight='balanced'
"""
import os, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

DATA_PATH  = "data/churn_dataset.csv"
MODEL_PATH = "model/churn_model.joblib"

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
    return df

def preprocess(df):
    # v2: EDA-recommended transforms applied
    drop_cols = ["promo_clicks", "referral_code", "tenure"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for col in ["account_age", "session_duration"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    for col in ["region", "device_type"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    X = df.drop("churn", axis=1)
    y = df["churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def train(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1":       round(f1_score(y_test, y_pred), 4),
        "auc":      round(roc_auc_score(y_test, y_proba), 4),
    }

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess(df)
    model = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    joblib.dump(model, MODEL_PATH)
    print("v2 pipeline complete:", metrics)
`,

  pipeline_v3_sast: `"""
pipeline.py — Churn Prediction Pipeline  [v3 — SAST security fixes applied]
Written by ML Engineer Agent · SAST approved
Changes from v2:
  - Replaced hardcoded DB_CONN with os.getenv('DB_CONN_STR')
  - Added os.path.abspath() path sanitization
  - Upgraded requests to 2.31.0 (CVE fix)
  - Removed debug print statements
"""
import os, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
from pathlib import Path

DB_CONN_STR = os.getenv("DB_CONN_STR")               # SAST fix: no hardcoded creds
DATA_PATH   = Path(os.path.abspath("data/churn_dataset.csv"))  # SAST fix: sanitized path
MODEL_PATH  = Path(os.path.abspath("model/churn_model.joblib"))

def load_data(path: Path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    drop_cols = ["promo_clicks", "referral_code", "tenure"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for col in ["account_age", "session_duration"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    for col in ["region", "device_type"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    X = df.drop("churn", axis=1)
    y = df["churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def train(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1":       round(f1_score(y_test, y_pred), 4),
        "auc":      round(roc_auc_score(y_test, y_proba), 4),
    }

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess(df)
    model = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(MODEL_PATH))
    print("v3 pipeline complete:", metrics)
`,

  deploy_v1: `"""
deploy.py — Model Deployment Script  [v1 — initial]
Written by ML Engineer Agent
"""
import os, joblib, pandas as pd
from pathlib import Path

MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.joblib")

def load_model():
    return joblib.load(Path(os.path.abspath(MODEL_PATH)))

def predict(features: dict) -> dict:
    model = load_model()
    df = pd.DataFrame([features])
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    return {"churn": bool(pred), "probability": round(float(proba), 4)}

if __name__ == "__main__":
    sample = {"account_age": 2.4, "session_duration": 1.8, "region_EU": 1}
    print(predict(sample))
`,

  deploy_v2_sast: `"""
deploy.py — Model Deployment Script  [v2 — SAST approved]
Written by ML Engineer Agent · SAST approved ✅
Changes from v1:
  - Path sanitization added (os.path.realpath)
  - Input validation added
  - Error handling improved
"""
import os, joblib, pandas as pd
from pathlib import Path

MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.joblib")

def load_model():
    safe_path = Path(os.path.realpath(os.path.abspath(MODEL_PATH)))
    if not safe_path.exists():
        raise FileNotFoundError(f"Model not found: {safe_path}")
    return joblib.load(safe_path)

def predict(features: dict) -> dict:
    if not features:
        raise ValueError("Features dict cannot be empty")
    model = load_model()
    df = pd.DataFrame([features])
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    return {"churn": bool(pred), "probability": round(float(proba), 4)}

if __name__ == "__main__":
    sample = {"account_age": 2.4, "session_duration": 1.8, "region_EU": 1}
    print(predict(sample))
`,

  eda_report: `# EDA Report  [v1]
Generated by Data Scientist Agent

## Dataset Overview
- Shape: 45,231 rows × 24 columns
- Target: \`churn\` (binary 0/1)

## Null Analysis
| Feature        | Null Rate | Action              |
|----------------|-----------|---------------------|
| promo_clicks   | 41.2% ⚠  | DROP — broken source |
| last_login     | 12.1%     | Impute median        |
| referral_code  | 8.4%      | DROP                 |

## Skewness
- account_age: 2.3 → log-transform recommended
- session_duration: 1.8 → log-transform recommended

## Multicollinearity
- age ↔ tenure: r=0.91 → DROP tenure

## Class Balance
- Class 0 (no churn): 78%
- Class 1 (churn): 22%
- Recommendation: class_weight='balanced'

## Recommendations Applied in pipeline.py v2
`,

  eda_report_updated: `# EDA Report  [v2 — updated after validation]
Generated by Data Scientist Agent · Updated after model validation

## Dataset Overview
- Shape: 45,231 rows × 24 columns

## Changes vs v1
- Confirmed: new features session_duration and page_depth show strong signal
- promo_clicks permanently removed from feature set (Data Analyst confirmed)
- Model accuracy with v2 features: 95.1% (up from 94.1%)

## Final Feature Set
KEEP:    account_age (log), session_duration (log), page_depth, region (OHE)
DROP:    promo_clicks, referral_code, tenure
ENCODE:  region, device_type
`,

  sast_report_findings: `# SAST Scan Report — Initial Findings
Generated by SAST Agent

## Result: ⚠ ISSUES FOUND

### HIGH (1)
- **File**: pipeline.py
- **Issue**: Hardcoded database connection string
- **Found**: DB_CONN = 'postgresql://admin:pass@prod-db:5432/ml'
- **Fix**: Replace with os.getenv('DB_CONN_STR')

### MEDIUM (1)
- **File**: requirements.txt
- **Issue**: requests==2.28 has CVE-2023-32681
- **Fix**: Upgrade to requests>=2.31.0

⛔ Deployment blocked until HIGH is resolved.
`,

  sast_report_approved: `# SAST Re-Scan Report — APPROVED
Generated by SAST Agent

## Result: ✅ PASSED

All previous findings resolved:
- ✅ Hardcoded DB connection → replaced with os.getenv()
- ✅ Path sanitization added with os.path.realpath()
- ✅ requests upgraded to 2.31.0

**Security score: 94/100**
**Approved for deployment.**
`,

  pentest_report: `# Runtime Pen Test Report
Generated by Runtime Security Agent

## Result: ✅ CLEAN

Endpoints tested: 14
- ✅ Auth: all endpoints require valid token
- ✅ SQL injection: no vectors found
- ✅ XSS: no reflected or stored vectors
- ✅ Rate limiting: 429 after 100 req/min
- ✅ Path traversal: patched and confirmed fixed
- ✅ /api/debug: disabled in prod

**Live system secure. Approved for production.**
`,

  feature_engineering: `"""
feature_engineering.py — Feature Engineering Pipeline
Generated by Data Scientist Agent · Updated after EDA v2
"""
import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop degraded / collinear features
    drop_cols = ["promo_clicks", "referral_code", "tenure"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    # Log-transform skewed features
    for col in ["account_age", "session_duration"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    # One-hot encode categoricals
    for col in ["region", "device_type"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    return df
`,

  requirements_v1: `# requirements.txt [v1 — initial]
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.26.0
joblib>=1.3.0
requests==2.28.0
python-dotenv>=1.0.0
`,

  requirements_v2: `# requirements.txt [v2 — SAST CVE fix applied]
# Changed: requests upgraded from 2.28.0 → 2.31.0 (CVE-2023-32681 fix)
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.26.0
joblib>=1.3.0
requests>=2.31.0
python-dotenv>=1.0.0
`,

  // Project prediction API server (written by ML Engineer after SAST approval)
  api_server: `"""
api_server.py — Project Prediction API
Generated by ML Engineer Agent · SAST approved
Run: uvicorn api_server:app --reload --port 5000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, joblib, pandas as pd
from pathlib import Path

app = FastAPI(title="Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_PATH = Path(os.path.abspath(os.getenv("MODEL_PATH", "model/model.joblib")))

@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH), "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(features: dict):
    if not MODEL_PATH.exists():
        raise HTTPException(404, f"Model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([features])
    return {
        "prediction":  bool(model.predict(df)[0]),
        "probability": round(float(model.predict_proba(df)[0][1]), 4),
    }

@app.get("/metrics")
def metrics():
    return {
        "accuracy": 94.1, "f1": 0.89, "auc": 0.96,
        "drift_score": 0.03, "latency_p95": 118,
        "inference_volume": 8420,
    }
`,

  // Project Dashboard — hits the PROJECT's prediction API on port 5000
  dashboard_jsx_v1: `// Dashboard.jsx [v1 — initial build]
// Project frontend — connects to backend at localhost:8000
// Written by Frontend Agent
import React, { useState, useEffect } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Dashboard() {
  const [metrics, setMetrics]     = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    fetch(\`\${API_BASE}/metrics\`).then(r => r.json()).then(setMetrics);
  }, []);

  const runPrediction = () => {
    fetch(\`\${API_BASE}/predict\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ account_age: 24, session_duration: 5.2 }),
    }).then(r => r.json()).then(setPrediction);
  };

  if (!metrics) return <div>Connecting to prediction API...</div>;

  return (
    <div className="dashboard">
      <h2>Model Dashboard</h2>
      <div className="metrics-grid">
        <MetricCard label="Accuracy"    value={\`\${metrics.accuracy}%\`} />
        <MetricCard label="F1 Score"    value={metrics.f1} />
        <MetricCard label="AUC"         value={metrics.auc} />
        <MetricCard label="Latency p95" value={\`\${metrics.latency_p95}ms\`} />
      </div>
      <button onClick={runPrediction}>Run Test Prediction</button>
      {/* v1: using innerHTML — XSS risk flagged by SAST */}
      {prediction && <div dangerouslySetInnerHTML={{ __html: JSON.stringify(prediction) }} />}
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span className="label">{label}</span>
      <span className="value">{value}</span>
    </div>
  );
}
`,

  dashboard_jsx_v2: `// Dashboard.jsx [v2 — SAST XSS fix applied]
// Project frontend — connects to backend at localhost:8000
// Written by Frontend Agent · SAST approved ✅
import React, { useState, useEffect } from 'react';
import DOMPurify from 'dompurify';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Dashboard() {
  const [metrics, setMetrics]       = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError]           = useState(null);

  useEffect(() => {
    fetch(\`\${API_BASE}/metrics\`)
      .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
      .then(setMetrics)
      .catch(e => setError(e.message));
  }, []);

  const runPrediction = () => {
    fetch(\`\${API_BASE}/predict\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ account_age: 24, session_duration: 5.2 }),
    })
      .then(r => r.json())
      .then(setPrediction)
      .catch(e => setError(e.message));
  };

  if (error)    return <div className="error">API Error: {DOMPurify.sanitize(error)}</div>;
  if (!metrics) return <div className="loading">Connecting to prediction API at {API_BASE}...</div>;

  return (
    <div className="dashboard">
      <h2>Model Dashboard</h2>
      <div className="metrics-grid">
        <MetricCard label="Accuracy"    value={\`\${metrics.accuracy}%\`} />
        <MetricCard label="F1 Score"    value={String(metrics.f1)} />
        <MetricCard label="AUC"         value={String(metrics.auc)} />
        <MetricCard label="Latency p95" value={\`\${metrics.latency_p95}ms\`} />
        <MetricCard label="Volume"      value={String(metrics.inference_volume)} />
      </div>
      <button onClick={runPrediction}>Run Test Prediction</button>
      {/* v2: DOMPurify — XSS fixed */}
      {prediction && (
        <div className="prediction-result">
          <strong>Prediction:</strong>
          <span>{DOMPurify.sanitize(String(prediction.prediction))}</span>
          <strong>Probability:</strong>
          <span>{DOMPurify.sanitize(String(prediction.probability))}</span>
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span className="label">{DOMPurify.sanitize(label)}</span>
      <span className="value">{DOMPurify.sanitize(value)}</span>
    </div>
  );
}
`,
};

// ─── TEAM SCENARIOS ───────────────────────────────────────────────────────────
export const SCENARIOS = {
  model: {
    keywords: ["model", "train", "churn", "ml", "predict", "classif", "accuracy", "logistic", "random"],
    flow: [
      { from: "orchestrator",  type: "team", delay: 700,   tag: "STATUS", content: "Task received. Assigning ML Engineer + Data Scientist for end-to-end pipeline." },
      { from: "ml_engineer",   type: "team", delay: 1800,  tag: "STATUS", content: "Starting pipeline. Dataset loaded: ~12k rows, 18 features. Writing pipeline.py v1.",
        fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v1 } },
      { from: "ml_engineer",   type: "p2p",  delay: 2600,  to: "data_scientist", content: "Can you run EDA before I finalize the pipeline?" },
      { from: "data_scientist",type: "team", delay: 4200,  tag: "REPORT", content: "EDA complete. 3 features with >40% nulls. Skew on account_age. Writing eda_report.md.",
        fileWrite: { agent: "data_scientist", filename: "eda_report.md", content: FILES.eda_report } },
      { from: "ml_engineer",   type: "team", delay: 5600,  tag: "STATUS", content: "Applying EDA recommendations. Updating pipeline.py → v2.",
        fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v2_eda } },
      { from: "ml_engineer",   type: "team", delay: 6200,  tag: "STATUS", content: "Training complete. Accuracy: 94.1%. Writing requirements.txt and sending to SAST.",
        fileWrite: { agent: "shared", filename: "requirements.txt", content: FILES.requirements_v1 } },
      { from: "ml_engineer",   type: "p2p",  delay: 6600,  to: "sast", content: "Sending pipeline.py for security scan." },
      { from: "sast",          type: "team", delay: 7800,  tag: "ALERT", content: "Found 1 HIGH: hardcoded DB string. 1 MEDIUM: requests CVE. Writing scan report.",
        fileWrite: { agent: "sast", filename: "scan_report.md", content: FILES.sast_report_findings } },
      { from: "ml_engineer",   type: "team", delay: 8700,  tag: "STATUS", content: "Fixes applied. pipeline.py → v3 (path sanitized, env vars). requirements.txt → v2.",
        fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v3_sast } },
      { from: "ml_engineer",   type: "team", delay: 9000,  tag: "STATUS", content: "requirements.txt updated — CVE patched.",
        fileWrite: { agent: "shared", filename: "requirements.txt", content: FILES.requirements_v2 } },
      { from: "sast",          type: "team", delay: 10000, tag: "REPORT", content: "Re-scan passed. ✅ All clear. Writing approval report.",
        fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
      { from: "ml_engineer",   type: "team", delay: 11000, tag: "STATUS", content: "Writing deploy.py and committing to GitHub.",
        fileWrite: { agent: "ml_engineer", filename: "deploy.py", content: FILES.deploy_v2_sast } },
      { from: "ml_engineer",   type: "team", delay: 12000, tag: "REPORT", content: "CI passed ✅ Model deployed. Accuracy: 94.1%. All files written to workspace." },
      { from: "data_analyst",  type: "team", delay: 12800, tag: "REPORT", content: "Monitoring report updated. Next health check: 9PM." },
      { from: "orchestrator",  type: "team", delay: 13600, tag: "DONE",   content: "✅ Task complete. Model trained, SAST-cleared, deployed. Check workspace for all files." },
    ],
  },
  security: {
    keywords: ["security", "scan", "audit", "vuln", "pentest", "owasp"],
    flow: [
      { from: "orchestrator",     type: "team", delay: 700,  tag: "STATUS", content: "Security audit initiated. SAST + Runtime Security assigned." },
      { from: "sast",             type: "team", delay: 3800, tag: "REPORT", content: "Static scan complete. 2 HIGH, 4 MEDIUM, 1 LOW. Writing report.",
        fileWrite: { agent: "sast", filename: "scan_report.md", content: FILES.sast_report_findings } },
      { from: "runtime_security", type: "team", delay: 5000, tag: "REPORT", content: "Dynamic scan complete. /api/debug still accessible in prod." },
      { from: "runtime_security", type: "team", delay: 7300, tag: "ALERT",  content: "🔴 CRITICAL: Deserialization endpoint exploitable. Immediate patch required." },
      { from: "sast",             type: "team", delay: 8200, tag: "STATUS", content: "Patches applied. Re-scan approved.",
        fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
      { from: "orchestrator",     type: "team", delay: 9000, tag: "DONE",   content: "Audit complete. Critical: 1 patched. All clear." },
    ],
  },
  dashboard: {
    keywords: ["frontend"],  // ONLY triggers when "frontend" is explicitly in the message
    flow: [
      { from: "orchestrator", type: "team", delay: 700,  tag: "STATUS", content: "Dashboard task. Frontend Agent + Data Analyst assigned." },
      { from: "frontend",     type: "p2p",  delay: 1500, to: "data_analyst", content: "What metrics should I surface?" },
      { from: "data_analyst", type: "team", delay: 2800, tag: "REPORT", content: "Metrics: accuracy, drift, latency p50/p95, inference volume." },
      { from: "frontend",     type: "team", delay: 4200, tag: "STATUS", content: "Dashboard.jsx v1 written. Sending to SAST for XSS review.",
        fileWrite: { agent: "frontend", filename: "Dashboard.jsx", content: FILES.dashboard_jsx_v1 } },
      { from: "sast",         type: "team", delay: 6000, tag: "ALERT",  content: "XSS risk: dangerouslySetInnerHTML in Tooltip. Must switch to textContent." },
      { from: "frontend",     type: "team", delay: 6900, tag: "STATUS", content: "Fixed. Dashboard.jsx → v2 with DOMPurify.",
        fileWrite: { agent: "frontend", filename: "Dashboard.jsx", content: FILES.dashboard_jsx_v2 } },
      { from: "sast",         type: "team", delay: 7700, tag: "REPORT", content: "Re-scan passed. ✅ Approved.",
        fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
      { from: "frontend",     type: "team", delay: 8800, tag: "REPORT", content: "Dashboard deployed. Lighthouse: 97/100." },
      { from: "frontend",     type: "team", delay: 9400, tag: "DONE",
        content: "Project frontend is live.\n\n  🌐 http://localhost:5173\n\nOpen in your browser — it connects to the prediction API. Asking ML Engineer to verify the connection now...",
        fileWrite: { agent: "frontend", filename: "Dashboard.jsx", content: FILES.dashboard_jsx_v2 } },
      { from: "ml_engineer",  type: "team", delay: 10600, tag: "STATUS",
        content: "Checking project frontend <-> prediction API connectivity...\n  Frontend:       http://localhost:5173\n  Prediction API: http://localhost:8000" },
      { from: "ml_engineer",  type: "team", delay: 12400, tag: "REPORT",
        content: "Prediction API reachable at http://localhost:8000\n  GET  /health -> 200 OK\n    {status: \"ok\", model: \"model/model.joblib\", model_loaded: true}\n  Project frontend <-> Prediction API: CONNECTED" },
      { from: "ml_engineer",  type: "team", delay: 14200, tag: "REPORT",
        content: "Live API test:\n  POST /predict  -> 200 OK\n    Request:  {account_age: 24, session_duration: 5.2}\n    Response: {prediction: true, probability: 0.8731}\n\n  GET  /metrics  -> 200 OK\n    {accuracy: 94.1, f1: 0.89, auc: 0.96}\n\nAll endpoints verified. Open the dashboard:\n  🌐 http://localhost:5173" },
      { from: "orchestrator", type: "team", delay: 15200, tag: "DONE",
        content: "✅ Project frontend live at http://localhost:5173 — prediction API connected and tested." },
    ],
  },
  eda: {
    keywords: ["eda", "data", "analysis", "dataset", "feature", "explore", "statistic"],
    flow: [
      { from: "orchestrator",   type: "team", delay: 700,  tag: "STATUS", content: "EDA task. Data Scientist + Data Analyst assigned." },
      { from: "data_scientist", type: "team", delay: 2200, tag: "STATUS", content: "EDA in progress. Shape: 45k × 24 cols.",
        fileWrite: { agent: "data_scientist", filename: "feature_engineering.py", content: FILES.feature_engineering } },
      { from: "data_scientist", type: "p2p",  delay: 3300, to: "data_analyst", content: "Check historical logs for null rate trends." },
      { from: "data_analyst",   type: "team", delay: 4600, tag: "REPORT", content: "promo_clicks null spike from 2% → 41% on Feb 8th. Tracking pixel broke." },
      { from: "data_scientist", type: "team", delay: 6000, tag: "REPORT", content: "EDA complete. Writing eda_report.md.",
        fileWrite: { agent: "data_scientist", filename: "eda_report.md", content: FILES.eda_report } },
      { from: "orchestrator",   type: "team", delay: 7000, tag: "DONE",   content: "📊 EDA complete. Data quality incident logged." },
    ],
  },
  default: {
    keywords: [],
    flow: [
      { from: "orchestrator", type: "team", delay: 700,  tag: "STATUS", content: "Analyzing request and assessing which agents are needed." },
      { from: "orchestrator", type: "team", delay: 3200, tag: "STATUS", content: "Team assessed. Ready for your next instruction." },
    ],
  },
};

// ─── DIRECT RESPONSE FLOWS ────────────────────────────────────────────────────
// steps[]     — messages in the private chat
// spawnGroup  — auto-creates a group chat, runs groupFlow there
// Every step with fileWrite rewrites the file on disk (overwrites previous version)

export const AGENT_DIRECT_RESPONSES = {
  ml_engineer: [
    {
      match: ["fix", "error", "bug", "broken", "fail"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "On it. Pulling latest error logs from the sandbox." },
        { delay: 2200, tag: "STATUS", content: "Root cause: feature shape mismatch in preprocessing. Applying fix now.",
          fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v2_eda } },
        { delay: 3800, tag: "REPORT", content: "Fix applied. pipeline.py updated ✍ All assertions pass. Need SAST review before I push — spawning security thread." },
      ],
      spawnGroup: {
        members: ["ml_engineer", "sast"],
        title: "Bug Fix · Security Review",
        reason: "ML Engineer needs SAST to review the fix before deployment.",
        groupFlow: [
          { from: "ml_engineer", delay: 600,  tag: "STATUS", content: "SAST — fix is ready. Sending pipeline.py diff for review." },
          { from: "sast",        delay: 2200, tag: "REPORT", content: "Reviewed. Fix looks clean. One note: new file read on line 23 needs path sanitization. Writing findings.",
            fileWrite: { agent: "sast", filename: "scan_report.md", content: FILES.sast_report_findings } },
          { from: "ml_engineer", delay: 3600, tag: "STATUS", content: "Good catch. Adding os.path.abspath(). Updating pipeline.py → v3.",
            fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v3_sast } },
          { from: "sast",        delay: 5000, tag: "REPORT", content: "Re-scan passed. ✅ No issues. pipeline.py approved. Updating approval report.",
            fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
          { from: "ml_engineer", delay: 6200, tag: "DONE",   content: "Fix committed to GitHub. CI started. deploy.py also updated.",
            fileWrite: { agent: "ml_engineer", filename: "deploy.py", content: FILES.deploy_v2_sast } },
        ],
      },
    },
    {
      match: ["retrain", "train again", "new model", "improve", "accuracy"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Starting retraining with updated feature set.",
          fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v2_eda } },
        { delay: 2500, tag: "STATUS", content: "Training complete. New accuracy: 95.8% — up from 94.1%. Pulling in Data Scientist to validate." },
        { delay: 3200, tag: "REPORT", content: "pipeline.py updated ✍ Spawning validation + approval thread." },
      ],
      spawnGroup: {
        members: ["ml_engineer", "data_scientist", "sast"],
        title: "Model Retrain · Validation",
        reason: "Data Scientist validates the retrained model, SAST clears it for deploy.",
        groupFlow: [
          { from: "ml_engineer",   delay: 600,  tag: "STATUS", content: "New model: 95.8% accuracy. Data Scientist — validate feature importances?" },
          { from: "data_scientist",delay: 2400, tag: "REPORT", content: "Validation complete. Feature importances stable. +1.7% from quantile transform on account_age. Updating EDA report.",
            fileWrite: { agent: "data_scientist", filename: "eda_report.md", content: FILES.eda_report_updated } },
          { from: "ml_engineer",   delay: 3800, tag: "STATUS", content: "SAST — sending updated deploy script for final review." },
          { from: "sast",          delay: 5400, tag: "REPORT", content: "Scan clean. ✅ No vulnerabilities. Updating approval report.",
            fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
          { from: "ml_engineer",   delay: 6600, tag: "DONE",   content: "New model deployed. MLflow entry logged. deploy.py updated.",
            fileWrite: { agent: "ml_engineer", filename: "deploy.py", content: FILES.deploy_v2_sast } },
        ],
      },
    },
    {
      match: ["deploy", "push", "github", "ci"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Preparing deployment. Writing deploy.py v1.",
          fileWrite: { agent: "ml_engineer", filename: "deploy.py", content: FILES.deploy_v1 } },
        { delay: 2000, tag: "STATUS", content: "Local tests passed. Spawning security sign-off thread." },
      ],
      spawnGroup: {
        members: ["ml_engineer", "sast", "runtime_security"],
        title: "Deployment · Security Sign-off",
        reason: "SAST static scan + Runtime Security pen test required before production deploy.",
        groupFlow: [
          { from: "ml_engineer",     delay: 600,  tag: "STATUS", content: "SAST — scan deploy.py. Runtime Security — pen test on staging after." },
          { from: "sast",            delay: 2200, tag: "REPORT", content: "Static scan: 1 MEDIUM — requests CVE. Fix before deploy. Writing report.",
            fileWrite: { agent: "sast", filename: "scan_report.md", content: FILES.sast_report_findings } },
          { from: "ml_engineer",     delay: 3400, tag: "STATUS", content: "requests upgraded to 2.31.0. deploy.py → v2. requirements.txt updated.",
            fileWrite: { agent: "ml_engineer", filename: "deploy.py", content: FILES.deploy_v2_sast } },
          { from: "ml_engineer",     delay: 3800, tag: "STATUS", content: "requirements.txt updated — CVE patched.",
            fileWrite: { agent: "shared", filename: "requirements.txt", content: FILES.requirements_v2 } },
          { from: "sast",            delay: 5000, tag: "REPORT", content: "Re-scan passed. ✅ Writing approval.",
            fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
          { from: "runtime_security",delay: 6600, tag: "REPORT", content: "Pen test on staging complete. All clear. Writing pen test report.",
            fileWrite: { agent: "runtime_security", filename: "pentest_report.md", content: FILES.pentest_report } },
          { from: "ml_engineer",     delay: 7800, tag: "DONE",   content: "Both approvals received. Deploying to production. 🚀" },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Checking current pipeline state." },
        { delay: 2200, tag: "REPORT", content: "Pipeline healthy. Last run: 9AM — passed. Model accuracy stable at 94.1%. No drift detected." },
      ],
    },
  ],

  data_scientist: [
    {
      match: ["eda", "analysis", "explore", "feature", "dataset"],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Running EDA on the latest dataset snapshot." },
        { delay: 2800, tag: "REPORT", content: "EDA complete. Writing eda_report.md and feature_engineering.py.",
          fileWrite: { agent: "data_scientist", filename: "eda_report.md", content: FILES.eda_report } },
        { delay: 3200, tag: "STATUS", content: "feature_engineering.py written. Looping in Data Analyst and ML Engineer.",
          fileWrite: { agent: "data_scientist", filename: "feature_engineering.py", content: FILES.feature_engineering } },
      ],
      spawnGroup: {
        members: ["data_scientist", "data_analyst", "ml_engineer"],
        title: "EDA · Feature Review",
        reason: "Sharing EDA findings with Data Analyst and ML Engineer for joint review.",
        groupFlow: [
          { from: "data_scientist", delay: 700,  tag: "REPORT", content: "EDA results: session_duration and page_depth show highest signal. promo_clicks permanently broken. Updating eda_report.md.",
            fileWrite: { agent: "data_scientist", filename: "eda_report.md", content: FILES.eda_report_updated } },
          { from: "data_analyst",   delay: 2200, tag: "REPORT", content: "Confirmed — promo_clicks null since Feb 8th. Safe to drop permanently." },
          { from: "ml_engineer",    delay: 3600, tag: "STATUS", content: "Dropping promo_clicks. Adding session_duration + page_depth. Updating pipeline.py.",
            fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v2_eda } },
          { from: "data_scientist", delay: 5200, tag: "REPORT", content: "Retrained model validates well. Updating feature_engineering.py.",
            fileWrite: { agent: "data_scientist", filename: "feature_engineering.py", content: FILES.feature_engineering } },
          { from: "ml_engineer",    delay: 6400, tag: "DONE",   content: "Updated pipeline committed. New accuracy: 95.3%." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Reviewing experiment logs." },
        { delay: 2400, tag: "REPORT", content: "Last experiment: A/B on feature transforms — Variant B won by +2.1%. Already promoted to main." },
      ],
    },
  ],

  data_analyst: [
    {
      match: ["report", "metrics", "kpi", "status", "dashboard"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Pulling latest metrics from monitoring file." },
        { delay: 2200, tag: "REPORT", content: "Current: Accuracy 94.1% | Drift 0.03 | Latency p95 118ms | Volume 8,420." },
        { delay: 3500, tag: "DONE",   content: "Full metrics report compiled." },
      ],
    },
    {
      match: ["incident", "drift", "drop", "degraded"],
      steps: [
        { delay: 800,  tag: "ALERT",  content: "Investigating drift. Pulling historical data." },
        { delay: 2000, tag: "REPORT", content: "Accuracy dropped 3.2% over 48h. Correlates with age feature distribution shift. Looping in ML + DS." },
      ],
      spawnGroup: {
        members: ["data_analyst", "ml_engineer", "data_scientist"],
        title: "Incident · Model Drift",
        reason: "Data Analyst detected drift — ML Engineer + Data Scientist investigate and fix.",
        groupFlow: [
          { from: "data_analyst",   delay: 700,  tag: "ALERT",  content: "Accuracy: 94.1% → 90.9% since yesterday 6PM. age feature distribution shifted." },
          { from: "data_scientist", delay: 2400, tag: "STATUS", content: "EDA on last 48h slice. Confirmed: new 18-22 cohort not in training data. Updating eda_report.md.",
            fileWrite: { agent: "data_scientist", filename: "eda_report.md", content: FILES.eda_report_updated } },
          { from: "ml_engineer",    delay: 4000, tag: "STATUS", content: "Retraining with updated distribution. Updating pipeline.py.",
            fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v2_eda } },
          { from: "ml_engineer",    delay: 6200, tag: "REPORT", content: "Retrained. Accuracy back to 94.8%. Updating deploy.py.",
            fileWrite: { agent: "ml_engineer", filename: "deploy.py", content: FILES.deploy_v2_sast } },
          { from: "data_analyst",   delay: 7400, tag: "DONE",   content: "Drift score back to 0.04. Incident INC-2024-007 closed." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "REPORT", content: "Last health check (9AM): all systems nominal. Next check: 9PM." },
      ],
    },
  ],

  sast: [
    {
      match: ["scan", "check", "review", "security", "vuln"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Initiating targeted scan." },
        { delay: 2600, tag: "REPORT", content: "Scan complete. 0 critical, 0 high. 1 medium: numpy 1.23 CVE — upgrade to 1.26. Writing report.",
          fileWrite: { agent: "sast", filename: "scan_report.md", content: FILES.sast_report_findings } },
        { delay: 3800, tag: "DONE",   content: "Security score: 91/100. Patch staged." },
      ],
    },
    {
      match: ["full", "audit", "monthly"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Starting full codebase audit." },
        { delay: 2000, tag: "STATUS", content: "Scan in progress. Looping in Runtime Security." },
      ],
      spawnGroup: {
        members: ["sast", "runtime_security"],
        title: "Full Security Audit",
        reason: "SAST static scan + Runtime Security pen test running in parallel.",
        groupFlow: [
          { from: "sast",             delay: 700,  tag: "STATUS", content: "Static scan complete. 1 HIGH: path traversal in tools/code_execution.py:88. Writing report.",
            fileWrite: { agent: "sast", filename: "scan_report.md", content: FILES.sast_report_findings } },
          { from: "runtime_security", delay: 2400, tag: "STATUS", content: "OWASP ZAP sweep started." },
          { from: "runtime_security", delay: 4600, tag: "ALERT",  content: "Path traversal CONFIRMED exploitable. Escalating to CRITICAL." },
          { from: "sast",             delay: 6000, tag: "STATUS", content: "Patch written: strict path whitelisting. Updating approval report.",
            fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
          { from: "runtime_security", delay: 7400, tag: "REPORT", content: "Exploit no longer works. Writing pen test report.",
            fileWrite: { agent: "runtime_security", filename: "pentest_report.md", content: FILES.pentest_report } },
          { from: "sast",             delay: 8600, tag: "DONE",   content: "Full audit complete. Score: 94/100. All findings resolved." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "REPORT", content: "Last audit: 3 days ago. Score: 91/100. No critical vulnerabilities." },
      ],
    },
  ],

  runtime_security: [
    {
      match: ["test", "pen", "attack", "threat", "monitor"],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Running OWASP ZAP sweep on live endpoints." },
        { delay: 2800, tag: "REPORT", content: "Pen test complete. No critical issues. Looping in SAST to cross-reference." },
      ],
      spawnGroup: {
        members: ["runtime_security", "sast"],
        title: "Pen Test · Cross-Reference",
        reason: "Cross-referencing live pen test results with SAST static findings.",
        groupFlow: [
          { from: "runtime_security", delay: 700,  tag: "REPORT", content: "1 medium: /api/users leaks internal IDs in error responses. Sharing with SAST." },
          { from: "sast",             delay: 2200, tag: "REPORT", content: "Confirmed statically — error handler in api/users.py:156 returns full user object. Writing patch.",
            fileWrite: { agent: "sast", filename: "scan_report.md", content: FILES.sast_report_findings } },
          { from: "sast",             delay: 3800, tag: "STATUS", content: "Patch applied. Error responses now generic. Updating approval.",
            fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
          { from: "runtime_security", delay: 5200, tag: "REPORT", content: "Verified — leak patched. Writing final pen test report.",
            fileWrite: { agent: "runtime_security", filename: "pentest_report.md", content: FILES.pentest_report } },
          { from: "runtime_security", delay: 6200, tag: "DONE",   content: "Cross-reference complete. All findings resolved." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "REPORT", content: "Live system nominal. Falco: 0 alerts last 24h. Last pen test: 2 days ago — clean." },
      ],
    },
  ],

  frontend: [
    {
      match: ["fix", "bug", "broken", "layout", "style", "component"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Running Playwright to reproduce." },
        { delay: 2400, tag: "STATUS", content: "Reproduced. CSS grid overflow on mobile. Fixing now." },
        { delay: 3800, tag: "REPORT", content: "Fix applied. Dashboard.jsx → v1 updated. Sending to SAST.",
          fileWrite: { agent: "frontend", filename: "Dashboard.jsx", content: FILES.dashboard_jsx_v1 } },
      ],
      spawnGroup: {
        members: ["frontend", "sast"],
        title: "UI Fix · Security Review",
        reason: "Frontend needs SAST to sign off before deploying the component fix.",
        groupFlow: [
          { from: "frontend", delay: 600,  tag: "STATUS", content: "SAST — diff ready. Grid fix + tooltip refactor. Please review both." },
          { from: "sast",     delay: 2400, tag: "REPORT", content: "Grid fix clean. Tooltip: dangerouslySetInnerHTML XSS risk. Fix before deploy. Writing report.",
            fileWrite: { agent: "sast", filename: "scan_report.md", content: FILES.sast_report_findings } },
          { from: "frontend", delay: 3600, tag: "STATUS", content: "Replacing dangerouslySetInnerHTML with DOMPurify. Dashboard.jsx → v2.",
            fileWrite: { agent: "frontend", filename: "Dashboard.jsx", content: FILES.dashboard_jsx_v2 } },
          { from: "sast",     delay: 5000, tag: "REPORT", content: "Re-reviewed. ✅ XSS resolved. Both changes approved. Updating approval.",
            fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
          { from: "frontend", delay: 6000, tag: "DONE",   content: "Deployed. Lighthouse: 97/100. Grid and tooltip issues resolved." },
        ],
      },
    },
    {
      match: ["frontend", "dashboard"],  // only explicit frontend requests
      steps: [
        { delay: 800,  tag: "STATUS", content: "Starting dashboard build. Writing Dashboard.jsx v1.",
          fileWrite: { agent: "frontend", filename: "Dashboard.jsx", content: FILES.dashboard_jsx_v1 } },
        { delay: 1600, tag: "STATUS", content: "Coordinating with Data Analyst on metrics. Spawning build thread." },
      ],
      spawnGroup: {
        members: ["frontend", "data_analyst", "sast"],
        title: "Dashboard Build",
        reason: "Frontend needs metrics from Data Analyst and security review from SAST.",
        groupFlow: [
          { from: "frontend",    delay: 600,  tag: "STATUS", content: "Data Analyst — which metrics? I'm thinking accuracy, drift, latency, volume." },
          { from: "data_analyst",delay: 1800, tag: "REPORT", content: "Add: data freshness indicator + p95 latency (not just p50). All 6 metrics confirmed." },
          { from: "frontend",    delay: 3200, tag: "STATUS", content: "All 6 metrics wired up. Dashboard.jsx → v2. Sending to SAST.",
            fileWrite: { agent: "frontend", filename: "Dashboard.jsx", content: FILES.dashboard_jsx_v2 } },
          { from: "sast",        delay: 5000, tag: "REPORT", content: "Reviewed. ✅ Clean — no XSS or injection risks. Writing approval.",
            fileWrite: { agent: "sast", filename: "scan_report_approved.md", content: FILES.sast_report_approved } },
          { from: "frontend",    delay: 6000, tag: "DONE",
            content: "Project dashboard deployed. Lighthouse: 98/100.\n\n  🌐 http://localhost:5173\n\nConnects to backend API at localhost:8000. Asking ML Engineer to verify now..." },
          { from: "ml_engineer",  delay: 7800, tag: "STATUS",
            content: "Checking project frontend <-> prediction API connectivity...\n  Frontend:       http://localhost:5173\n  Prediction API: http://localhost:8000" },
          { from: "ml_engineer",  delay: 10000, tag: "REPORT",
            content: "Prediction API reachable at http://localhost:8000\n  GET  /health -> 200 OK\n    {status: \"ok\", model: \"model/model.joblib\", model_loaded: true}\n  Project frontend <-> Prediction API: CONNECTED" },
          { from: "ml_engineer",  delay: 11600, tag: "REPORT",
            content: "Live API test:\n  POST /predict  -> 200 OK\n    Request:  {account_age: 24, session_duration: 5.2}\n    Response: {prediction: true, probability: 0.8731}\n\n  GET  /metrics  -> 200 OK\n    {accuracy: 94.1, f1: 0.89}\n\nDashboard live at: 🌐 http://localhost:5173" },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "REPORT", content: "UI status: all components live. Lighthouse: 97/100. No open issues." },
      ],
    },
  ],

  orchestrator: [
    {
      match: [],
      steps: [
        { delay: 700,  tag: "STATUS", content: "Understood. Assessing what's needed." },
        { delay: 2000, tag: "REPORT", content: "All agents idle and ready. Tell me what to prioritize and I'll route it immediately." },
      ],
    },
  ],
};

// ─── Exports ──────────────────────────────────────────────────────────────────

export function detectScenario(userMessage) {
  const msg = userMessage.toLowerCase();
  for (const [key, scenario] of Object.entries(SCENARIOS)) {
    if (key === "default") continue;
    if (scenario.keywords.some(kw => msg.includes(kw))) return scenario;
  }
  return SCENARIOS.default;
}

export function detectDirectResponse(agentId, userMessage) {
  const msg = userMessage.toLowerCase();
  const responses = AGENT_DIRECT_RESPONSES[agentId];
  if (!responses) return { steps: [], spawnGroup: null };
  for (const resp of responses) {
    if (resp.match.length === 0) continue;
    if (resp.match.some(kw => msg.includes(kw))) {
      return { steps: resp.steps, spawnGroup: resp.spawnGroup || null };
    }
  }
  const last = responses[responses.length - 1];
  return { steps: last.steps, spawnGroup: last.spawnGroup || null };
}
