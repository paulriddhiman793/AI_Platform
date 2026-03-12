// agents.js â€” Agent definitions, scenarios, and direct response flows
// Each step can carry a `fileWrite` field:
//   { agent: "ml_engineer", filename: "pipeline.py", content: "..." }
// The UI fires this to the backend which writes it to disk.

export const AGENTS = {
  orchestrator: {
    id: "orchestrator", name: "Orchestrator", shortName: "ORCH", icon: "OR",
    color: "#6366f1", bgColor: "#1e1b4b",
    role: "Routes tasks, coordinates the team, speaks to you",
    status: "idle",
  },
  ml_engineer: {
    id: "ml_engineer", name: "ML Engineer", shortName: "ML", icon: "ML",
    color: "#10b981", bgColor: "#022c22",
    role: "Builds, trains, deploys ML models. Self-healing debug loop.",
    status: "idle",
  },
  data_scientist: {
    id: "data_scientist", name: "Data Scientist", shortName: "DS", icon: "DS",
    color: "#3b82f6", bgColor: "#0c1a3a",
    role: "EDA, hypothesis testing, feature engineering, experiments.",
    status: "idle",
  },
  data_analyst: {
    id: "data_analyst", name: "Data Analyst", shortName: "DA", icon: "DA",
    color: "#f59e0b", bgColor: "#1c1000",
    role: "Business insights, dashboards, twice-daily health reports.",
    status: "idle",
  },
  github: {
    id: "github", name: "GitHub Agent", shortName: "GH", icon: "GH",
    color: "#93c5fd", bgColor: "#0b1022",
    role: "Pushes repo, per-agent branches, and merges to main.",
    status: "idle",
  },
};

// â”€â”€â”€ File content templates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const FILES = {

  // â”€â”€ ML Engineer files â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  pipeline_v1: `"""
pipeline.py â€” Churn Prediction Pipeline  [v1 â€” initial build]
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
    # v1: basic preprocessing â€” EDA recommendations not yet applied
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
pipeline.py â€” Churn Prediction Pipeline  [v2 â€” EDA recommendations applied]
Written by ML Engineer Agent Â· Reviewed by Data Scientist
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

  deploy_v1: `"""
deploy.py â€” Model Deployment Script  [v1 â€” initial]
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

  eda_report: `# EDA Report  [v1]
Generated by Data Scientist Agent

## Dataset Overview
- Shape: 45,231 rows Ã— 24 columns
- Target: \`churn\` (binary 0/1)

## Null Analysis
| Feature        | Null Rate | Action              |
|----------------|-----------|---------------------|
| promo_clicks   | 41.2% âš   | DROP â€” broken source |
| last_login     | 12.1%     | Impute median        |
| referral_code  | 8.4%      | DROP                 |

## Skewness
- account_age: 2.3 â†’ log-transform recommended
- session_duration: 1.8 â†’ log-transform recommended

## Multicollinearity
- age â†” tenure: r=0.91 â†’ DROP tenure

## Class Balance
- Class 0 (no churn): 78%
- Class 1 (churn): 22%
- Recommendation: class_weight='balanced'

## Recommendations Applied in pipeline.py v2
`,

  eda_report_updated: `# EDA Report  [v2 â€” updated after validation]
Generated by Data Scientist Agent Â· Updated after model validation

## Dataset Overview
- Shape: 45,231 rows Ã— 24 columns

## Changes vs v1
- Confirmed: new features session_duration and page_depth show strong signal
- promo_clicks permanently removed from feature set (Data Analyst confirmed)
- Model accuracy with v2 features: 95.1% (up from 94.1%)

## Final Feature Set
KEEP:    account_age (log), session_duration (log), page_depth, region (OHE)
DROP:    promo_clicks, referral_code, tenure
ENCODE:  region, device_type
`,

  feature_engineering: `"""
feature_engineering.py â€” Feature Engineering Pipeline
Generated by Data Scientist Agent Â· Updated after EDA v2
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

  requirements_v1: `# requirements.txt [v1 â€” initial]
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.26.0
joblib>=1.3.0
requests==2.28.0
python-dotenv>=1.0.0
`,

  // Project prediction API server (written by ML Engineer)
  api_server: `"""
api_server.py â€” Project Prediction API
Generated by ML Engineer Agent
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

};

// â”€â”€â”€ TEAM SCENARIOS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
      { from: "ml_engineer",   type: "team", delay: 5600,  tag: "STATUS", content: "Applying EDA recommendations. Updating pipeline.py â†’ v2.",
        fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v2_eda } },
      { from: "ml_engineer",   type: "team", delay: 6200,  tag: "STATUS", content: "Training complete. Accuracy: 94.1%. Writing requirements.txt.",
        fileWrite: { agent: "shared", filename: "requirements.txt", content: FILES.requirements_v1 } },
      { from: "ml_engineer",   type: "team", delay: 7800,  tag: "STATUS", content: "Writing deploy.py and committing to GitHub.",
        fileWrite: { agent: "ml_engineer", filename: "deploy.py", content: FILES.deploy_v1 } },
      { from: "ml_engineer",   type: "team", delay: 12000, tag: "REPORT", content: "CI passed âœ… Model deployed. Accuracy: 94.1%. All files written to workspace." },
      { from: "data_analyst",  type: "team", delay: 12800, tag: "REPORT", content: "Monitoring report updated. Next health check: 9PM." },
      { from: "orchestrator",  type: "team", delay: 13600, tag: "DONE",   content: "âœ… Task complete. Model trained and deployed. Check workspace for all files." },
    ],
  },
  eda: {
    keywords: ["eda", "data", "analysis", "dataset", "feature", "explore", "statistic"],
    flow: [
      { from: "orchestrator",   type: "team", delay: 700,  tag: "STATUS", content: "EDA task. Data Scientist + Data Analyst assigned." },
      { from: "data_scientist", type: "team", delay: 2200, tag: "STATUS", content: "EDA in progress. Shape: 45k Ã— 24 cols.",
        fileWrite: { agent: "data_scientist", filename: "feature_engineering.py", content: FILES.feature_engineering } },
      { from: "data_scientist", type: "p2p",  delay: 3300, to: "data_analyst", content: "Check historical logs for null rate trends." },
      { from: "data_analyst",   type: "team", delay: 4600, tag: "REPORT", content: "promo_clicks null spike from 2% â†’ 41% on Feb 8th. Tracking pixel broke." },
      { from: "data_scientist", type: "team", delay: 6000, tag: "REPORT", content: "EDA complete. Writing eda_report.md.",
        fileWrite: { agent: "data_scientist", filename: "eda_report.md", content: FILES.eda_report } },
      { from: "orchestrator",   type: "team", delay: 7000, tag: "DONE",   content: "ðŸ“Š EDA complete. Data quality incident logged." },
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

// â”€â”€â”€ DIRECT RESPONSE FLOWS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// steps[]     â€” messages in the private chat
// spawnGroup  â€” auto-creates a group chat, runs groupFlow there
// Every step with fileWrite rewrites the file on disk (overwrites previous version)

export const AGENT_DIRECT_RESPONSES = {
  ml_engineer: [
    {
      match: ["fix", "error", "bug", "broken", "fail"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "On it. Pulling latest error logs from the sandbox." },
        { delay: 2200, tag: "STATUS", content: "Root cause: feature shape mismatch in preprocessing. Applying fix now.",
          fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v2_eda } },
        { delay: 3800, tag: "REPORT", content: "Fix applied. pipeline.py updated âœ All assertions pass. Ready to push." },
      ],
    },
    {
      match: ["retrain", "train again", "new model", "improve", "accuracy"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Starting retraining with updated feature set.",
          fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v2_eda } },
        { delay: 2500, tag: "STATUS", content: "Training complete. New accuracy: 95.8% â€” up from 94.1%. Pulling in Data Scientist to validate." },
        { delay: 3200, tag: "REPORT", content: "pipeline.py updated âœ Retrain complete and validated." },
      ],
    },
    {
      match: ["deploy", "push", "github", "ci"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Preparing deployment. Writing deploy.py v1.",
          fileWrite: { agent: "ml_engineer", filename: "deploy.py", content: FILES.deploy_v1 } },
        { delay: 2000, tag: "STATUS", content: "Local tests passed. Pushing to GitHub." },
      ],
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Checking current pipeline state." },
        { delay: 2200, tag: "REPORT", content: "Pipeline healthy. Last run: 9AM â€” passed. Model accuracy stable at 94.1%. No drift detected." },
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
        title: "EDA Â· Feature Review",
        reason: "Sharing EDA findings with Data Analyst and ML Engineer for joint review.",
        groupFlow: [
          { from: "data_scientist", delay: 700,  tag: "REPORT", content: "EDA results: session_duration and page_depth show highest signal. promo_clicks permanently broken. Updating eda_report.md.",
            fileWrite: { agent: "data_scientist", filename: "eda_report.md", content: FILES.eda_report_updated } },
          { from: "data_analyst",   delay: 2200, tag: "REPORT", content: "Confirmed â€” promo_clicks null since Feb 8th. Safe to drop permanently." },
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
        { delay: 2400, tag: "REPORT", content: "Last experiment: A/B on feature transforms â€” Variant B won by +2.1%. Already promoted to main." },
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
        title: "Incident Â· Model Drift",
        reason: "Data Analyst detected drift â€” ML Engineer + Data Scientist investigate and fix.",
        groupFlow: [
          { from: "data_analyst",   delay: 700,  tag: "ALERT",  content: "Accuracy: 94.1% â†’ 90.9% since yesterday 6PM. age feature distribution shifted." },
          { from: "data_scientist", delay: 2400, tag: "STATUS", content: "EDA on last 48h slice. Confirmed: new 18-22 cohort not in training data. Updating eda_report.md.",
            fileWrite: { agent: "data_scientist", filename: "eda_report.md", content: FILES.eda_report_updated } },
          { from: "ml_engineer",    delay: 4000, tag: "STATUS", content: "Retraining with updated distribution. Updating pipeline.py.",
            fileWrite: { agent: "ml_engineer", filename: "pipeline.py", content: FILES.pipeline_v2_eda } },
          { from: "ml_engineer",    delay: 6200, tag: "REPORT", content: "Retrained. Accuracy back to 94.8%. Updating deploy.py.",
            fileWrite: { agent: "ml_engineer", filename: "deploy.py", content: FILES.deploy_v1 } },
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

// â”€â”€â”€ Exports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

