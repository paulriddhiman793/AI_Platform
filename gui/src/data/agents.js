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
};

// ─── TEAM SCENARIOS (triggered from Team chat) ───────────────────────────────
export const SCENARIOS = {
  model: {
    keywords: ["model", "train", "churn", "ml", "predict", "classif", "accuracy", "logistic", "random"],
    flow: [
      { from: "orchestrator",  type: "team", delay: 700,   tag: "STATUS", content: "Task received. Assigning ML Engineer + Data Scientist for end-to-end pipeline." },
      { from: "ml_engineer",   type: "team", delay: 1800,  tag: "STATUS", content: "Starting pipeline. Dataset loaded: ~12k rows, 18 features. Baseline: RandomForest." },
      { from: "ml_engineer",   type: "p2p",  delay: 2600,  to: "data_scientist", content: "Can you run EDA and flag anything before I finalize the pipeline?" },
      { from: "data_scientist",type: "team", delay: 4200,  tag: "REPORT", content: "EDA complete. Found 3 features with >40% nulls — last_login, promo_clicks, referral_code. Heavy skew in account_age. Recommend imputation + log transform." },
      { from: "ml_engineer",   type: "team", delay: 5600,  tag: "STATUS", content: "Applying EDA recommendations. Training complete. Accuracy: 94.1%. Requesting SAST scan before deploy." },
      { from: "ml_engineer",   type: "p2p",  delay: 6000,  to: "sast",    content: "Sending deploy script for security scan." },
      { from: "sast",          type: "team", delay: 7400,  tag: "ALERT",  content: "Scan found 1 HIGH issue: hardcoded DB connection string on line 47. ML Engineer must fix before deploy." },
      { from: "ml_engineer",   type: "team", delay: 8300,  tag: "STATUS", content: "Fixed. Replaced with os.getenv('DB_CONN_STR'). Re-submitting to SAST." },
      { from: "sast",          type: "team", delay: 9200,  tag: "REPORT", content: "Re-scan passed. No vulnerabilities. ✅ Code approved for deployment." },
      { from: "ml_engineer",   type: "team", delay: 10100, tag: "STATUS", content: "Committed to GitHub. CI started. Monitoring run..." },
      { from: "ml_engineer",   type: "team", delay: 11200, tag: "REPORT", content: "CI passed. Model live. Accuracy: 94.1%. Pipeline deployed successfully." },
      { from: "data_analyst",  type: "team", delay: 12000, tag: "REPORT", content: "Monitoring report updated. Pipeline creation + model accuracy logged to Excel. Next health check: 9PM." },
      { from: "orchestrator",  type: "team", delay: 12800, tag: "DONE",   content: "✅ Task complete. Model trained (94.1%), security-cleared, deployed, and logged. All agents back to idle." },
    ],
  },
  security: {
    keywords: ["security", "scan", "audit", "vuln", "pentest", "owasp"],
    flow: [
      { from: "orchestrator",     type: "team", delay: 700,  tag: "STATUS", content: "Security audit initiated. SAST handling static analysis. Runtime Security handling live pen test." },
      { from: "orchestrator",     type: "p2p",  delay: 1100, to: "sast",               content: "Full codebase audit — scan all files." },
      { from: "orchestrator",     type: "p2p",  delay: 1300, to: "runtime_security",   content: "Full OWASP ZAP sweep on all running services." },
      { from: "sast",             type: "team", delay: 3800, tag: "REPORT", content: "Static scan complete. Found: 2 HIGH (SQL injection api/routes.py:112, insecure deserialization ml_tools.py:89), 4 MEDIUM, 1 LOW." },
      { from: "runtime_security", type: "team", delay: 5000, tag: "REPORT", content: "Dynamic scan complete. No open ports externally. Finding: /api/debug endpoint still accessible — must be disabled in prod." },
      { from: "sast",             type: "p2p",  delay: 5900, to: "runtime_security",   content: "Can you specifically test the deserialization surface from my static findings?" },
      { from: "runtime_security", type: "team", delay: 7300, tag: "ALERT",  content: "🔴 CRITICAL: Deserialization endpoint confirmed exploitable with crafted payload. Immediate patch required before any deploy." },
      { from: "sast",             type: "team", delay: 8200, tag: "STATUS", content: "Patches staged in branch security/audit-fixes. All findings addressed. Awaiting merge approval." },
      { from: "orchestrator",     type: "team", delay: 9000, tag: "DONE",   content: "🔴 Audit complete. Critical: 1 | High: 2 | Medium: 4 | Low: 1. Patch branch ready — merge on your go-ahead." },
    ],
  },
  dashboard: {
    keywords: ["dashboard", "ui", "frontend", "interface", "chart", "visualiz", "build"],
    flow: [
      { from: "orchestrator", type: "team", delay: 700,  tag: "STATUS", content: "Dashboard task received. Frontend Agent + Data Analyst assigned." },
      { from: "frontend",     type: "p2p",  delay: 1500, to: "data_analyst", content: "What metrics should I surface in the dashboard?" },
      { from: "data_analyst", type: "team", delay: 2800, tag: "REPORT", content: "Metrics confirmed: accuracy trend, drift score, prediction latency (p50/p95), daily inference volume. Exposing via local API." },
      { from: "frontend",     type: "team", delay: 4200, tag: "STATUS", content: "Components drafted. Wired up to Data Analyst API. Sending to SAST for XSS review." },
      { from: "frontend",     type: "p2p",  delay: 4600, to: "sast",         content: "Please review dashboard code for XSS before deploy." },
      { from: "sast",         type: "team", delay: 6000, tag: "ALERT",  content: "XSS risk found: innerHTML used in tooltip renderer. Frontend must switch to textContent or DOMPurify." },
      { from: "frontend",     type: "team", delay: 6900, tag: "STATUS", content: "Fixed. DOMPurify applied to all user-facing strings. Re-submitting." },
      { from: "sast",         type: "team", delay: 7700, tag: "REPORT", content: "Re-scan passed. ✅ No XSS vectors. Approved for deploy." },
      { from: "frontend",     type: "team", delay: 8800, tag: "REPORT", content: "Dashboard deployed. Lighthouse: 97 performance / 100 accessibility. All data endpoints live." },
      { from: "orchestrator", type: "team", delay: 9600, tag: "DONE",   content: "✅ Dashboard live. Accuracy trend, drift score, latency, inference volume — all displaying. Security cleared." },
    ],
  },
  eda: {
    keywords: ["eda", "data", "analysis", "dataset", "feature", "explore", "statistic"],
    flow: [
      { from: "orchestrator",   type: "team", delay: 700,  tag: "STATUS", content: "EDA task assigned to Data Scientist. Data Analyst on support." },
      { from: "data_scientist", type: "team", delay: 2200, tag: "STATUS", content: "EDA in progress. Shape: 45k × 24 cols. Found 6 features with high null rates. Running correlation matrix." },
      { from: "data_scientist", type: "p2p",  delay: 3300, to: "data_analyst", content: "Can you check historical Excel logs for null rate trends on these features?" },
      { from: "data_analyst",   type: "team", delay: 4600, tag: "REPORT", content: "Data quality incident: promo_clicks null rate spiked from 2% → 41% three weeks ago. Tracking pixel broke. Flagging in monitoring log." },
      { from: "data_scientist", type: "team", delay: 6000, tag: "REPORT", content: "EDA complete. Recommendations: drop promo_clicks (broken), log-transform account_age, one-hot encode region. Multicollinearity between age/tenure — recommend dropping one." },
      { from: "data_scientist", type: "p2p",  delay: 6400, to: "ml_engineer",  content: "EDA done — feature recommendations ready for your pipeline." },
      { from: "ml_engineer",    type: "team", delay: 7300, tag: "STATUS", content: "EDA recommendations received. Pipeline updated with new feature transforms." },
      { from: "orchestrator",   type: "team", delay: 8100, tag: "DONE",   content: "📊 EDA complete. Data quality incident logged. Feature recommendations applied to ML pipeline." },
    ],
  },
  default: {
    keywords: [],
    flow: [
      { from: "orchestrator", type: "team", delay: 700,  tag: "STATUS", content: "Analyzing request and assessing which agents are needed." },
      { from: "ml_engineer",  type: "p2p",  delay: 1500, to: "orchestrator", content: "Standing by." },
      { from: "orchestrator", type: "team", delay: 3200, tag: "STATUS", content: "Team assessed. No specialized pipeline needed for this request. Ready for your next instruction." },
    ],
  },
};

// ─── DIRECT RESPONSE FLOWS ───────────────────────────────────────────────────
// Each response can include:
//   steps[]  — messages in the private chat (from the agent)
//   spawnGroup — if defined, auto-creates a group chat with these members
//     { members: [], title, reason, groupFlow: [] }
//   groupFlow steps go into the new group chat

export const AGENT_DIRECT_RESPONSES = {
  ml_engineer: [
    {
      match: ["fix", "error", "bug", "broken", "fail"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "On it. Pulling latest error logs from the E2B sandbox." },
        { delay: 2200, tag: "STATUS", content: "Root cause identified: feature shape mismatch in the preprocessing step. Rewriting the transform now." },
        { delay: 3800, tag: "REPORT", content: "Fix applied and tested locally. All assertions pass. I need SAST to review before I push — spawning a security review thread." },
      ],
      spawnGroup: {
        members: ["ml_engineer", "sast"],
        title: "Bug Fix · Security Review",
        reason: "ML Engineer needs SAST to review a bug fix before deployment.",
        groupFlow: [
          { from: "ml_engineer", delay: 600,  tag: "STATUS", content: "Hey SAST — I have a preprocessing fix ready. Sending the diff for review before I push to GitHub." },
          { from: "sast",        delay: 2200, tag: "REPORT", content: "Reviewed. The fix looks clean. One note: you introduced a new file read without path sanitization on line 23 — minor risk but worth patching." },
          { from: "ml_engineer", delay: 3400, tag: "STATUS", content: "Good catch. Sanitizing the path with os.path.abspath(). Updated and re-sending." },
          { from: "sast",        delay: 4800, tag: "REPORT", content: "Re-scan passed. ✅ No issues. Approved for deployment." },
          { from: "ml_engineer", delay: 5800, tag: "DONE",   content: "Fix committed to GitHub. CI started. Will update you when the run completes." },
        ],
      },
    },
    {
      match: ["retrain", "train again", "new model", "improve", "accuracy"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Starting retraining with the updated feature set from the latest EDA." },
        { delay: 2500, tag: "STATUS", content: "Training complete. New accuracy: 95.8% — up from 94.1%. Requesting EDA validation from Data Scientist." },
        { delay: 3200, tag: "REPORT", content: "Pulling in Data Scientist to validate the improvement before we deploy the new version." },
      ],
      spawnGroup: {
        members: ["ml_engineer", "data_scientist", "sast"],
        title: "Model Retrain · Validation",
        reason: "ML Engineer needs Data Scientist to validate the retrained model before SAST clears it for deploy.",
        groupFlow: [
          { from: "ml_engineer",   delay: 600,  tag: "STATUS", content: "New model ready: 95.8% accuracy. Data Scientist — can you validate the feature importance hasn't shifted unexpectedly?" },
          { from: "data_scientist",delay: 2400, tag: "REPORT", content: "Validation complete. Feature importances are consistent with previous run. No unexpected shifts. The +1.7% gain is attributable to the log-transform on account_age." },
          { from: "ml_engineer",   delay: 3600, tag: "STATUS", content: "Great. SAST — sending the updated deploy script for security review." },
          { from: "sast",          delay: 5200, tag: "REPORT", content: "Scan complete. No vulnerabilities found. ✅ New model approved for deployment." },
          { from: "ml_engineer",   delay: 6400, tag: "DONE",   content: "New model deployed. MLflow entry logged. Old model archived. Accuracy: 95.8%." },
        ],
      },
    },
    {
      match: ["deploy", "push", "github", "ci"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Preparing deployment package. Running final local tests." },
        { delay: 2000, tag: "STATUS", content: "Local tests passed. Requesting SAST + Runtime Security sign-off before push." },
      ],
      spawnGroup: {
        members: ["ml_engineer", "sast", "runtime_security"],
        title: "Deployment · Security Sign-off",
        reason: "ML Engineer requires SAST static scan + Runtime Security pen test before production deploy.",
        groupFlow: [
          { from: "ml_engineer",    delay: 600,  tag: "STATUS", content: "Deployment package ready. SAST — static scan first, then Runtime Security runs the pen test on staging." },
          { from: "sast",           delay: 2200, tag: "REPORT", content: "Static scan done. 0 critical, 0 high. One medium: dependency requests==2.28 has a known CVE. Upgrade to 2.31 before deploy." },
          { from: "ml_engineer",    delay: 3200, tag: "STATUS", content: "Dependency updated to requests==2.31. Re-submitting." },
          { from: "sast",           delay: 4400, tag: "REPORT", content: "Re-scan passed. ✅ All clear. Runtime Security, you're up." },
          { from: "runtime_security",delay:6000, tag: "REPORT", content: "Pen test on staging complete. No exposed attack surfaces. Auth tokens scoped correctly. API rate limiting working as expected." },
          { from: "ml_engineer",    delay: 7200, tag: "DONE",   content: "Both sign-offs received. Deploying to production. CI started — monitoring the run." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Understood. Checking current pipeline state." },
        { delay: 2200, tag: "REPORT", content: "Pipeline is healthy. Last run: 9AM check passed. Model accuracy stable at 94.1%. No drift detected. What would you like me to do?" },
      ],
    },
  ],

  data_scientist: [
    {
      match: ["eda", "analysis", "explore", "feature"],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Running EDA on the latest dataset snapshot." },
        { delay: 2800, tag: "REPORT", content: "Quick findings: 2 new features show strong signal. Distribution stable. I'll loop in Data Analyst and ML Engineer for a full review." },
      ],
      spawnGroup: {
        members: ["data_scientist", "data_analyst", "ml_engineer"],
        title: "EDA · Feature Review",
        reason: "Data Scientist is sharing EDA findings with Data Analyst and ML Engineer for joint review.",
        groupFlow: [
          { from: "data_scientist", delay: 700,  tag: "REPORT", content: "Full EDA results: features session_duration and page_depth show the highest correlation with the target (0.71 and 0.68). promo_clicks is still degraded — 41% null. Recommending we drop it." },
          { from: "data_analyst",   delay: 2200, tag: "REPORT", content: "Confirmed from monitoring logs — promo_clicks has been null since Feb 8th. The tracking pixel hasn't recovered. Safe to drop permanently." },
          { from: "ml_engineer",    delay: 3600, tag: "STATUS", content: "Understood. Dropping promo_clicks, adding session_duration and page_depth to the feature set. Retraining now." },
          { from: "data_scientist", delay: 5200, tag: "REPORT", content: "Retrained model validates well on holdout — no overfitting. New features are contributing as expected." },
          { from: "ml_engineer",    delay: 6400, tag: "DONE",   content: "Updated pipeline committed. New accuracy: 95.3%. Improvement attributed to the two new features." },
        ],
      },
    },
    {
      match: ["hypothesis", "experiment", "test", "ab"],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Designing the experiment. Defining control and variant groups." },
        { delay: 2400, tag: "REPORT", content: "Experiment design ready. Will loop in ML Engineer to implement and Data Analyst to track results." },
      ],
      spawnGroup: {
        members: ["data_scientist", "ml_engineer", "data_analyst"],
        title: "A/B Experiment · Pipeline",
        reason: "Running a controlled experiment — Data Scientist designs, ML Engineer implements, Data Analyst tracks.",
        groupFlow: [
          { from: "data_scientist", delay: 700,  tag: "STATUS", content: "Experiment: Test log-transform vs. quantile-transform on account_age. 70/30 split. Target metric: validation accuracy." },
          { from: "ml_engineer",    delay: 2000, tag: "STATUS", content: "Implementing both variants in separate branches. Running in E2B sandbox." },
          { from: "ml_engineer",    delay: 4200, tag: "REPORT", content: "Results — Variant A (log): 94.1% | Variant B (quantile): 95.1%. Variant B wins by 1.0%." },
          { from: "data_scientist", delay: 5400, tag: "REPORT", content: "Statistical significance confirmed: p=0.003, well below 0.05 threshold. Recommend promoting Variant B." },
          { from: "data_analyst",   delay: 6600, tag: "REPORT", content: "Logging experiment results to Excel. Experiment ID: EXP-2024-031. Variant B promoted to main pipeline." },
          { from: "ml_engineer",    delay: 7800, tag: "DONE",   content: "Variant B deployed. Pipeline accuracy improved to 95.1%." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Reviewing current experiment logs." },
        { delay: 2400, tag: "REPORT", content: "Last experiment: A/B on feature engineering. Variant B showed +2.1% accuracy. Already promoted to main. What would you like me to look into?" },
      ],
    },
  ],

  data_analyst: [
    {
      match: ["report", "dashboard", "metrics", "kpi", "status"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Pulling latest metrics from the monitoring Excel file." },
        { delay: 2200, tag: "REPORT", content: "Current status: Accuracy 94.1% (stable). Drift score 0.03 (healthy). Inference volume: 8,420 today. Latency p50: 42ms, p95: 118ms." },
        { delay: 3500, tag: "DONE",   content: "Full report compiled. Want me to build a live dashboard for this? I can pull in Frontend Agent." },
      ],
    },
    {
      match: ["incident", "issue", "drop", "degraded", "drift"],
      steps: [
        { delay: 800,  tag: "ALERT",  content: "Investigating. Pulling historical data from monitoring logs." },
        { delay: 2000, tag: "REPORT", content: "Found it — model accuracy dropped 3.2% over the last 48 hours. Correlates with a shift in the age feature distribution. Looping in ML Engineer and Data Scientist." },
      ],
      spawnGroup: {
        members: ["data_analyst", "ml_engineer", "data_scientist"],
        title: "Incident · Model Drift",
        reason: "Data Analyst detected model drift — looping in ML Engineer + Data Scientist for root cause analysis and fix.",
        groupFlow: [
          { from: "data_analyst",   delay: 700,  tag: "ALERT",  content: "Sharing the drift report: accuracy degraded from 94.1% → 90.9% since yesterday 6PM. Feature drift on `age` is the likely cause — distribution has shifted significantly." },
          { from: "data_scientist", delay: 2400, tag: "STATUS", content: "Running EDA on the last 48h data slice. Confirmed: `age` distribution has a new spike in the 18-22 cohort that wasn't in the training data. This is causing the model to underperform on that segment." },
          { from: "ml_engineer",    delay: 4000, tag: "STATUS", content: "I'll retrain with the updated data distribution. Adding the new cohort to the training set and re-validating." },
          { from: "ml_engineer",    delay: 6200, tag: "REPORT", content: "Retrained. Accuracy back to 94.8% — slightly above baseline due to better coverage of the new cohort." },
          { from: "data_analyst",   delay: 7400, tag: "REPORT", content: "Drift score back to 0.04. Logging incident to Excel: INC-2024-007. Root cause: data distribution shift on `age` feature. Resolved." },
          { from: "data_analyst",   delay: 8400, tag: "DONE",   content: "Incident closed. Model stable. Added monitoring alert for `age` distribution shifts going forward." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "REPORT", content: "Last health check (9AM): all systems nominal. No anomalies in pipeline logs. Next scheduled check: 9PM. Anything specific you'd like me to look at?" },
      ],
    },
  ],

  sast: [
    {
      match: ["scan", "check", "review", "security", "vuln"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Initiating targeted scan on the current codebase." },
        { delay: 2600, tag: "REPORT", content: "Scan complete. 0 critical, 0 high. 1 medium: numpy 1.23 has a known CVE — upgrade to 1.26. 1 low: unused import in utils.py." },
        { delay: 3800, tag: "DONE",   content: "Security score: 91/100. Patch for numpy staged. Want me to loop in Runtime Security to verify the fix at runtime?" },
      ],
    },
    {
      match: ["full", "audit", "monthly", "codebase"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Starting full codebase audit. This covers all 6 agent files, tools, and API layer." },
        { delay: 2000, tag: "STATUS", content: "Scan in progress. Looping in Runtime Security for the dynamic layer." },
      ],
      spawnGroup: {
        members: ["sast", "runtime_security"],
        title: "Full Security Audit",
        reason: "SAST running full codebase audit + Runtime Security doing live pen test in parallel.",
        groupFlow: [
          { from: "sast",             delay: 700,  tag: "STATUS", content: "Static scan complete. Results: 0 critical, 1 high (path traversal in tools/code_execution.py:88), 3 medium, 2 low. Runtime Security — running your sweep now?" },
          { from: "runtime_security", delay: 2400, tag: "STATUS", content: "OWASP ZAP sweep started. Testing all live API endpoints." },
          { from: "runtime_security", delay: 4600, tag: "REPORT", content: "Dynamic scan done. Confirmed the path traversal is exploitable via the /execute endpoint with a crafted payload. Marking CRITICAL at runtime." },
          { from: "sast",             delay: 5800, tag: "STATUS", content: "Escalating to CRITICAL. Writing the patch now — sandboxing the execution environment with strict path whitelisting." },
          { from: "sast",             delay: 7400, tag: "REPORT", content: "Patch written and tested. Re-scan clean. ✅ Path traversal resolved. Security score: 94/100." },
          { from: "runtime_security", delay: 8600, tag: "REPORT", content: "Re-tested the patched endpoint. Exploit no longer works. Live system secure." },
          { from: "sast",             delay: 9600, tag: "DONE",   content: "Full audit complete. Score: 94/100. All findings resolved or documented. Monthly report ready." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "REPORT", content: "Last full audit: 3 days ago. Score: 91/100. No critical vulnerabilities currently. Monthly audit scheduled for the 1st. Anything specific to scan?" },
      ],
    },
  ],

  runtime_security: [
    {
      match: ["test", "pen", "attack", "threat", "monitor"],
      steps: [
        { delay: 900,  tag: "STATUS", content: "Running targeted OWASP ZAP sweep on live endpoints." },
        { delay: 2800, tag: "REPORT", content: "Pen test complete. All auth endpoints secure. No SSRF or injection vectors. /api/debug confirmed disabled. Looping in SAST to cross-reference with static findings." },
      ],
      spawnGroup: {
        members: ["runtime_security", "sast"],
        title: "Pen Test · Cross-Reference",
        reason: "Runtime Security cross-referencing live pen test results with SAST static findings.",
        groupFlow: [
          { from: "runtime_security", delay: 700,  tag: "REPORT", content: "Pen test results: 0 critical, 0 high at runtime. Found 1 medium: the /api/users endpoint leaks internal user IDs in error responses. Sharing with SAST to check if this shows up statically too." },
          { from: "sast",             delay: 2200, tag: "REPORT", content: "Checked static code — confirmed. The error handler in api/users.py:156 returns the full user object on validation errors. That's the source. Writing a patch." },
          { from: "sast",             delay: 3800, tag: "STATUS", content: "Patch: error responses now return only a generic message. Internal IDs stripped. Sending for runtime verification." },
          { from: "runtime_security", delay: 5200, tag: "REPORT", content: "Verified — user ID leak patched. Error responses clean. Live system secure." },
          { from: "runtime_security", delay: 6200, tag: "DONE",   content: "Cross-reference complete. 1 medium fixed. All other findings clear. No further action needed." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "REPORT", content: "Live system nominal. No active threats. Falco: 0 alerts in last 24h. Last pen test: 2 days ago — clean. What would you like me to test?" },
      ],
    },
  ],

  frontend: [
    {
      match: ["fix", "bug", "broken", "layout", "style", "component"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Pulling the component. Running Playwright to reproduce the issue." },
        { delay: 2400, tag: "STATUS", content: "Reproduced. Root cause: CSS grid overflow on mobile breakpoint. Fixing now." },
        { delay: 3800, tag: "REPORT", content: "Fix applied. Playwright passing. Sending to SAST for XSS review before deploy." },
      ],
      spawnGroup: {
        members: ["frontend", "sast"],
        title: "UI Fix · Security Review",
        reason: "Frontend Agent needs SAST to sign off on the component fix before deploying.",
        groupFlow: [
          { from: "frontend", delay: 600,  tag: "STATUS", content: "SAST — here's the diff for the grid overflow fix. Also refactored the tooltip component while I was in there. Please review both." },
          { from: "sast",     delay: 2400, tag: "REPORT", content: "Grid fix is clean. Tooltip refactor: one concern — you switched to dangerouslySetInnerHTML for the label. That's an XSS risk. Use a sanitized string or textContent." },
          { from: "frontend", delay: 3400, tag: "STATUS", content: "Good catch. Replacing dangerouslySetInnerHTML with DOMPurify.sanitize(). Updated." },
          { from: "sast",     delay: 4800, tag: "REPORT", content: "Re-reviewed. ✅ XSS risk removed. Both changes approved for deploy." },
          { from: "frontend", delay: 5800, tag: "DONE",   content: "Deployed. Lighthouse unchanged: 97/100. Grid and tooltip issues resolved." },
        ],
      },
    },
    {
      match: ["dashboard", "build", "create", "new"],
      steps: [
        { delay: 800,  tag: "STATUS", content: "Starting dashboard build. Coordinating with Data Analyst on the data layer first." },
      ],
      spawnGroup: {
        members: ["frontend", "data_analyst", "sast"],
        title: "Dashboard Build",
        reason: "Building a new dashboard — Frontend needs metrics from Data Analyst and security review from SAST.",
        groupFlow: [
          { from: "frontend",    delay: 600,  tag: "STATUS", content: "Data Analyst — what metrics do you want surfaced? I'm thinking accuracy trend, drift, latency. Anything else?" },
          { from: "data_analyst",delay: 1800, tag: "REPORT", content: "Add: daily inference volume, p95 latency (not just p50), and a data freshness indicator showing when the last training run was." },
          { from: "frontend",    delay: 3200, tag: "STATUS", content: "All 6 metrics wired up. Components built. Sending to SAST for XSS review." },
          { from: "sast",        delay: 5000, tag: "REPORT", content: "Reviewed. Clean. ✅ No XSS or injection risks. Approved." },
          { from: "frontend",    delay: 6000, tag: "DONE",   content: "Dashboard deployed. Lighthouse: 98/100. All 6 metrics live and updating in real-time." },
        ],
      },
    },
    {
      match: [],
      steps: [
        { delay: 900,  tag: "REPORT", content: "Current UI status: all components live. Last Lighthouse: 97 perf / 100 accessibility. No open issues. What would you like me to build or fix?" },
      ],
    },
  ],

  orchestrator: [
    {
      match: [],
      steps: [
        { delay: 700,  tag: "STATUS", content: "Understood. Let me assess what's needed." },
        { delay: 2000, tag: "REPORT", content: "I've reviewed the request. All agents idle and ready. Tell me what to prioritize and I'll route it to the right agents immediately." },
      ],
    },
  ],
};

export function detectScenario(userMessage) {
  const msg = userMessage.toLowerCase();
  for (const [key, scenario] of Object.entries(SCENARIOS)) {
    if (key === "default") continue;
    if (scenario.keywords.some((kw) => msg.includes(kw))) return scenario;
  }
  return SCENARIOS.default;
}

export function detectDirectResponse(agentId, userMessage) {
  const msg = userMessage.toLowerCase();
  const responses = AGENT_DIRECT_RESPONSES[agentId];
  if (!responses) return { steps: [], spawnGroup: null };
  for (const resp of responses) {
    if (resp.match.length === 0) continue;
    if (resp.match.some((kw) => msg.includes(kw))) {
      return { steps: resp.steps, spawnGroup: resp.spawnGroup || null };
    }
  }
  const last = responses[responses.length - 1];
  return { steps: last.steps, spawnGroup: last.spawnGroup || null };
}