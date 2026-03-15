# AI Engineering Platform

A multi-agent AI engineering workspace where specialized agents collaborate to build, deploy, and monitor ML-driven projects through a shared workspace and real-time GUI.

## What This Project Does

This platform simulates and coordinates a real software/ML team:

- Orchestrates multiple agents for ML, data, and GitHub automation.
- Writes all generated artifacts to a structured workspace on disk.
- Streams agent messages, p2p interactions, and file writes to a live React GUI over WebSocket.
- Supports both team-level task routing and direct/private agent instructions.
- Automates repository sync to GitHub after task completion (main push, per-agent branches, merge to main).

## Core Features

- Multi-agent orchestration with task routing and result aggregation
- Shared file workspace with per-agent folders
- FastAPI + WebSocket backend for real-time status and messaging
- React GUI with team chat, direct chats, group threads, and activity feed
- Dataset upload from GUI (`.csv`) to `shared/datasets/`
- Automatic transparency run on uploaded dataset (backend-triggered)
- Transparency output persistence to `shared/output.txt`
- Hybrid RAG indexing over transparency output:
  - Dense retrieval via LanceDB
  - Lexical retrieval via TF-IDF
  - Hybrid scoring for better recall/precision
- Data Scientist/Data Analyst report generation now uses hybrid-RAG context from full transparency output
- Security pipeline: removed
- Frontend generation: disabled
- GitHub automation agent:
  - Push full repo to `main`
  - Push each agent folder to `agent/<name>` branch
  - Merge all agent branches back into `main`

## Agent Roster

- `orchestrator`: Receives user intent, routes tasks, tracks completion
- `ml_engineer`: Builds pipeline/API/deployment assets
- `data_scientist`: EDA and feature recommendations
- `data_analyst`: Metrics, monitoring, incident/reporting
- `github`: Repository sync, branch push, merge workflow

## Repository Structure

```text
agents/         Agent implementations
api/            FastAPI server, startup, message bus
tools/          Workspace manager and utilities
gui/            React + Vite frontend
test_api_key.py Standalone Groq API key test script
```

## Runtime Architecture

1. `api/main.py` starts the platform.
2. Workspace is configured and a project directory is created.
3. Agents subscribe to message-bus channels.
4. GUI connects via `ws://localhost:8000/ws`.
5. User messages are routed to orchestrator or a direct agent.
6. Optional: user uploads dataset from GUI (`dataset_upload` websocket event).
7. Backend stores dataset in `shared/datasets/`, runs transparency pipeline, writes `shared/output.txt`, then builds LanceDB + lexical hybrid index under `shared/rag/`.
8. Agents write files via `tools/workspace.py`.
9. File events and statuses stream back to GUI in real time.

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Git installed and authenticated (for GitHub sync)
- LanceDB Python package for vector indexing (`pip install lancedb`)

## Environment Variables

Create a `.env` in repo root (optional):

```env
GITHUB_REPO_URL=https://github.com/<owner>/<repo>.git
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_TOKENS=1200
GROQ_MAX_RETRIES=4
GROQ_MIN_INTERVAL_SEC=1.2
```

Notes:
- `GITHUB_REPO_URL` is optional if you use the GUI “Connect GitHub” flow.
- If both are present, `GITHUB_REPO_URL` overrides GUI settings.
- Groq settings are optional and only required for LLM-powered reporting.

## Setup

### Backend

```bash
python -m api.main
```

Startup prompts (only if CLI init is enabled):
- Output directory for generated projects
- Project name
- GitHub repo URL (if not valid in `.env`)

### GUI

```bash
cd gui
npm install
npm run dev
```

Open:
- GUI: `http://localhost:5173`
- Backend WS/API: `http://localhost:8000`

## How to Use

### Team Chat

Use Team chat for broad instructions like:
- "build a churn model and deploy it"
- "analyse data" (after uploading dataset)

Orchestrator assigns agents and summarizes outcomes.

### Dataset Upload + RAG Flow

1. Use the GUI upload button to upload a `.csv` dataset.
2. Backend silently:
   - saves dataset to `shared/datasets/<file>.csv`
   - runs transparency pipeline on that dataset
   - writes full output to `shared/output.txt`
   - builds hybrid RAG index in `shared/rag/` (LanceDB + TF-IDF)
3. When Data Scientist / Data Analyst run analysis, they query this hybrid RAG context to write richer reports grounded in the full transparency output.

### Direct/Private Chat

Message a specific agent directly for targeted work.

If backend is connected:
- Direct messages go to real backend agents.
- If multi-agent collaboration is required, a dedicated group thread is auto-created.
- Future tasks re-use the same group thread for the same exact agent set.

If backend is offline:
- GUI falls back to local mock flows.

## GitHub Automation Flow

After assigned agents finish, orchestrator triggers `github` agent to:

1. Ensure git repo exists at project root.
2. Commit and push full project to `main`.
3. Create/push per-agent branches:
   - `agent/ml_engineer`
   - `agent/data_scientist`
   - `agent/data_analyst`
   - `agent/shared`
4. Merge these branches into `main`.
5. Push merged `main`.
6. Write report: `github/sync_report.md` in workspace.

## Common Issues

- GUI connected but no real agent behavior:
  - Ensure backend is running via `python -m api.main`.
- GitHub sync fails:
  - If using GUI connect, confirm the token has repo permissions.
  - If using `.env`, check `GITHUB_REPO_URL` points to a valid repo.
- Frontend on wrong port in mock mode:
  - This occurs when backend is offline and GUI runs local mock responses.

## Current Status

This project is structured as a progressive platform with both real execution paths and mock-friendly UI behaviors. It is suitable for experimentation, demos, and incremental hardening into production-grade workflows.

## Development Phases

The platform is intentionally being built in phases. Each phase describes what is **implemented**, what is **optional**, and how it behaves at runtime.

### Phase 1 — Core Orchestration (Current)

Purpose: establish reliable multi-agent routing and file outputs.

- Orchestrator routes tasks via keyword-based intent detection.
- Agents coordinate and write artifacts into the workspace.
- GUI receives real-time messages and file events over WebSocket.
- GitHub automation can push and branch agent outputs.

### Phase 2 — LLM-Driven Reasoning (Optional / Deferred)

Purpose: enable model-driven decision making inside agents.

- Agents can use LLMs for richer planning and reporting.
- Tool-use patterns expand per agent role.
- This phase is optional and can be disabled entirely.

### Phase 3 — Execution Hardening (Implemented as "Check Agents")

Purpose: validate environment and workflows before running expensive tasks.

- A GUI "Check Agents" button runs Phase 3 checks in the project folder.
- Writes Phase 3 reports to the project workspace.
- Focuses on readiness and consistency rather than LLM intelligence.

### Phase 4 — State & Reliability (Active)

Purpose: track tasks, status, and durability across runs.

- Every agent logs task start/complete/error to shared state.
- State is persisted to `shared/state/` for inspection and reporting.
- A Phase 4 report can be generated from state logs.

### Phase 5 — Production Hardening (Planned)

Purpose: move from prototype workflow to production-grade reliability.

- Stronger observability and structured logging.
- Robust CI/CD integration and deployment checks.
- Policy/permission controls and scalability tuning.
