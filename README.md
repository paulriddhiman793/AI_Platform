# AI Engineering Platform

A multi-agent AI engineering workspace where specialized agents collaborate to build, secure, deploy, and monitor ML-driven projects through a shared workspace and real-time GUI.

## What This Project Does

This platform simulates and coordinates a real software/ML team:

- Orchestrates multiple agents for ML, data, frontend, security, runtime validation, and GitHub automation.
- Writes all generated artifacts to a structured workspace on disk.
- Streams agent messages, p2p interactions, and file writes to a live React GUI over WebSocket.
- Supports both team-level task routing and direct/private agent instructions.
- Automates repository sync to GitHub after task completion (main push, per-agent branches, merge to main).

## Core Features

- Multi-agent orchestration with task routing and result aggregation
- Shared file workspace with per-agent folders
- FastAPI + WebSocket backend for real-time status and messaging
- React GUI with team chat, direct chats, group threads, and activity feed
- Security pipeline:
  - SAST static scans
  - Runtime security dynamic checks
- Frontend generation and local dev server automation
- GitHub automation agent:
  - Push full repo to `main`
  - Push each agent folder to `agent/<name>` branch
  - Merge all agent branches back into `main`

## Agent Roster

- `orchestrator`: Receives user intent, routes tasks, tracks completion
- `ml_engineer`: Builds pipeline/API/deployment assets
- `data_scientist`: EDA and feature recommendations
- `data_analyst`: Metrics, monitoring, incident/reporting
- `frontend`: Builds dashboard and frontend files
- `sast`: Static security scanning and remediation loop
- `runtime_security`: Runtime exploitability checks
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
6. Agents write files via `tools/workspace.py`.
7. File events and statuses stream back to GUI in real time.

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Git installed and authenticated (for GitHub sync)

## Environment Variables

Create a `.env` in repo root:

```env
GITHUB_REPO_URL=https://github.com/<owner>/<repo>.git
```

Notes:
- Startup validates `GITHUB_REPO_URL`.
- If missing/invalid, startup prompts you and writes back to `.env`.
- `GITHUB_REPO_URL` must be a **repository URL**, not a profile URL.

## Setup

### Backend

```bash
python -m api.main
```

Startup prompts:
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
- "run security audit"
- "build frontend dashboard"

Orchestrator assigns agents and summarizes outcomes.

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
   - `agent/frontend`
   - `agent/sast`
   - `agent/runtime_security`
   - `agent/shared`
4. Merge these branches into `main`.
5. Push merged `main`.
6. Write report: `github/sync_report.md` in workspace.

## Common Issues

- GUI connected but no real agent behavior:
  - Ensure backend is running via `python -m api.main`.
- GitHub sync fails:
  - Check `GITHUB_REPO_URL` points to an existing repo.
  - Ensure git authentication is configured.
- Frontend on wrong port in mock mode:
  - This occurs when backend is offline and GUI runs local mock responses.

## Current Status

This project is structured as a progressive platform with both real execution paths and mock-friendly UI behaviors. It is suitable for experimentation, demos, and incremental hardening into production-grade workflows.

## Development Phases

The platform is intentionally being built in phases.

### Phase 1 (Current)

- Core multi-agent architecture is implemented.
- Orchestrator routing is primarily rule/keyword-driven.
- Agents can coordinate, send p2p messages, and write structured workspace files.
- GUI + backend realtime loop is active.
- GitHub automation flow is integrated.
- Some behaviors are still deterministic/mock-style to stabilize orchestration first.

### Phase 2 (Next)

- LLMs become the primary reasoning "brains" for agents.
- Agent decisions, decomposition, and responses become model-driven.
- Tool-use patterns are expanded per agent role.
- More natural multi-step planning and contextual reasoning is introduced.

### Phase 3

- Real code generation, execution, and validation loops are expanded.
- Sandbox/tool integrations are hardened for actual build-test-fix cycles.
- Security review and remediation become deeper and less template-driven.

### Phase 4

- Memory/state layers are improved (persistent context beyond single run).
- Better long-horizon task tracking and cross-agent knowledge reuse.
- Improved reliability, retries, and failure recovery.

### Phase 5+

- Production hardening:
  - stronger observability
  - robust CI/CD integration
  - policy/permissions controls
  - scale/performance tuning
- Move from prototype-oriented behavior to deployment-grade reliability.
