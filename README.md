---
title: AI Platform
emoji: "🤖"
colorFrom: "gray"
colorTo: "blue"
sdk: "docker"
app_file: Dockerfile
pinned: false
---
# AI Engineering Platform

A multi-agent ML workspace with a real-time web UI. Users upload datasets, run analysis and training, and download outputs from a platform-managed project folder.

## What This Platform Does

- Orchestrates Data Scientist, Data Analyst, and ML Engineer tasks through a single UI
- Stores all outputs in a per-project folder under platform storage
- Streams status, files, and reports to the UI in real time
- Provides a built-in file browser with ZIP download
- Supports GitHub push/merge via a dedicated GitHub agent
- Enforces per-user project isolation

## Architecture Diagram

```
                 +-----------------------------+
                 |         Frontend UI         |
                 |  React + Vite (Vercel/Local)|
                 +--------------+--------------+
                                | WebSocket + HTTP
                                v
                     +-----------------------+
                     |     FastAPI Backend   |
                     |  Auth + WS + REST     |
                     +-----------+-----------+
                                 | message bus
                                 v
        +---------------+----------------+----------------+
        | Data Scientist| Data Analyst   | ML Engineer    |
        +---------------+----------------+----------------+
                                 |
                                 v
                     +-----------------------+
                     |   Platform Projects   |
                     | (files, reports, ZIP) |
                     +-----------------------+
```

## Tech Stack

- Frontend: React, Vite
- Backend: FastAPI, WebSocket
- Storage: Local project folders under platform_projects/
- Auth: MongoDB (pymongo)
- ML/DS: Python, pandas, scikit-learn
- GitHub: REST API + git CLI

## Supported Data Types

- Supported now: tabular CSV datasets
- In progress: CNN (image), NLP (text), and hybrid datasets

## Quick Start (Local)

### 1) Backend

```bash
python -m api.main
```

### 2) Frontend

```bash
cd gui
npm install
npm run dev
```

Open http://localhost:5173

## How It Works (User Flow)

1. Log in
2. Create a new chat (New Chat button)
3. Upload a CSV dataset
4. Run analysis by sending: analyse data
5. Train models by sending: train model
6. View outputs via Access Files (preview or download ZIP)

### Example follow-up prompt
After training, ask:
```
Show training process for XGB on engineered dataset
```

## Example Workflow

1. Upload Housing.csv
2. Send analyse data
3. Data Scientist generates engineered dataset + suggestions
4. Data Analyst produces EDA reports + graphs
5. Send train model
6. ML Engineer trains on raw + engineered datasets
7. Compare metrics and download ZIP from Access Files

## Example Use Case

Housing Price Prediction

- Input: Housing.csv (tabular)
- Output:
  - EDA findings and graphs
  - Engineered features CSV
  - Model comparison report
  - Best-model summary
  - Downloadable ZIP with all artifacts

## Projects, Files, and History

- Each New Chat creates a new project folder under the platform storage root.
- Projects are owned by the logged-in user. You only see your own projects.
- The Projects list in the sidebar lets you switch between previous projects.
- Access Files opens an in-platform file browser and allows ZIP download.

## Dataset Caching

- If a dataset has already been feature-engineered, the Data Scientist reuses cached outputs.
- Cache is keyed by dataset hash and stored under platform storage .cache/engineered/.
- If cached suggestions are missing, FE is re-run and cache is refreshed.

## GitHub Integration

- Use Connect GitHub in the UI with a PAT and repo name.
- The GitHub agent will push:
  - main
  - agent/data_scientist
  - agent/data_analyst
  - agent/ml_engineer
  - agent/shared

If the remote repo already has commits, the agent merges remote history automatically before pushing.

## CI/CD Pipeline (GitHub Actions)

This repo includes two workflows:

1) CI (.github/workflows/ci.yml)
   - Installs backend dependencies and runs a Python syntax check.
   - Installs frontend dependencies and runs a Vite build.

2) Deploy Backend to Hugging Face (.github/workflows/deploy_hf_backend.yml)
   - Pushes the backend into your HF Space: rppooo/ai_platform
   - Uses a Docker Space based on deploy/hf_space/Dockerfile

### Required Secrets

Set this in GitHub ? Repo ? Settings ? Secrets and variables ? Actions:

- HF_TOKEN: a Hugging Face token with write access to the space.

### Notes

- The frontend remains on Vercel. Only the backend is deployed to HF Spaces.
- HF Space expects the backend to listen on port 7860 (handled via PORT env).

## Environment Variables

Backend (.env at repo root):

```
HOST=0.0.0.0
PORT=8000
PUBLIC_BASE_URL=https://your-public-backend-url   # optional but recommended for accurate startup logs
FRONTEND_ORIGINS=https://your-frontend-domain
PLATFORM_STORAGE_ROOT=./platform_projects
MONGO_URI=mongodb://localhost:27017
MONGO_DB=ai_platform
GITHUB_REPO_URL=https://github.com/<owner>/<repo>.git   # optional fallback
```

Frontend (Vercel or local .env in gui/):

```
VITE_API_URL=https://your-public-backend-url
VITE_WS_URL=wss://your-public-backend-url/ws
```

Ready-to-fill templates are included in:

- `.env.example`
- `gui/.env.example`

## Deploying Frontend (Vercel) with Local Backend

You can host the frontend on Vercel while the backend runs on your machine. You must expose the backend publicly (Cloudflare Tunnel or ngrok).

High-level steps:

1. Start a tunnel to your backend (example):
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
2. Set Vercel env vars:
   - VITE_API_URL=<tunnel_url>
   - VITE_WS_URL=<tunnel_url>/ws
3. Set backend CORS:
   - FRONTEND_ORIGINS=https://<your-vercel-app>.vercel.app

## Troubleshooting

- CORS error on login: verify FRONTEND_ORIGINS and restart backend.
- Tunnel can’t reach backend: use http://127.0.0.1:8000 in the tunnel URL (IPv4).
- Access Files shows Forbidden: log out/in or select the correct project.

## Repository Structure

```
agents/         Agent implementations
api/            FastAPI server + WebSocket
tools/          Workspace, RAG, utilities
gui/            React + Vite frontend
platform_projects/  Runtime project storage (ignored by git)
```


