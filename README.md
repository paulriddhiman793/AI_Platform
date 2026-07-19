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

A secure, autonomous multi-agent ML workspace with a real-time, glassmorphic web UI. Users upload datasets, run analysis and training, and download outputs from a platform-managed, per-user isolated project folder.

---

## 🚀 What This Platform Does

- **Multi-Agent Orchestration**: Coordinates **Data Scientist**, **Data Analyst**, and **ML Engineer** agents through a single real-time UI.
- **Real-Time Streaming & Local Pairing**: Streams logs, files, and status indicators (`Live Agents` vs `Mock Mode`) over WebSockets, with optional local **ML Engineer Worker** pairing for local machine execution.
- **Git & GitHub Integration**: Features a dedicated GitHub agent supporting automated branching, merging, and pushing directly to remote repositories.

---

## 🏛 Architecture Diagram

```
                 +-----------------------------+
                 |         Frontend UI         |
                 |  React + Vite (Vercel/Local)|
                 +--------------+--------------+
                                | WebSocket + HTTP (Rate Limited & Protected)
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

---

## 🔐 Key Features & Hardening

### 1. Authentication & API Hardening (`api/`)
- **Rigorous Pydantic v2 Schemas**: Strict input validation and sanitization across all REST (`/auth/*`, `/files`, `/predict`, `/metrics`) and WebSocket (`/ws`) endpoints.
- **PBKDF2 Password Hashing**: Uses `PBKDF2HMAC` with SHA-256 and **600,000 iterations** combined with automatic **lazy re-hashing** on login and minimum 12-character password complexity enforcement.
- **Encryption at Rest**: Encrypts authentication tokens (`.auth_tokens.json`) and GitHub configuration (`shared/github_config.json`) using symmetric `Fernet` ciphers with automatic 24-hour token TTL pruning.
- **Rate Limiting (`slowapi`)**: Defends endpoints against brute-force and DDoS attacks:
  - `5 requests/minute` on `/auth/login` and `/auth/register`.
  - `20 requests/minute` on `/worker/exec` and `/worker/write_file`.
  - `60 requests/minute` on `/files`, `/predict`, `/metrics`, and `/worker/download`.
- **ProxyMessageBus (Redis / In-Memory Pub/Sub)**: Seamlessly routes real-time agent-to-agent and WebSocket messages via `RedisMessageBus` (`redis.asyncio`) when Redis is connected (`REDIS_URL`), falling back transparently to `InMemoryMessageBus` (`asyncio.Queue`) for zero-dependency local execution.
- **Command Allowlist & Subprocess Protection**: Restricts worker command execution (`/worker/exec`) to an explicit allowlist (`python`, `pytest`, `git`, `pip`, `node`, `npm`, `tsc`, `vite`) and eliminates command injection vulnerabilities in dynamic code runners.

### 2. Frontend UI/UX Polish (`gui/`)
- **Anti-Autofill Protection**: Employs a `readOnly` focus-flip mechanism and unique field naming to prevent browsers (Chrome, Edge, Firefox) from automatically exposing stored credentials on page load.
- **Persistent Header Controls**: Action buttons (`✨ New Chat`, `Connect GitHub`, `⚡ Get ML Engineer`, `📁 Access Files`) are permanently accessible from the top navigation bar with clear real-time status indicators (`⚡ Live Agents` vs `🤖 Mock Mode`).
- **Zero-Overlap Glassmorphic Layout**: Designed with custom glowing borders, ambient pulsing background lighting (`Aurora`, `ParticleBackground`), and responsive flexbox spacing to ensure zero text or element overlaps across all screen sizes.

### 3. Supply Chain & Vulnerability Management
- **Pre-Commit Secrets Scanning**: Pre-commit hooks (`gitleaks`, `trufflehog`, `ruff`, `bandit`) automatically intercept hardcoded secrets or vulnerable code before commit.
- **Automated Dependency Updates**: Scheduled weekly updates configured via `.github/dependabot.yml` across Python (`/`), Node.js (`/gui`), and GitHub Actions (`/`).
- **CI/CD Vulnerability & SBOM Scanning**: Every pull request and build automatically runs `google/osv-scanner-action` and `anchore/syft-action` to output `osv-scanner-results.json` and `sbom.spdx.json`.

---

## 🛠 Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Framer Motion
- **Backend**: Python 3.12, FastAPI, Starlette, SlowAPI, Cryptography (Fernet)
- **Storage**: Local per-user project folders under `platform_projects/` 
- **Auth & Database**: MongoDB (`pymongo`) + PBKDF2 HMAC SHA-256 + Encrypted Tokens
- **ML & Data Science**: `pandas`, `scikit-learn`, `numpy`
- **Security & Static Analysis**: `pytest`, `ruff`, `mypy`, `bandit`, `pre-commit`, `gitleaks`, `trufflehog`, `OSV-Scanner`, `Anchore Syft`

---

## ⚡ Quick Start (Local Development)

### 1) Start the Backend Server
```powershell
python -m api.main
```

### 2) Start the Frontend Dev Server
```powershell
cd gui
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🧪 Automated Testing Guide

We provide a comprehensive testing suite verifying unit logic, security rules, integration workflows, and production builds. For exact step-by-step instructions and command breakdowns, refer to **[`test_guide.txt`](file:///d:/Downloads/Agents/AI_Platform/test_guide.txt)**.

### Quick Verification Command (Run Unit & Security Tests + Frontend Build)
```powershell
pytest tests/unit -v ; cd gui ; npm run build ; cd ..
```

---

## 🔄 User Flow & Workflow Example

1. **Log In / Register**: Secure authentication with strict complexity checks.
2. **Start Workspace**: Click **`✨ New Chat`** to create an isolated project directory.
3. **Upload Dataset**: Click **`📁 Upload CSV Dataset`** to attach your tabular data.
4. **Analyze**: Send prompt `analyse data` to trigger the Data Scientist and Data Analyst agents.
5. **Train Model**: Send prompt `train model` to instruct the ML Engineer agent to train and compare models on raw and feature-engineered datasets.
6. **Access & Download**: Click **`📁 Access Files`** to browse reports, generated graphs, engineered feature CSVs, and download a complete ZIP bundle.

---

## 🌐 Environment Variables

### Backend (`.env` at repo root):
```ini
HOST=0.0.0.0
PORT=8000
PUBLIC_BASE_URL=http://localhost:8000
FRONTEND_ORIGINS=http://localhost:5173
PLATFORM_STORAGE_ROOT=./platform_projects
MONGO_URI=mongodb://localhost:27017
MONGO_DB=ai_platform
GITHUB_REPO_URL=https://github.com/<owner>/<repo>.git
```

### Frontend (`gui/.env`):
```ini
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

---

## 📁 Repository Structure

```
├── agents/             # Autonomous agent implementations (Scientist, Analyst, Engineer, GitHub)
├── api/                # FastAPI server, security middleware, rate limiting, encryption & WebSocket
├── gui/                # React + Vite + TypeScript glassmorphic web frontend
├── tools/              # Workspace isolation, file management, and utility functions
├── tests/              # Comprehensive pytest suite (unit, security, integration, contract)
├── .github/            # GitHub Actions CI workflows (OSV-Scanner, Syft SBOM, Dependabot, HF deploy)
├── test_guide.txt      # Step-by-step guide to running every automated test across the project
└── README.md           # Project documentation
```
