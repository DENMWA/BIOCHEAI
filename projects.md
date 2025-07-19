
# 🧬 BioCheAi Development Roadmap

This file tracks the development roadmap and GitHub project board structure for the BioCheAi platform.

---

## ✅ CURRENT STAGE: v5.0 (Self-Learning AI Platform)

### 🚀 Core Features Completed
- Modular architecture (DNA, RNA, PTM modules)
- Streamlit UI with multi-input support
- ML-powered PTM prediction logic
- Initial deployment config (Render, Docker, GitHub-ready)

---

## 🧠 IN PROGRESS

### 🔄 Self-Retraining + AutoML Engine
- [x] User data ingestion and logging
- [x] Scheduled model retraining via `trigger.py`
- [ ] Model benchmarking and promotion logic
- [ ] Integration with TPOT or AutoSklearn

### 📥 Open-Source Dataset Ingestion
- [x] Parser engine for public sources (PhosphoSitePlus, RMBase)
- [ ] Cron job or webhook integration
- [ ] Versioned public datasets tracker

### 📊 Admin Dashboard (Streamlit)
- [x] Model version tracker
- [ ] Real-time training monitor
- [ ] Manual approve/reject interface for new models

---

## 🧩 FUTURE PHASES (v5.1+)

### 🌐 API + Platform Expansion
- [ ] REST API endpoints for all modules
- [ ] OAuth2 login support
- [ ] Team collaboration & shared datasets

### 💊 Clinical & Diagnostic Extension
- [ ] ACMG variant classification engine
- [ ] Disease panel builder
- [ ] Drug repurposing insights from PTM profiles

---

## 📁 GitHub Project Board
> Create this under your repo > Projects > "Roadmap v5.0"

| Column | Cards |
|--------|-------|
| 🔨 Backlog | Ideas and enhancements |
| 🚧 In Progress | Currently being implemented |
| ✅ Done | Completed and merged features |

---

## 📅 Suggested Milestones
- `v5.0-beta` — Initial ML-powered build with self-training
- `v5.1` — Full AutoML + admin controls
- `v5.2` — Clinical annotation layer
- `v6.0` — Collaborative and API-enabled research platform

