# 📊 AI Due Diligence Copilot

A full-stack RAG (Retrieval-Augmented Generation) platform that analyzes company filings, financial statements, and investor documents to accelerate due diligence workflows. Delivers source-cited answers, executive summaries, structured risk scoring, and downloadable PDF reports — plus a bonus ATS resume scoring tool.

**Built on a 100% free stack** — no paid APIs, no credit card required.

🔗 **Live Demo:** [https://ai-due-diligence.streamlit.app](#) *(add your link once deployed)*
📂 **Source Code:** [github.com/YOUR_USERNAME/ai-due-diligence-copilot](#)

---

## Features

- 💬 **Document Q&A** — ask natural-language questions about uploaded filings, get answers cited back to the exact source chunk
- 📋 **Executive Summary Generator** — auto-generates a summary plus key financial risks and growth opportunities
- 📈 **Risk Dashboard** — structured risk scoring (0-10) across Financial, Operational, Market, and Regulatory categories, visualized with an interactive gauge and bar chart
- 📄 **PDF Report Export** — one-click download of a polished PDF combining the summary and risk breakdown
- 🎯 **ATS Resume Score Checker** — bonus tool: upload a resume (and optionally a job description) to get an ATS compatibility score, missing keywords, and improvement suggestions

---

## Architecture

```
Streamlit (frontend)  →  FastAPI (backend)  →  ChromaDB (vector store)
                                ↓                       ↑
                          Groq API (LLM)      sentence-transformers (embeddings)
```

The frontend never touches the LLM, database, or API keys directly — it only calls the backend's REST API. This mirrors how real production systems separate concerns between client and server.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Vector database | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) — runs locally, no API cost |
| LLM inference | Groq API (`openai/gpt-oss-20b`) — free tier, fast inference |
| PDF parsing | pdfplumber |
| PDF generation | reportlab |
| Charts | Plotly |
| Config | python-dotenv |

---

## Project Structure

```
ai-due-diligence-copilot/
├── requirements.txt
├── .gitignore
├── README.md
├── backend/
│   ├── main.py              # FastAPI app & API routes
│   ├── ingest.py             # PDF extraction, chunking, embedding, storage
│   ├── rag_pipeline.py       # Retrieval, answer generation, risk scoring, ATS scoring
│   ├── report_generator.py   # PDF report builder (reportlab)
│   └── .env                  # Your Groq API key (not committed to Git)
└── frontend/
    └── app.py                 # Streamlit UI
```

---

## Setup (Local)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-due-diligence-copilot.git
cd ai-due-diligence-copilot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — no credit card required.

### 5. Configure environment variables
Inside the `backend` folder, create a file named exactly `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

### 6. Run the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```
API docs available at `http://localhost:8000/docs`

### 7. Run the frontend (in a new terminal)
```bash
cd frontend
streamlit run app.py
```
App opens at `http://localhost:8501`

---

## Free Deployment

This project deploys across two free platforms:

**Backend → [Render](https://render.com)**
- New Web Service → connect this repo
- Root directory: `backend`
- Build command: `pip install -r ../requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add `GROQ_API_KEY` under Environment Variables

**Frontend → [Streamlit Community Cloud](https://share.streamlit.io)**
- New app → connect this repo
- Main file path: `frontend/app.py`
- Under Settings → Secrets, add:
  ```
  BACKEND_URL = "https://your-backend-name.onrender.com"
  ```

> Render's free tier sleeps after 15 minutes of inactivity — the first request after idle time may take 30-50 seconds to wake up.

---

## How to Use

1. Upload a company filing (PDF) in the sidebar
2. **Ask a Question** — get cited answers from the document
3. **Executive Summary** — generate a structured summary + export as PDF
4. **Risk Dashboard** — view risk scores across four categories with charts
5. **ATS Score Checker** — upload a resume to check ATS compatibility

---

## Screenshots

*(Add 2-3 screenshots here — Ask a Question, Risk Dashboard, and ATS Score Checker are the most visually compelling)*

---

## Limitations

- Only supports text-based PDFs (no OCR for scanned documents yet)
- No user authentication or multi-user document isolation
- No persistent chat history across sessions
- Risk and ATS scores are AI-generated heuristics, not certified assessments — always pair with human review

---

## Resume Description

> **AI Due Diligence Copilot** — Built a full-stack RAG platform (FastAPI, Streamlit, ChromaDB) that ingests financial filings and generates source-cited answers, executive summaries, and structured risk assessments with interactive visualizations and PDF export. Deployed live using free-tier infrastructure (Render + Streamlit Cloud) with local embeddings and Groq-hosted LLM inference — zero infrastructure cost.

---

## License

This project is open source and available for learning purposes.
