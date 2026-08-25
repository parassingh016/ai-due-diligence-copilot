"""
main.py
FastAPI backend for the AI Due Diligence Copilot.
Run with: uvicorn main:app --reload --port 8000
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ingest import ingest_document, list_documents, delete_document, extract_text
from rag_pipeline import (
    generate_answer,
    generate_executive_summary,
    generate_risk_scores,
    generate_ats_score,
)
from report_generator import generate_pdf_report

app = FastAPI(title="AI Due Diligence Copilot API")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    doc_filter: str | None = None
    n_results: int = 5


@app.get("/")
def root():
    return {"status": "AI Due Diligence Copilot API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        num_chunks = ingest_document(tmp_path, file.filename)
        os.unlink(tmp_path)

        return {
            "message": f"Successfully ingested '{file.filename}'",
            "chunks_created": num_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_documents(request: QueryRequest):
    """Ask a question across ingested documents."""
    try:
        answer, sources = generate_answer(
            request.question, request.n_results, request.doc_filter
        )
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/{doc_name}")
def get_summary(doc_name: str):
    """Generate an executive summary + risk assessment for a document."""
    try:
        summary = generate_executive_summary(doc_name)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk-score/{doc_name}")
def get_risk_score(doc_name: str):
    """Generate structured risk scores (Financial, Operational, Market, Regulatory) for a document."""
    try:
        risk_data = generate_risk_scores(doc_name)
        return risk_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export-pdf/{doc_name}")
def export_pdf(doc_name: str):
    """Generate and download a PDF report combining the executive summary and risk scores."""
    try:
        summary = generate_executive_summary(doc_name)
        risk_data = generate_risk_scores(doc_name)
        pdf_path = generate_pdf_report(doc_name, summary, risk_data)

        safe_name = doc_name.rsplit(".", 1)[0].replace(" ", "_")
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{safe_name}_due_diligence_report.pdf",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def get_documents():
    """List all ingested documents."""
    return {"documents": list_documents()}


@app.delete("/documents/{doc_name}")
def remove_document(doc_name: str):
    """Delete a document and its chunks from the vector store."""
    delete_document(doc_name)
    return {"message": f"Deleted '{doc_name}'"}


@app.post("/ats-score")
async def ats_score(
    resume: UploadFile = File(...),
    job_description: str = Form(default=""),
):
    """
    Score a resume's ATS (Applicant Tracking System) compatibility.
    Analyzed directly — not stored in the vector database like other documents.
    """
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await resume.read()
            tmp.write(content)
            tmp_path = tmp.name

        resume_text = extract_text(tmp_path)
        os.unlink(tmp_path)

        if not resume_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in the resume (it may be a scanned image).",
            )

        result = generate_ats_score(resume_text, job_description)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))