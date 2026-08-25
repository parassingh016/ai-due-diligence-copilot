"""
rag_pipeline.py
Handles: retrieval from ChromaDB + answer generation using Groq's free API
"""

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from ingest import embedder, collection

# Load .env from the same folder as this file, regardless of where the terminal was launched from
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        f"GROQ_API_KEY not found. Make sure a '.env' file exists at {env_path} "
        "with a line like: GROQ_API_KEY=gsk_your_key_here"
    )

groq_client = Groq(api_key=api_key)
MODEL_NAME = "openai/gpt-oss-20b"  # fast, available on personal Groq accounts


def retrieve_context(query: str, n_results: int = 5, doc_filter: str | None = None):
    """Retrieve the most relevant chunks for a query."""
    query_embedding = embedder.encode([query]).tolist()
    where_clause = {"source": doc_filter} if doc_filter else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where_clause,
    )
    return results


def generate_answer(query: str, n_results: int = 5, doc_filter: str | None = None):
    """Retrieve context and generate a cited answer."""
    results = retrieve_context(query, n_results, doc_filter)

    if not results["documents"] or not results["documents"][0]:
        return "No relevant documents found. Please upload a document first.", []

    chunks = results["documents"][0]
    sources = results["metadatas"][0]

    context = "\n\n".join(
        f"[Source: {s['source']}, chunk {s['chunk_id']}]\n{c}"
        for c, s in zip(chunks, sources)
    )

    prompt = f"""You are a due diligence analyst assistant. Answer the question using ONLY the context below.
Cite the source for every claim using this exact format: [Source: name, chunk X].
If the answer is not contained in the context, clearly say so instead of guessing.

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1500,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content, sources


def generate_executive_summary(doc_name: str):
    """Generate an executive summary, risks, and opportunities for a single document."""
    doc_chunks = collection.get(where={"source": doc_name})

    if not doc_chunks["documents"]:
        return "Document not found. Please ingest it first."

    full_context = "\n\n".join(doc_chunks["documents"])

    prompt = f"""Analyze the following company document and produce a structured report with:

1. Executive Summary (3-4 sentences)
2. Key Financial Risks (bullet points)
3. Growth Opportunities (bullet points)

Document:
{full_context[:8000]}"""

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1500,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_risk_scores(doc_name: str) -> dict:
    """
    Analyze a document and return structured risk scores (1-10, where 10 = highest risk)
    across four categories, plus an overall score and short justifications.
    """
    doc_chunks = collection.get(where={"source": doc_name})

    if not doc_chunks["documents"]:
        raise ValueError("Document not found. Please ingest it first.")

    full_context = "\n\n".join(doc_chunks["documents"])

    prompt = f"""You are a due diligence risk analyst. Analyze the document below and score its risk
level from 1 (very low risk) to 10 (very high risk) across these four categories:
Financial, Operational, Market, and Regulatory.

Respond with ONLY valid JSON, no other text, no markdown code fences, in exactly this format:
{{
  "overall_score": <integer 1-10>,
  "categories": {{
    "Financial": <integer 1-10>,
    "Operational": <integer 1-10>,
    "Market": <integer 1-10>,
    "Regulatory": <integer 1-10>
  }},
  "justifications": {{
    "Financial": "<one short sentence>",
    "Operational": "<one short sentence>",
    "Market": "<one short sentence>",
    "Regulatory": "<one short sentence>"
  }}
}}

Document:
{full_context[:8000]}"""

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=3000,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )

    if response.choices[0].finish_reason == "length":
        raise ValueError(
            "The model's response was cut off before completing (hit the token limit). "
            "Try again — if this keeps happening, the document may be unusually long."
        )

    return _parse_json_response(response.choices[0].message.content)


def _parse_json_response(raw_text: str | None) -> dict:
    """Defensively parse an LLM response as JSON, stripping markdown fences if present."""
    if not raw_text or not raw_text.strip():
        raise ValueError(
            "The model returned an empty response. This can happen with reasoning models "
            "when max_tokens is too low or the input is very long — try again, or shorten the input."
        )

    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    import json
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"Model did not return valid JSON. Raw response: {raw_text[:300]}")


def generate_ats_score(resume_text: str, job_description: str = "") -> dict:
    """
    Analyze a resume's ATS (Applicant Tracking System) compatibility.
    If a job description is provided, scores keyword match against it specifically.
    """
    jd_section = (
        f"\nTarget Job Description (score keyword match against this):\n{job_description[:4000]}"
        if job_description.strip()
        else "\nNo specific job description was provided — score general ATS best practices only."
    )

    prompt = f"""You are an ATS (Applicant Tracking System) resume screening expert. Analyze the resume
below and score its ATS compatibility from 0-100 across these categories:
Keyword Match, Formatting & Structure, Sections Completeness, Impact & Achievements.

Also identify specific missing keywords (skills/terms from the job description not found in the resume,
or general in-demand keywords if no job description is given) and give concrete, actionable suggestions.

IMPORTANT: Keep your response compact. List AT MOST 6 missing keywords and AT MOST 5 suggestions,
each suggestion under 15 words. Respond with ONLY valid JSON, no other text, no markdown code fences,
in exactly this format:
{{
  "overall_score": <integer 0-100>,
  "categories": {{
    "Keyword Match": <integer 0-100>,
    "Formatting & Structure": <integer 0-100>,
    "Sections Completeness": <integer 0-100>,
    "Impact & Achievements": <integer 0-100>
  }},
  "missing_keywords": ["<keyword1>", "<keyword2>"],
  "suggestions": ["<short suggestion 1>", "<short suggestion 2>"]
}}

Resume:
{resume_text[:6000]}
{jd_section}"""

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=3000,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )

    if response.choices[0].finish_reason == "length":
        raise ValueError(
            "The model's response was cut off before completing (hit the token limit). "
            "Try again — if this keeps happening, the resume text may be unusually long."
        )

    return _parse_json_response(response.choices[0].message.content)