"""
app.py
Streamlit frontend for the AI Due Diligence Copilot.
Talks to the FastAPI backend over HTTP.
Run with: streamlit run app.py
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import os

# Uses 127.0.0.1 locally by default. When deployed, set BACKEND_URL in Streamlit's
# "Secrets" (Settings > Secrets) to your live Render backend URL, e.g.:
# BACKEND_URL = "https://your-backend.onrender.com"
try:
    BACKEND_URL = st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "https://api.render.com/deploy/srv-da6lutpsrm7s73aj3c8g?key=LWHqFifKr4I"))
except Exception:
    # No secrets.toml file exists at all (normal for local development) — fall back to env var or default
    BACKEND_URL = os.getenv("BACKEND_URL", "https://api.render.com/deploy/srv-da6lutpsrm7s73aj3c8g?key=LWHqFifKr4I")

st.set_page_config(page_title="AI Due Diligence Copilot", page_icon="📊", layout="wide")

st.title("📊 AI Due Diligence Copilot")
st.caption("RAG-powered analysis of company filings, financial statements, and investor documents.")

# ---------- Sidebar: Upload + Document Management ----------
with st.sidebar:
    st.header("📁 Documents")

    uploaded_file = st.file_uploader("Upload a company filing (PDF)", type="pdf")
    if uploaded_file is not None:
        if st.button("Ingest Document", type="primary"):
            with st.spinner("Extracting text, chunking, and embedding..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                upload_url = f"{BACKEND_URL}/upload"
                try:
                    response = requests.post(upload_url, files=files, timeout=120)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"{data['message']} ({data['chunks_created']} chunks)")
                    else:
                        st.error(
                            f"Upload failed — Status {response.status_code} from {upload_url}\n\n"
                            f"Response: {response.text}"
                        )
                except requests.exceptions.ConnectionError as e:
                    st.error(f"Cannot reach backend at {upload_url}.\n\nDetails: {e}")

    st.divider()
    st.subheader("Ingested Documents")
    try:
        docs_response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        documents = docs_response.json().get("documents", [])
    except requests.exceptions.ConnectionError:
        documents = []
        st.warning("Backend not running.")

    if documents:
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {doc}")
            if col2.button("🗑️", key=f"del_{doc}"):
                requests.delete(f"{BACKEND_URL}/documents/{doc}")
                st.rerun()
    else:
        st.info("No documents uploaded yet.")

# ---------- Main Area: Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Ask a Question", "📋 Executive Summary", "📈 Risk Dashboard", "🎯 ATS Score Checker"]
)

with tab1:
    st.subheader("Ask about your documents")

    doc_filter = st.selectbox(
        "Limit search to a specific document (optional)",
        options=["All documents"] + documents,
    )
    doc_filter = None if doc_filter == "All documents" else doc_filter

    query = st.text_area(
        "Your question",
        placeholder="e.g., What are the key financial risks mentioned in this filing?",
        height=100,
    )

    if st.button("Get Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        elif not documents:
            st.warning("Upload a document first.")
        else:
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    payload = {"question": query, "doc_filter": doc_filter, "n_results": 5}
                    response = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=60)
                    if response.status_code == 200:
                        result = response.json()
                        st.markdown("### Answer")
                        st.markdown(result["answer"])

                        with st.expander("📚 Sources used"):
                            for i, source in enumerate(result["sources"], 1):
                                st.write(f"{i}. **{source['source']}** — chunk {source['chunk_id']}")
                    else:
                        st.error(response.json().get("detail", "Query failed."))
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend. Make sure FastAPI is running on port 8000.")

with tab2:
    st.subheader("Generate Executive Summary & Risk Assessment")

    if not documents:
        st.info("Upload a document first.")
    else:
        selected_doc = st.selectbox("Select a document", options=documents, key="summary_doc")

        col1, col2 = st.columns([1, 1])
        with col1:
            generate_clicked = st.button("Generate Summary", type="primary")
        with col2:
            if st.button("📄 Export as PDF Report"):
                with st.spinner("Building PDF report (summary + risk scores)..."):
                    try:
                        response = requests.get(
                            f"{BACKEND_URL}/export-pdf/{selected_doc}", timeout=90
                        )
                        if response.status_code == 200:
                            st.download_button(
                                label="⬇️ Download Report",
                                data=response.content,
                                file_name=f"{selected_doc.rsplit('.', 1)[0]}_due_diligence_report.pdf",
                                mime="application/pdf",
                            )
                        else:
                            st.error(f"Failed to generate PDF: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach backend. Make sure FastAPI is running on port 8000.")

        if generate_clicked:
            with st.spinner("Analyzing document..."):
                try:
                    response = requests.get(f"{BACKEND_URL}/summary/{selected_doc}", timeout=60)
                    if response.status_code == 200:
                        st.markdown(response.json()["summary"])
                    else:
                        st.error("Failed to generate summary.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend. Make sure FastAPI is running on port 8000.")

with tab3:
    st.subheader("Risk Score Dashboard")

    if not documents:
        st.info("Upload a document first.")
    else:
        selected_doc_risk = st.selectbox("Select a document", options=documents, key="risk_doc")

        if st.button("Analyze Risk", type="primary"):
            with st.spinner("Scoring risk across categories..."):
                try:
                    response = requests.get(f"{BACKEND_URL}/risk-score/{selected_doc_risk}", timeout=60)
                    if response.status_code == 200:
                        risk_data = response.json()
                        st.session_state["risk_data"] = risk_data
                    else:
                        st.error(f"Failed to generate risk score: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend. Make sure FastAPI is running on port 8000.")

        if "risk_data" in st.session_state:
            risk_data = st.session_state["risk_data"]
            overall = risk_data["overall_score"]
            categories = risk_data["categories"]
            justifications = risk_data.get("justifications", {})

            # Overall score gauge
            risk_color = "#22c55e" if overall <= 3 else "#f59e0b" if overall <= 6 else "#ef4444"
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall,
                title={"text": "Overall Risk Score"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 3], "color": "#dcfce7"},
                        {"range": [3, 6], "color": "#fef9c3"},
                        {"range": [6, 10], "color": "#fee2e2"},
                    ],
                },
            ))
            gauge_fig.update_layout(height=300, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(gauge_fig, width='stretch')

            # Category breakdown bar chart
            bar_fig = go.Figure(go.Bar(
                x=list(categories.values()),
                y=list(categories.keys()),
                orientation="h",
                marker_color=["#ef4444" if v >= 7 else "#f59e0b" if v >= 4 else "#22c55e" for v in categories.values()],
                text=list(categories.values()),
                textposition="outside",
            ))
            bar_fig.update_layout(
                title="Risk by Category",
                xaxis=dict(range=[0, 10], title="Risk Score"),
                height=300,
                margin=dict(t=40, b=30, l=100, r=30),
            )
            st.plotly_chart(bar_fig, width='stretch')

            # Justifications
            st.subheader("Category Details")
            for category, score in categories.items():
                with st.expander(f"{category} — Score: {score}/10"):
                    st.write(justifications.get(category, "No detail provided."))

with tab4:
    st.subheader("Resume ATS Score Checker")
    st.caption("Upload a resume PDF to check its ATS (Applicant Tracking System) compatibility. Optionally paste a job description to score keyword match against that specific role.")

    resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf", key="resume_upload")
    job_description = st.text_area(
        "Paste job description (optional — improves keyword match accuracy)",
        placeholder="Paste the job posting text here...",
        height=150,
    )

    if st.button("Check ATS Score", type="primary"):
        if resume_file is None:
            st.warning("Please upload a resume PDF first.")
        else:
            with st.spinner("Analyzing resume for ATS compatibility..."):
                try:
                    files = {"resume": (resume_file.name, resume_file.getvalue(), "application/pdf")}
                    data = {"job_description": job_description}
                    response = requests.post(
                        f"{BACKEND_URL}/ats-score", files=files, data=data, timeout=60
                    )
                    if response.status_code == 200:
                        st.session_state["ats_data"] = response.json()
                    else:
                        st.error(f"Failed to score resume: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend. Make sure FastAPI is running on port 8000.")

    if "ats_data" in st.session_state:
        ats_data = st.session_state["ats_data"]
        overall = ats_data["overall_score"]
        categories = ats_data["categories"]
        missing_keywords = ats_data.get("missing_keywords", [])
        suggestions = ats_data.get("suggestions", [])

        # Overall score gauge (0-100 scale, higher = better here, unlike risk score)
        score_color = "#22c55e" if overall >= 75 else "#f59e0b" if overall >= 50 else "#ef4444"
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall,
            title={"text": "ATS Compatibility Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": score_color},
                "steps": [
                    {"range": [0, 50], "color": "#fee2e2"},
                    {"range": [50, 75], "color": "#fef9c3"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
            },
        ))
        gauge_fig.update_layout(height=300, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(gauge_fig, width='stretch')

        # Category breakdown bar chart (higher = better here)
        bar_fig = go.Figure(go.Bar(
            x=list(categories.values()),
            y=list(categories.keys()),
            orientation="h",
            marker_color=["#22c55e" if v >= 75 else "#f59e0b" if v >= 50 else "#ef4444" for v in categories.values()],
            text=list(categories.values()),
            textposition="outside",
        ))
        bar_fig.update_layout(
            title="Score by Category",
            xaxis=dict(range=[0, 100], title="Score"),
            height=300,
            margin=dict(t=40, b=30, l=140, r=30),
        )
        st.plotly_chart(bar_fig, width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔑 Missing Keywords")
            if missing_keywords:
                for kw in missing_keywords:
                    st.markdown(f"- `{kw}`")
            else:
                st.write("No major keyword gaps found.")

        with col2:
            st.subheader("💡 Suggestions to Improve")
            if suggestions:
                for s in suggestions:
                    st.markdown(f"- {s}")
            else:
                st.write("No specific suggestions.")
