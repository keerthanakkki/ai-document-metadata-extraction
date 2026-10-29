"""
frontend/streamlit_app.py
──────────────────────────
Streamlit UI for the AI Document Metadata Extraction system.

Run from the project root:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import requests
import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="AI Document Metadata Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API base URL (defaults to local FastAPI) ──────────────────
_DEFAULT_API_BASE = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* General body */
    body { font-family: 'Segoe UI', sans-serif; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Success / Warning / Error badges */
    .badge-success {
        background: #1a4731; color: #4ade80;
        padding: 2px 10px; border-radius: 12px; font-size: 0.78em;
    }
    .badge-warning {
        background: #3b2f00; color: #fbbf24;
        padding: 2px 10px; border-radius: 12px; font-size: 0.78em;
    }
    .badge-error {
        background: #3b0f0f; color: #f87171;
        padding: 2px 10px; border-radius: 12px; font-size: 0.78em;
    }
    .badge-missing {
        background: #1e2130; color: #94a3b8;
        padding: 2px 10px; border-radius: 12px; font-size: 0.78em;
    }

    /* Stage progress items */
    .stage-active { color: #60a5fa; font-weight: 600; }
    .stage-done   { color: #4ade80; }
    .stage-wait   { color: #64748b; }

    /* Section headers */
    h2 { border-bottom: 1px solid #2d3250; padding-bottom: 6px; }

    /* Table styling */
    table { width: 100%; }
    th { background: #1e2130 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────


def _api_base() -> str:
    return st.session_state.get("api_base", _DEFAULT_API_BASE).rstrip("/")


def check_api_health() -> dict | None:
    try:
        resp = requests.get(f"{_api_base()}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def confidence_badge(level: str) -> str:
    mapping = {
        "high":    ("badge-success", "✔ High"),
        "medium":  ("badge-warning", "~ Medium"),
        "low":     ("badge-warning", "⚠ Low"),
        "unknown": ("badge-missing", "? Unknown"),
        "missing": ("badge-error",   "✗ Missing"),
    }
    cls, label = mapping.get(level.lower(), ("badge-missing", level))
    return f'<span class="{cls}">{label}</span>'


def result_to_df(extraction: dict) -> pd.DataFrame:
    """Convert the six main fields into a display DataFrame."""
    field_labels = {
        "agreement_value":      "Agreement Value",
        "agreement_start_date": "Agreement Start Date",
        "agreement_end_date":   "Agreement End Date",
        "renewal_notice_days":  "Renewal Notice (Days)",
        "party_one":            "Party One (Landlord)",
        "party_two":            "Party Two (Tenant)",
    }
    rows = []
    for key, label in field_labels.items():
        value = extraction.get(key)
        rows.append({"Field": label, "Extracted Value": value if value is not None else "—"})
    return pd.DataFrame(rows)


def to_csv_bytes(extraction: dict, filename: str) -> bytes:
    rows = [{"filename": filename}]
    for k in [
        "agreement_value", "agreement_start_date", "agreement_end_date",
        "renewal_notice_days", "party_one", "party_two",
    ]:
        rows[0][k] = extraction.get(k)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def _fmt_seconds(secs: float | None) -> str:
    if secs is None:
        return ""
    if secs < 1:
        return f"{secs * 1000:.0f} ms"
    return f"{secs:.1f} s"


# ─────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────

if "api_base" not in st.session_state:
    st.session_state["api_base"] = _DEFAULT_API_BASE
if "processing" not in st.session_state:
    st.session_state["processing"] = False
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_url_input = st.text_input(
        "FastAPI Backend URL",
        value=st.session_state["api_base"],
        key="api_url_input",
    )
    if api_url_input != st.session_state["api_base"]:
        st.session_state["api_base"] = api_url_input
        st.rerun()

    st.markdown("---")
    st.markdown("### 🩺 API Status")

    health = check_api_health()
    if health is None:
        st.error("❌ Cannot reach API at " + _api_base())
        st.caption("Start FastAPI: `uvicorn app.main:app --reload`")
    else:
        status_icon = "✅" if health.get("status") == "ok" else "⚠️"
        st.success(f"{status_icon} API is running")
        model_ok = health.get("model_available", False)
        model_name = health.get("model", "unknown")
        if model_ok:
            st.info(f"🤖 Model **{model_name}** ready")
        else:
            st.warning(f"⚠️ Model **{model_name}** not available")
            model_err = health.get("model_error", "")
            if model_err:
                st.caption(model_err)

    st.markdown("---")
    st.markdown(
        """
        **Supported formats:**
        - DOCX (Word documents)
        - PNG, JPG, JPEG (scanned images)

        **Fields extracted:**
        1. Agreement Value
        2. Start Date
        3. End Date
        4. Renewal Notice Days
        5. Party One (Landlord)
        6. Party Two (Tenant)
        """
    )


# ─────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────

st.title("📄 AI-Powered Document Metadata Extraction")
st.markdown(
    """
    Upload a **rental / lease agreement** document and this system will automatically
    extract key metadata using OCR and a large language model (LLM).

    > The extraction is **AI-based** — the model reads and understands the document
    > rather than using hardcoded rules or regular expressions.
    """
)

st.divider()

# ── File uploader ─────────────────────────────────────────────
st.subheader("1️⃣  Upload Document")
uploaded_file = st.file_uploader(
    "Choose a rental agreement file",
    type=["docx", "png", "jpg", "jpeg"],
    help="DOCX (Word), PNG, JPG or JPEG. Maximum 20 MB.",
    label_visibility="collapsed",
    disabled=st.session_state["processing"],
)

if uploaded_file is not None:
    file_info_col, _ = st.columns([2, 3])
    with file_info_col:
        size_kb = round(uploaded_file.size / 1024, 1)
        st.markdown(
            f"📎 **{uploaded_file.name}** &nbsp; `{size_kb} KB` &nbsp; "
            f"`{uploaded_file.type or 'unknown'}`",
            unsafe_allow_html=True,
        )

st.divider()

# ── Extract button ────────────────────────────────────────────
st.subheader("2️⃣  Extract Metadata")

extract_btn = st.button(
    "🔍 Extract Metadata",
    type="primary",
    disabled=(
        uploaded_file is None
        or health is None
        or st.session_state["processing"]
    ),
    use_container_width=True,
    key="extract_btn",
)

if uploaded_file is None:
    st.info("⬆️ Please upload a document above to enable extraction.")
elif health is None:
    st.error("❌ Cannot run extraction — API is offline.")

# ── Processing pipeline with staged progress ──────────────────
if extract_btn and uploaded_file is not None and health is not None:
    st.session_state["processing"] = True
    st.session_state["last_result"] = None
    st.session_state["last_error"] = None

    is_image = uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg"))
    is_docx  = uploaded_file.name.lower().endswith(".docx")

    # Build stage list based on file type
    stages = [
        ("validate",  "🔍 Validating document format"),
        ("extract",   "📄 Extracting text via python-docx" if is_docx else "📷 Running OCR on scanned image"),
        ("llm",       "🤖 AI semantic extraction (Ollama)"),
        ("parse",     "✅ Validating structured output"),
    ]

    stage_area = st.empty()

    def _render_stages(current_idx: int, done: bool = False, error_idx: int = -1):
        lines = []
        for i, (_, label) in enumerate(stages):
            if error_idx >= 0 and i == error_idx:
                lines.append(f'<span class="badge-error">✗</span> ~~{label}~~')
            elif i < current_idx or done:
                lines.append(f'<span class="stage-done">✔</span> {label}')
            elif i == current_idx:
                lines.append(f'<span class="stage-active">⏳ {label}…</span>')
            else:
                lines.append(f'<span class="stage-wait">○ {label}</span>')
        stage_area.markdown("<br>".join(lines), unsafe_allow_html=True)

    _render_stages(0)

    wall_start = time.monotonic()
    resp = None
    conn_error = None

    try:
        _render_stages(0)
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }
        _render_stages(1)
        resp = requests.post(f"{_api_base()}/extract", files=files, timeout=300)
        _render_stages(2)

    except requests.exceptions.ConnectionError as conn_err:
        err_str = str(conn_err).lower()
        if any(k in err_str for k in ("wsarecv", "aborted", "reset", "eof")):
            conn_error = (
                "❌ **Connection to Ollama was aborted mid-generation.**\n\n"
                "This usually means Ollama ran out of memory or was killed by Windows.\n\n"
                "**Fix:** Restart Ollama in your terminal:\n```\nollama serve\n```\nThen click **Extract Metadata** again."
            )
        else:
            conn_error = (
                "❌ Could not connect to the API.\n\n"
                "Make sure FastAPI is running: `uvicorn app.main:app --reload`"
            )
    except requests.exceptions.Timeout:
        conn_error = (
            "⏰ The request timed out after 5 minutes.\n\n"
            "The document may be large or the model is very slow. "
            "Try again or use a smaller file."
        )

    wall_elapsed = time.monotonic() - wall_start
    st.session_state["processing"] = False

    if conn_error:
        _render_stages(1, error_idx=1)
        st.error(conn_error)
        st.button("🔄 Retry", key="retry_conn", use_container_width=True)
        st.stop()

    # ── Handle HTTP errors ─────────────────────────────────────
    if resp.status_code == 503:
        detail = resp.json().get("detail", "Model unavailable.")
        _render_stages(2, error_idx=2)
        st.error(f"🤖 **Model Unavailable:** {detail}")
        st.button("🔄 Retry", key="retry_503", use_container_width=True)
        st.stop()

    if resp.status_code == 400:
        detail = resp.json().get("detail", "Invalid file.")
        _render_stages(0, error_idx=0)
        st.error(f"📂 **File Error:** {detail}")
        st.stop()

    if resp.status_code != 200:
        _render_stages(3, error_idx=3)
        st.error(f"❌ API error {resp.status_code}: {resp.text[:300]}")
        st.button("🔄 Retry", key="retry_500", use_container_width=True)
        st.stop()

    _render_stages(3, done=True)

    payload = resp.json()
    extraction = payload.get("extraction", {})
    ex_status = extraction.get("status", "unknown")
    st.session_state["last_result"] = payload

    # ── Status banner ──────────────────────────────────────────
    st.divider()
    st.subheader("3️⃣  Extraction Results")

    # Timing summary from backend
    timings = extraction.get("timings", {})
    method = extraction.get("extraction_method") or "unknown"
    backend_total = timings.get("total_sec")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric("🕐 Total (wall clock)", _fmt_seconds(wall_elapsed))
    with col_t2:
        st.metric("📄 Text Extraction", _fmt_seconds(timings.get("text_extraction_sec")))
    with col_t3:
        st.metric("🤖 LLM Inference", _fmt_seconds(timings.get("llm_inference_sec")))
    with col_t4:
        st.metric("🔧 Method", method.upper())

    if ex_status == "success":
        st.success("✅ Extraction completed successfully.")
    elif ex_status == "partial":
        st.warning("⚠️ Extraction partially successful — some fields could not be determined.")
    elif ex_status == "model_unavailable":
        st.error(f"🤖 Model unavailable: {extraction.get('error_message', '')}")
        st.button("🔄 Retry", key="retry_model", use_container_width=True)
        st.stop()
    elif ex_status == "failed":
        err = extraction.get("error_message", "Unknown error")
        st.error(f"❌ Extraction failed: {err}")
        st.button("🔄 Retry", key="retry_fail", use_container_width=True)
        st.stop()

    # ── Results table ──────────────────────────────────────────
    df = result_to_df(extraction)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Field-level evidence & confidence ─────────────────────
    field_details = extraction.get("field_details", {})
    if field_details:
        with st.expander("🔍 Field-Level Evidence & Confidence", expanded=False):
            for field, detail in field_details.items():
                label = field.replace("_", " ").title()
                conf_level = detail.get("confidence_level", "unknown")
                is_missing = detail.get("is_missing", False)
                badge_html = confidence_badge("missing" if is_missing else conf_level)
                evidence = detail.get("evidence") or "—"
                conf_score = detail.get("confidence_score")
                score_str = f"{conf_score:.0%}" if conf_score is not None else "n/a"

                st.markdown(
                    f"**{label}** &nbsp; {badge_html} &nbsp; `{score_str}`",
                    unsafe_allow_html=True,
                )
                st.caption(f"Evidence: {evidence}")
                st.markdown("---")

    # ── Timing breakdown detail ────────────────────────────────
    if timings:
        with st.expander("⏱️ Stage Timing Breakdown", expanded=False):
            timing_rows = []
            stage_labels = {
                "validation_sec":      "File Validation",
                "text_extraction_sec": "Text Extraction",
                "few_shot_prep_sec":   "Few-Shot Prep",
                "llm_inference_sec":   "LLM Inference",
                "result_assembly_sec": "Result Assembly",
                "total_sec":           "Total (backend)",
            }
            for k, label in stage_labels.items():
                v = timings.get(k)
                if v is not None:
                    timing_rows.append({"Stage": label, "Duration": _fmt_seconds(v)})
            st.dataframe(pd.DataFrame(timing_rows), hide_index=True, use_container_width=True)

    # ── Warnings for low-confidence / missing fields ───────────
    missing_fields = [
        k for k, v in field_details.items() if v.get("is_missing", False)
    ]
    low_conf_fields = [
        k for k, v in field_details.items()
        if not v.get("is_missing") and v.get("confidence_level") in ("low", "unknown")
    ]

    if missing_fields:
        st.warning(
            f"⚠️ The following fields could **not** be extracted: "
            f"{', '.join(f.replace('_', ' ').title() for f in missing_fields)}"
        )
    if low_conf_fields:
        st.info(
            f"ℹ️ Low confidence on: "
            f"{', '.join(f.replace('_', ' ').title() for f in low_conf_fields)}. "
            "Manual review recommended."
        )

    # ── Extracted text preview ─────────────────────────────────
    text_preview = extraction.get("document_text_preview")
    if text_preview:
        with st.expander("📝 Document Text Preview (first 500 chars)", expanded=False):
            st.code(text_preview, language=None)

    # ── Downloads ──────────────────────────────────────────────
    st.divider()
    st.subheader("4️⃣  Download Results")

    col1, col2 = st.columns(2)

    json_payload = {
        "filename": payload.get("filename"),
        "extraction": {
            k: extraction.get(k)
            for k in [
                "agreement_value", "agreement_start_date", "agreement_end_date",
                "renewal_notice_days", "party_one", "party_two",
                "status", "error_message", "extraction_method", "timings",
            ]
        },
    }

    with col1:
        st.download_button(
            label="⬇️ Download as JSON",
            data=json.dumps(json_payload, indent=2, default=str),
            file_name=f"{Path(uploaded_file.name).stem}_metadata.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="⬇️ Download as CSV",
            data=to_csv_bytes(extraction, uploaded_file.name),
            file_name=f"{Path(uploaded_file.name).stem}_metadata.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "AI Document Metadata Extraction System · "
    "Built with FastAPI + EasyOCR + Ollama · "
    "Extraction is AI-based; no hardcoded rules are used."
)
