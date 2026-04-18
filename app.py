import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
import streamlit as st
import fitz
import re
import time
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
from keybert import KeyBERT
from rouge_score import rouge_scorer
 
# ─────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Document Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
 
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
 
.app-header { text-align: center; padding: 28px 0 8px; }
.app-title { font-size: 34px; font-weight: 600; letter-spacing: -0.5px; color: #f0f6fc; }
.app-sub { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #8b949e; margin-top: 4px; }
 
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #161b22; border-radius: 10px;
    padding: 4px; border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; font-weight: 500; font-size: 13px;
    color: #8b949e; padding: 7px 20px;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #f0f6fc !important;
    border: 1px solid #30363d !important;
}
 
.kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }
.kpi-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px 18px; }
.kpi-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
.kpi-value { font-size: 28px; font-weight: 600; color: #f0f6fc; line-height: 1; }
.kpi-sub { font-size: 11px; margin-top: 4px; color: #3fb950; }
 
.sec-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8b949e;
    text-transform: uppercase; letter-spacing: 1px;
    padding-bottom: 8px; border-bottom: 1px solid #21262d; margin: 20px 0 14px;
}
 
.insight-box {
    background: #0d1117; border: 1px solid #30363d;
    border-left: 3px solid #58a6ff; border-radius: 0 10px 10px 0;
    padding: 16px 20px; font-size: 14px; color: #c9d1d9; line-height: 1.75; margin: 12px 0;
}
.summary-box {
    background: #0d1117; border: 1px solid #30363d;
    border-left: 3px solid #3fb950; border-radius: 0 10px 10px 0;
    padding: 16px 20px; font-size: 14px; color: #c9d1d9; line-height: 1.75; margin: 12px 0;
}
 
.stChatMessage { border-radius: 14px; margin-bottom: 10px; }
 
.thinking {
    display: flex; align-items: center; gap: 6px;
    padding: 10px 0; color: #8b949e; font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
}
.dot-flash {
    width: 7px; height: 7px; border-radius: 50%;
    background: #58a6ff; animation: blink 1.2s infinite;
}
.dot-flash:nth-child(2) { animation-delay: 0.2s; }
.dot-flash:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,80%,100%{opacity:0.2} 40%{opacity:1} }
 
.stButton > button {
    border-radius: 8px; font-weight: 500; font-size: 13px;
    border: 1px solid #30363d; background: #21262d; color: #f0f6fc;
    transition: border-color 0.15s;
}
.stButton > button:hover { border-color: #58a6ff; color: #58a6ff; }
 
.kw-chip {
    display: inline-block; background: #1f2d3d; border: 1px solid #1f6feb;
    border-radius: 20px; padding: 4px 12px; font-size: 12px; color: #58a6ff;
    margin: 3px; font-family: 'IBM Plex Mono', monospace;
}
 
.rouge-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0; }
.rouge-badge {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 10px 16px; text-align: center; min-width: 120px;
}
.rouge-badge .rb-label { font-size: 11px; color: #8b949e; font-family: 'IBM Plex Mono', monospace; }
.rouge-badge .rb-val { font-size: 22px; font-weight: 600; color: #f0f6fc; }
</style>
""", unsafe_allow_html=True)
 
st.markdown("""
<div class="app-header">
  <div class="app-title">AI Document Intelligence</div>
  <div class="app-sub">// pdf · excel · summarize · chat — 100% offline</div>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────
for key, val in {
    "context_text": "", "messages": [], "show_attach": False,
    "df": None, "df_name": "", "pdf_text": "", "summary_cache": {},
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val
 
# ─────────────────────────────────────────────────────────
#  Load models
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_summarizer():
    try:
        return pipeline("summarization", model="./models/t5-summarizer", tokenizer="./models/t5-summarizer")
    except Exception:
        return None
 
@st.cache_resource(show_spinner=False)
def load_kw_model():
    try:
        return KeyBERT(model="./models/keybert")
    except Exception:
        return None
 
with st.spinner("Loading AI models…"):
    summarizer = load_summarizer()
    kw_model = load_kw_model()
 
# ─────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    ollama_model = "llama3"
 
    
    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("Clear All", use_container_width=True):
    
        for k in ["context_text", "pdf_text"]:
            st.session_state[k] = ""
        st.session_state.df = None
        st.session_state.df_name = ""
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("## History")
    if st.button("Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    for item in st.session_state.chat_history[::-1]:
        if st.button(item["question"][:30]):
            st.session_state.messages = [
                {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": item["answer"]}
        ]
 
    if st.session_state.df is not None:
        st.info(f"📊 {st.session_state.df_name}\n{st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols")
    if st.session_state.pdf_text:
        st.info(f"📄 PDF loaded ({len(st.session_state.pdf_text):,} chars)")
 
# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
 
def split_chunks(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]
 
def run_summarizer(text, compression):
    if not summarizer:
        return "[Summarizer unavailable — check ./models/t5-summarizer]"
    cache_key = text[:150] + str(compression)
    if cache_key in st.session_state.summary_cache:
        return st.session_state.summary_cache[cache_key]
    chunks = split_chunks(text)
    partials = []
    for chunk in chunks:
        if len(chunk.split()) > 30:
            try:
                out = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
                partials.append(out[0]["summary_text"])
            except Exception:
                pass
    combined = " ".join(partials)
    orig_words = len(text.split())
    target = max(50, int(orig_words * (compression / 100)))
    try:
        final = summarizer(combined[:2000], max_length=min(target, 512),
                           min_length=max(30, int(target * 0.5)), do_sample=False)
        result = final[0]["summary_text"]
    except Exception:
        result = combined[:1000]
    st.session_state.summary_cache[cache_key] = result
    return result
 
def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return response.json().get("response", "No response returned.")
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama is not running. Please run ollama run llama3 in your terminal."
    except Exception as e:
        return f"⚠️ Error: {e}"
 
def thinking_html(label="thinking…"):
    return f"""<div class="thinking">
  <div class="dot-flash"></div><div class="dot-flash"></div><div class="dot-flash"></div>
  &nbsp; {label}
</div>"""
 
def numeric_cols(df): return df.select_dtypes(include="number").columns.tolist()
def cat_cols(df): return df.select_dtypes(include=["object", "category"]).columns.tolist()
 
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=36, b=0),
    font=dict(family="IBM Plex Mono, monospace", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f78166", "#bc8cff", "#ffa657", "#39d353"]
 
# ─────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────
tab_chat, tab_pdf, tab_excel, tab_dashboard = st.tabs([
    "💬  Chat", "📄  PDF Analysis", "📊  Excel Analysis", "🧩  Dashboard"
])
 
# ═══════════════════════════════════════════════════════
#  TAB 1 — CHAT
# ═══════════════════════════════════════════════════════
with tab_chat:
    c1, _ = st.columns([1, 14])
    with c1:
        if st.button("➕", help="Attach PDF or text"):
            st.session_state.show_attach = not st.session_state.show_attach
 
    if st.session_state.show_attach:
        with st.expander("📎 Attach context", expanded=True):
            up_pdf = st.file_uploader("Upload PDF", type="pdf", key="chat_pdf_up")
            if up_pdf:
                doc = fitz.open(stream=up_pdf.read(), filetype="pdf")
                text = clean_text(" ".join(p.get_text() for p in doc))
                st.session_state.context_text = text
                st.session_state.pdf_text = text
                st.success(f"PDF loaded — {len(text):,} chars")
            paste = st.text_area("Or paste text", height=100, key="chat_paste")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Load text"):
                    st.session_state.context_text = paste
            with col_b:
                if st.button("Clear context"):
                    st.session_state.context_text = ""
 
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
 
    if prompt := st.chat_input("Ask anything…"):
        ctx = st.session_state.context_text
        if not ctx and st.session_state.df is not None:
            df_ctx = st.session_state.df
            ctx = (f"Dataset: {st.session_state.df_name}\nShape: {df_ctx.shape}\n"
                   f"Columns: {', '.join(df_ctx.columns)}\n\nStats:\n{df_ctx.describe(include='number').to_string()}")
        full_prompt = (
            f"You are a professional AI assistant.\n\n"
            + (f"Use the context below if relevant.\n\nContext:\n{ctx[:8000]}\n\n" if ctx else "")
            + f"User question: {prompt}"
        )
 
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
 
        with st.chat_message("assistant"):
            thinking_ph = st.empty()
            thinking_ph.markdown(thinking_html("thinking…"), unsafe_allow_html=True)
 
            t0 = time.time()
            raw_answer = ask_ollama(full_prompt)
            elapsed = round(time.time() - t0, 1)
 
            thinking_ph.empty()
            answer_ph = st.empty()
            displayed = ""
            for word in raw_answer.split():
                displayed += word + " "
                answer_ph.markdown(displayed + "▌")
                time.sleep(0.018)
            answer_ph.markdown(displayed)
            st.caption(f"⏱ {elapsed}s · model: {ollama_model}")
 
        st.session_state.messages.append({"role": "assistant", "content": displayed})
        st.session_state.chat_history.append({
            "question": prompt,
            "answer": displayed
        })
 
# ═══════════════════════════════════════════════════════
#  TAB 2 — PDF ANALYSIS
# ═══════════════════════════════════════════════════════
with tab_pdf:
    st.markdown('<div class="sec-label">Upload PDF</div>', unsafe_allow_html=True)
    up_pdf2 = st.file_uploader("Drop a PDF", type="pdf", key="pdf_tab_up")
    if up_pdf2:
        with st.spinner("Extracting text…"):
            doc = fitz.open(stream=up_pdf2.read(), filetype="pdf")
            raw = " ".join(p.get_text() for p in doc)
            text = clean_text(raw)
            st.session_state.pdf_text = text
            st.session_state.context_text = text
 
    text = st.session_state.pdf_text
 
    if text:
        word_count = len(text.split())
        sent_count = len(re.split(r"[.!?]+", text))
        char_count = len(text)
        avg_word = round(char_count / max(word_count, 1), 1)
 
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi-card"><div class="kpi-label">words</div><div class="kpi-value">{word_count:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">sentences</div><div class="kpi-value">{sent_count:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">characters</div><div class="kpi-value">{char_count:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">avg word len</div><div class="kpi-value">{avg_word}</div></div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown('<div class="sec-label">Text preview</div>', unsafe_allow_html=True)
        st.text_area("", text[:2000], height=160, label_visibility="collapsed")
 
        st.markdown('<div class="sec-label">Word length distribution</div>', unsafe_allow_html=True)
        lengths = pd.Series([len(w) for w in text.split()]).value_counts().sort_index()
        fig = px.bar(x=lengths.index, y=lengths.values, labels={"x": "Word length", "y": "Count"},
                     color=lengths.values, color_continuous_scale="Blues")
        fig.update_layout(**PLOTLY_BASE, height=240, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
 
        st.markdown('<div class="sec-label">Summarize</div>', unsafe_allow_html=True)
        if st.button("Generate Summary", use_container_width=True, key="pdf_sum_btn"):
            with st.spinner("Running T5 summarizer…"):
                summary = run_summarizer(text, 25)
            sw = len(summary.split())
            reduction = round((1 - sw / max(word_count, 1)) * 100, 1)
 
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="kpi-row">
              <div class="kpi-card"><div class="kpi-label">original words</div><div class="kpi-value">{word_count:,}</div></div>
              <div class="kpi-card"><div class="kpi-label">summary words</div><div class="kpi-value">{sw}</div></div>
              <div class="kpi-card"><div class="kpi-label">reduction</div><div class="kpi-value">{reduction}%</div></div>
              <div class="kpi-card"><div class="kpi-label">compression</div><div class="kpi-value">25%</div></div>
            </div>
            """, unsafe_allow_html=True)
 
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(text[:2000], summary)
            st.markdown('<div class="sec-label">ROUGE scores</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="rouge-row">
              <div class="rouge-badge"><div class="rb-label">ROUGE-1</div><div class="rb-val">{round(scores['rouge1'].fmeasure,3)}</div></div>
              <div class="rouge-badge"><div class="rb-label">ROUGE-2</div><div class="rb-val">{round(scores['rouge2'].fmeasure,3)}</div></div>
              <div class="rouge-badge"><div class="rb-label">ROUGE-L</div><div class="rb-val">{round(scores['rougeL'].fmeasure,3)}</div></div>
            </div>
            """, unsafe_allow_html=True)
 
            if kw_model:
                st.markdown('<div class="sec-label">Key concepts</div>', unsafe_allow_html=True)
                kws = kw_model.extract_keywords(text[:3000], top_n=12)
                chips = "".join(f'<span class="kw-chip">{kw} <span style="opacity:0.5">{round(sc,2)}</span></span>' for kw, sc in kws)
                st.markdown(chips, unsafe_allow_html=True)
 
            st.download_button("⬇ Download summary", summary, "summary.txt", use_container_width=True)
    else:
        st.info("Upload a PDF above to begin analysis.")
 
# ═══════════════════════════════════════════════════════
#  TAB 3 — EXCEL ANALYSIS
# ═══════════════════════════════════════════════════════
with tab_excel:
    st.markdown('<div class="sec-label">Upload spreadsheet</div>', unsafe_allow_html=True)
    up_xl = st.file_uploader("Drop an Excel or CSV file", type=["xlsx", "xls", "csv"], key="xl_up")
    if up_xl:
        with st.spinner("Reading file…"):
            df = pd.read_csv(up_xl) if up_xl.name.endswith(".csv") else pd.read_excel(up_xl, engine="openpyxl")
            st.session_state.df = df
            st.session_state.df_name = up_xl.name
 
    df = st.session_state.df
 
    if df is not None:
        ncols = numeric_cols(df)
        ccols = cat_cols(df)
        missing = int(df.isnull().sum().sum())
        dup = int(df.duplicated().sum())
 
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi-card"><div class="kpi-label">rows</div><div class="kpi-value">{df.shape[0]:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">columns</div><div class="kpi-value">{df.shape[1]}</div></div>
          <div class="kpi-card"><div class="kpi-label">missing values</div><div class="kpi-value">{missing:,}</div><div class="kpi-sub">{'⚠ needs attention' if missing > 0 else '✓ clean'}</div></div>
          <div class="kpi-card"><div class="kpi-label">duplicates</div><div class="kpi-value">{dup:,}</div></div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown('<div class="sec-label">Data preview</div>', unsafe_allow_html=True)
        n_preview = st.slider("Rows", 5, min(200, len(df)), 10, key="xl_prev")
        st.dataframe(df.head(n_preview), use_container_width=True)
 
        st.markdown('<div class="sec-label">Descriptive statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").fillna("—"), use_container_width=True)
 
        # Missing value heatmap
        if missing > 0:
            st.markdown('<div class="sec-label">Missing value map</div>', unsafe_allow_html=True)
            fig = px.imshow(df.isnull().astype(int).T, color_continuous_scale=["#161b22", "#f78166"], aspect="auto")
            fig.update_layout(**PLOTLY_BASE, height=max(200, df.shape[1] * 26))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
 
        # Column inspector
        st.markdown('<div class="sec-label">Column inspector</div>', unsafe_allow_html=True)
        sel_col = st.selectbox("Select column", df.columns.tolist(), key="xl_col")
        col_series = df[sel_col]
        is_num = pd.api.types.is_numeric_dtype(col_series)
 
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f"*Type:* {col_series.dtype}")
            st.markdown(f"*Non-null:* {col_series.notna().sum():,}")
            st.markdown(f"*Null:* {col_series.isnull().sum():,}")
            st.markdown(f"*Unique:* {col_series.nunique():,}")
            if is_num:
                for label, val in [("Min", col_series.min()), ("Max", col_series.max()),
                                   ("Mean", col_series.mean()), ("Median", col_series.median()),
                                   ("Std", col_series.std()), ("Skew", col_series.skew()),
                                   ("Kurt", col_series.kurtosis())]:
                    st.markdown(f"*{label}:* {val:.4g}")
        with right:
            if is_num:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=col_series.dropna(), marker_color="#58a6ff", opacity=0.8, name="dist"))
                fig.add_trace(go.Box(x=col_series.dropna(), marker_color="#3fb950", name="box", boxmean=True))
                fig.update_layout(**PLOTLY_BASE, height=280, barmode="overlay")
                st.plotly_chart(fig, use_container_width=True)
            else:
                top = col_series.value_counts().head(15).reset_index()
                top.columns = [sel_col, "count"]
                fig = px.bar(top, x="count", y=sel_col, orientation="h",
                             color="count", color_continuous_scale="Blues")
                fig.update_layout(**PLOTLY_BASE, height=300, coloraxis_showscale=False,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
 
        # Correlation + scatter matrix
        if len(ncols) >= 2:
            st.markdown('<div class="sec-label">Correlation matrix</div>', unsafe_allow_html=True)
            corr = df[ncols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(**PLOTLY_BASE, height=420)
            st.plotly_chart(fig, use_container_width=True)
 
            st.markdown('<div class="sec-label">Scatter matrix</div>', unsafe_allow_html=True)
            scat_cols = ncols[:5]
            color_by = ccols[0] if ccols else None
            fig = px.scatter_matrix(df, dimensions=scat_cols, color=color_by, color_discrete_sequence=COLORS)
            fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.6))
            fig.update_layout(**PLOTLY_BASE, height=520)
            st.plotly_chart(fig, use_container_width=True)
 
        # Outlier detection
        if ncols:
            st.markdown('<div class="sec-label">Outlier detection (Z-score > 3)</div>', unsafe_allow_html=True)
            z_col = st.selectbox("Column for outlier scan", ncols, key="xl_z")
            z_scores = (df[z_col] - df[z_col].mean()) / df[z_col].std()
            outliers = df[abs(z_scores) > 3]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df[z_col], mode="markers",
                marker=dict(
                    color=["#f78166" if abs(z) > 3 else "#58a6ff" for z in z_scores],
                    size=[9 if abs(z) > 3 else 4 for z in z_scores],
                ),
            ))
            fig.update_layout(**PLOTLY_BASE, height=300, xaxis_title="Index", yaxis_title=z_col)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"🔴 {len(outliers)} outlier rows detected")
            if len(outliers) > 0:
                with st.expander("Show outlier rows"):
                    st.dataframe(outliers, use_container_width=True)
 
        # AI summary
        st.markdown('<div class="sec-label">AI insights</div>', unsafe_allow_html=True)
        if st.button("Generate AI Insights", use_container_width=True, key="xl_ai_sum"):
            num_df = df.select_dtypes(include="number")
            if not num_df.empty:
                snippet = num_df.describe().to_string()
            else:
                snippet = "No numeric columns available"
            q = (f"You are a senior data analyst. Analyze this dataset and give exactly 5 key insights, "
                 f"one per line, numbered:\n\nFile: {st.session_state.df_name}\n"
                 f"Shape: {df.shape}\nColumns: {', '.join(df.columns)}\n\nStats:\n{snippet}")
            thinking_ph = st.empty()
            thinking_ph.markdown(thinking_html("analyzing data…"), unsafe_allow_html=True)
            answer = ask_ollama(q)
            thinking_ph.empty()
            st.markdown(f'<div class="insight-box">{answer}</div>', unsafe_allow_html=True)
 
        st.download_button("⬇ Download cleaned CSV", df.to_csv(index=False).encode("utf-8"),
                           f"cleaned_{st.session_state.df_name}.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Upload an Excel (.xlsx / .xls) or CSV file above to begin.")
 
# ═══════════════════════════════════════════════════════
#  TAB 4 — DASHBOARD
# ═══════════════════════════════════════════════════════
with tab_dashboard:
    df = st.session_state.df
 
    if df is None:
        st.info("Upload a spreadsheet in the *Excel Analysis* tab first.")
    else:
        ncols = numeric_cols(df)
        ccols = cat_cols(df)
 
        st.markdown(f'<div class="sec-label">{st.session_state.df_name}</div>', unsafe_allow_html=True)
 
        # Auto KPI indicators
        if ncols:
            kpi_disp = ncols[:4]
            kpi_cols_ui = st.columns(len(kpi_disp))
            for i, nc in enumerate(kpi_disp):
                with kpi_cols_ui[i]:
                    fig = go.Figure(go.Indicator(
                        mode="number+delta",
                        value=float(df[nc].mean()),
                        delta={"reference": float(df[nc].median()), "relative": True, "valueformat": ".1%"},
                        title={"text": nc, "font": {"size": 12, "color": "#8b949e"}},
                        number={"font": {"size": 24, "color": "#f0f6fc"}, "valueformat": ".2f"},
                    ))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=130,
                                      margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig, use_container_width=True)
 
        # Chart builder
        st.markdown('<div class="sec-label">Chart builder</div>', unsafe_allow_html=True)
        cb1, cb2, cb3, cb4 = st.columns(4)
        chart_type = cb1.selectbox("Type", ["Bar", "Line", "Scatter", "Area", "Box", "Violin", "Pie", "Funnel"], key="db_type")
        x_ax = cb2.selectbox("X axis", df.columns.tolist(), key="db_x")
        y_ax = cb3.selectbox("Y axis", ncols if ncols else df.columns.tolist(), key="db_y")
        color_by = cb4.selectbox("Color by", ["None"] + ccols + ncols, key="db_color")
        color_arg = None if color_by == "None" else color_by
 
        try:
            pkw = dict(data_frame=df, x=x_ax, y=y_ax, color=color_arg,
                       color_discrete_sequence=COLORS, template="plotly_dark")
            chart_fns = {
                "Bar":     lambda: px.bar(**pkw),
                "Line":    lambda: px.line(**pkw),
                "Scatter": lambda: px.scatter(**pkw, opacity=0.7),
                "Area":    lambda: px.area(**pkw),
                "Box":     lambda: px.box(**pkw),
                "Violin":  lambda: px.violin(**pkw, box=True),
                "Pie":     lambda: px.pie(df, names=x_ax, values=y_ax,
                                          color_discrete_sequence=COLORS, template="plotly_dark"),
                "Funnel":  lambda: px.funnel(**pkw),
            }
            fig = chart_fns[chart_type]()
            fig.update_layout(**PLOTLY_BASE, height=440)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")
 
        # Time series
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        obj_date = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "year", "month"])]
        all_date = list(set(date_cols + obj_date))
        if all_date and ncols:
            st.markdown('<div class="sec-label">Time series</div>', unsafe_allow_html=True)
            ts1, ts2 = st.columns(2)
            ts_x = ts1.selectbox("Date column", all_date, key="ts_x")
            ts_y = ts2.multiselect("Value columns", ncols, default=ncols[:1], key="ts_y")
            if ts_y:
                try:
                    ts_df = df[[ts_x] + ts_y].copy()
                    ts_df[ts_x] = pd.to_datetime(ts_df[ts_x], errors="coerce")
                    ts_df = ts_df.dropna(subset=[ts_x]).sort_values(ts_x)
                    fig = px.line(ts_df, x=ts_x, y=ts_y, color_discrete_sequence=COLORS, template="plotly_dark")
                    fig.update_layout(**PLOTLY_BASE, height=360)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Time series error: {e}")
 
        # Distribution overview grid
        if len(ncols) >= 2:
            st.markdown('<div class="sec-label">Distribution overview</div>', unsafe_allow_html=True)
            disp_cols = ncols[:6]
            n_rows = (len(disp_cols) + 2) // 3
            fig = make_subplots(rows=n_rows, cols=3, subplot_titles=disp_cols)
            for idx, nc in enumerate(disp_cols):
                r, c = divmod(idx, 3)
                fig.add_trace(go.Histogram(x=df[nc].dropna(), name=nc,
                                           marker_color=COLORS[idx % len(COLORS)], showlegend=False),
                              row=r+1, col=c+1)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=260*n_rows,
                              margin=dict(l=0, r=0, t=40, b=0),
                              font=dict(family="IBM Plex Mono, monospace", size=11))
            st.plotly_chart(fig, use_container_width=True)
 
        # Category breakdown
        if ccols and ncols:
            st.markdown('<div class="sec-label">Category breakdown</div>', unsafe_allow_html=True)
            gb1, gb2, gb3 = st.columns(3)
            grp_cat = gb1.selectbox("Group by", ccols, key="grp_cat")
            grp_val = gb2.selectbox("Measure", ncols, key="grp_val")
            agg_fn  = gb3.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"], key="grp_agg")
            grouped = df.groupby(grp_cat)[grp_val].agg(agg_fn).reset_index().sort_values(grp_val, ascending=False)
            fig = px.bar(grouped, x=grp_cat, y=grp_val, color=grp_val,
                         color_continuous_scale="Blues", template="plotly_dark")
            fig.update_layout(**PLOTLY_BASE, height=360, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
 
        # Ask AI about dashboard
        st.markdown('<div class="sec-label">Ask AI about this data</div>', unsafe_allow_html=True)
        ai_q = st.text_input("Your question", placeholder="e.g. Which category has the highest average sales?", key="dash_q")
        if st.button("Ask AI", key="dash_ask"):
            if ai_q.strip():
                num_df = df.select_dtypes(include="number")
                if not num_df.empty:
                    stats_text = num_df.describe().to_string()
                else:
                    stats_text = "No numeric columns available"
                ctx_data = (
                    f"Dataset: {st.session_state.df_name}\n"
                    f"Columns: {', '.join(df.columns)}\n"
                    f"Shape: {df.shape}\n\nStats:\n{stats_text}\n\n"
                    f"Sample:\n{df.head().to_string()}"
                    )
                thinking_ph = st.empty()
                thinking_ph.markdown(thinking_html("analyzing data…"), unsafe_allow_html=True)
                answer = ask_ollama(f"Data context:\n{ctx_data}\n\nQuestion: {ai_q}")
                thinking_ph.empty()
                st.markdown(f'<div class="insight-box">{answer}</div>', unsafe_allow_html=True)
 
        st.download_button("⬇ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           f"export_{st.session_state.df_name}.csv", mime="text/csv", use_container_width=True)
