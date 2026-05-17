from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.charts import (  # noqa: E402
    counterfactual_impact_chart,
    embedding_scatter_chart,
    financial_signal_chart,
    model_confidence_chart,
    model_latency_chart,
    prediction_bar_chart,
    similarity_heatmap,
    token_attribution_chart,
    tone_drift_chart,
    transcript_chunk_chart,
)
from attention.engine import AttentionEngine  # noqa: E402
from attention.heatmaps import attention_heatmap  # noqa: E402
from ber_tokenscope.schemas import EmbeddingDocument  # noqa: E402
from embeddings.engine import EmbeddingEngine  # noqa: E402
from explainability.engine import ExplainabilityEngine  # noqa: E402
from financial_nlp.sentiment import FinancialTextAnalyzer  # noqa: E402
from financial_nlp.transcript_analysis import (  # noqa: E402
    PeriodDocument,
    TranscriptAnalyzer,
)
from model_comparison.engine import ModelComparisonEngine  # noqa: E402
from observability.tracker import RunTracker  # noqa: E402


def configure_page() -> None:
    st.set_page_config(
        page_title="BERTokenScope",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --bs-bg: #ffffff;
            --bs-surface: #f8fbff;
            --bs-panel: #f1f7ff;
            --bs-panel-strong: #e2efff;
            --bs-ink: #102033;
            --bs-muted: #52657a;
            --bs-soft: #d7e8ff;
            --bs-border: rgba(37, 99, 235, 0.18);
            --bs-border-strong: rgba(37, 99, 235, 0.32);
            --bs-blue: #2563eb;
            --bs-blue-strong: #1d4ed8;
            --bs-blue-soft: rgba(37, 99, 235, 0.10);
            --bs-light-blue: #38bdf8;
            --bs-red: #dc2626;
            --bs-red-soft: rgba(220, 38, 38, 0.10);
        }
        html {
            scroll-behavior: smooth;
        }
        .stApp {
            background:
                linear-gradient(180deg, #ffffff 0%, #f8fbff 46%, #ffffff 100%);
            color: var(--bs-ink);
        }
        .block-container {
            padding-top: 3.1rem;
            padding-bottom: 4.5rem;
            max-width: 96rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--bs-ink);
        }
        p, li, span, label {
            color: var(--bs-muted);
        }
        [data-testid="stSidebar"] {
            background: #f4f9ff;
            border-right: 1px solid var(--bs-border);
        }
        [data-testid="stSidebar"] * {
            color: var(--bs-muted);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--bs-ink);
        }
        [data-testid="stMetric"] {
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(56, 189, 248, 0.12)),
                #ffffff;
            border: 1px solid var(--bs-border);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 28px rgba(37, 99, 235, 0.07);
        }
        [data-testid="stMetricLabel"] {
            color: var(--bs-muted);
        }
        [data-testid="stMetricValue"] {
            color: var(--bs-ink);
        }
        [data-testid="stMetricDelta"] {
            background: var(--bs-red-soft);
            border-radius: 999px;
            padding: 0.08rem 0.5rem;
            width: fit-content;
        }
        [data-testid="stMetricDelta"] * {
            color: var(--bs-red) !important;
        }
        .bs-title-wrap {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .bs-title-icon {
            width: 3.55rem;
            height: 3.55rem;
            flex: 0 0 auto;
            transform: translateY(-0.12rem);
        }
        .bs-title-icon svg {
            display: block;
            width: 100%;
            height: 100%;
        }
        .bs-hero {
            padding: 0 0 1.35rem;
            margin-bottom: 0.8rem;
            border: 0;
            background: transparent;
            box-shadow: none;
        }
        .bs-hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.12fr) minmax(18rem, 0.68fr);
            gap: 2rem;
            align-items: center;
        }
        .bs-title {
            font-size: 4rem;
            font-weight: 520;
            color: var(--bs-ink);
            margin: 0;
            line-height: 0.98;
        }
        .bs-subtitle {
            color: var(--bs-muted);
            font-size: 1.23rem;
            margin-top: 1.2rem;
            max-width: 58rem;
        }
        .bs-badge {
            display: inline-block;
            border: 1px solid var(--bs-border-strong);
            border-radius: 999px;
            padding: 0.22rem 0.66rem;
            color: var(--bs-blue-strong);
            background: var(--bs-blue-soft);
            font-size: 0.78rem;
            margin-right: 0.35rem;
            margin-top: 1rem;
        }
        .bs-badge-red {
            border-color: rgba(220, 38, 38, 0.25);
            color: var(--bs-red);
            background: var(--bs-red-soft);
        }
        .bs-hero-visual {
            min-height: 13rem;
            border: 1px solid var(--bs-border);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(56, 189, 248, 0.12)),
                #ffffff;
            padding: 1rem;
            display: grid;
            align-content: center;
            gap: 0.55rem;
            box-shadow: 0 24px 46px rgba(37, 99, 235, 0.12);
        }
        .bs-node-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.45rem;
        }
        .bs-node {
            border: 1px solid var(--bs-border);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.52rem 0.3rem;
            text-align: center;
            font-size: 0.78rem;
            color: var(--bs-blue-strong);
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.07);
        }
        .bs-node-hot {
            border-color: rgba(220, 38, 38, 0.24);
            color: var(--bs-red);
            background: rgba(220, 38, 38, 0.05);
        }
        .bs-hero-rule {
            height: 1px;
            background: linear-gradient(90deg, var(--bs-blue), var(--bs-light-blue), transparent);
            margin-top: 1.15rem;
        }
        .bs-sidebar-nav {
            display: flex;
            flex-direction: column;
            gap: 0.48rem;
            margin-top: 0.7rem;
            width: 100%;
        }
        .bs-sidebar-nav a {
            display: block;
            color: var(--bs-red);
            text-decoration: none;
            border: 1px solid rgba(220, 38, 38, 0.28);
            background: var(--bs-red-soft);
            border-radius: 999px;
            padding: 0.56rem 0.8rem;
            font-size: 0.9rem;
            width: 100%;
            box-sizing: border-box;
            text-align: center;
            line-height: 1.15;
        }
        .bs-sidebar-nav a:hover {
            background: rgba(220, 38, 38, 0.16);
            border-color: var(--bs-red);
            color: #991b1b;
        }
        .bs-sidebar-nav a.bs-nav-active {
            background: rgba(220, 38, 38, 0.18);
            border-color: var(--bs-red);
            color: #991b1b;
            font-weight: 650;
        }
        [data-testid="stSidebar"] .stButton {
            width: 100%;
        }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border: 1px solid var(--bs-border-strong) !important;
            border-radius: 999px !important;
            background: var(--bs-blue-soft) !important;
            padding: 0.48rem 0.8rem !important;
        }
        [data-testid="stSidebar"] .stButton > button p,
        [data-testid="stSidebar"] .stButton > button span,
        [data-testid="stSidebar"] .stButton > button div {
            color: var(--bs-blue-strong) !important;
            font-weight: 620;
        }
        .bs-anchor {
            scroll-margin-top: 5.2rem;
        }
        .bs-section {
            margin-top: 2.35rem;
            padding: 0;
            border: 0;
            background: transparent;
            box-shadow: none;
        }
        .bs-section-title {
            color: var(--bs-ink);
            font-size: 1.6rem;
            font-weight: 650;
            margin: 0 0 0.25rem;
        }
        .bs-section-note {
            color: var(--bs-muted);
            margin-bottom: 1.05rem;
        }
        .bs-callout {
            border: 1px solid var(--bs-border);
            border-radius: 8px;
            background: var(--bs-blue-soft);
            padding: 0.72rem 0.9rem;
            color: var(--bs-blue-strong);
            margin-bottom: 1rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--bs-border);
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
        }
        [data-testid="stPlotlyChart"] {
            border: 1px solid var(--bs-border);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.75rem;
            box-shadow: 0 10px 28px rgba(37, 99, 235, 0.06);
        }
        div[data-testid="stCodeBlock"] {
            border: 1px solid var(--bs-border);
            border-radius: 8px;
            background: #ffffff;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea {
            background-color: #ffffff !important;
            border-color: var(--bs-border-strong) !important;
            color: var(--bs-ink) !important;
        }
        .stButton > button {
            border-radius: 8px;
            border-color: var(--bs-blue) !important;
        }
        .stButton > button[kind="primary"] {
            background: var(--bs-blue) !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"] *,
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] div {
            color: #ffffff !important;
        }
        hr {
            border-color: var(--bs-border);
            margin: 2rem 0 0.5rem;
        }
        @media (max-width: 900px) {
            .bs-hero-grid {
                grid-template-columns: 1fr;
                gap: 0.9rem;
            }
            .bs-title {
                font-size: 2.65rem;
            }
            .bs-title-icon {
                width: 2.8rem;
                height: 2.8rem;
            }
            .bs-subtitle {
                font-size: 1.04rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="bs-hero">
            <div class="bs-hero-grid">
                <div>
                    <div class="bs-title-wrap">
                        <span class="bs-title-icon" aria-hidden="true">
                            <svg viewBox="0 0 256 256" role="img" focusable="false" xmlns="http://www.w3.org/2000/svg">
                                <g fill="none" stroke="#0ea5d8" stroke-width="10" stroke-linecap="round" stroke-linejoin="round">
                                    <circle cx="37" cy="22" r="16"/>
                                    <circle cx="219" cy="22" r="16"/>
                                    <path d="M37 38v20m182-20v20"/>
                                    <path d="M31 58h12m170 0h12"/>
                                    <rect x="10" y="73" width="236" height="156" rx="12"/>
                                    <path d="M10 111h31l13 25h148l13-25h31"/>
                                    <path d="M29 73V58h18v15m33 0V58h18v15m33 0V58h18v15m33 0V58h18v15m33 0V58h18v15"/>
                                    <path d="M25 229v11h34v-11m27 0v11h34v-11m27 0v11h34v-11m27 0v11h34v-11"/>
                                    <circle cx="128" cy="156" r="47"/>
                                    <path d="M139 119l-17 61h24l-31 47 10-55h-18l15-53z"/>
                                    <path d="M211 22h17"/>
                                    <path d="M29 22l7 8 13-15"/>
                                </g>
                            </svg>
                        </span>
                        <div class="bs-title">BERTokenScope</div>
                    </div>
                    <div class="bs-subtitle">
                        Transformer intelligence for attention visualization, financial NLP,
                        masked-token prediction, embeddings, and explainable model behavior.
                    </div>
                    <span class="bs-badge">Attention Explorer</span>
                    <span class="bs-badge">Financial NLP</span>
                    <span class="bs-badge">Embedding Intelligence</span>
                    <span class="bs-badge bs-badge-red">Offline-safe demo</span>
                </div>
                <div class="bs-hero-visual">
                    <div class="bs-node-row">
                        <div class="bs-node">[CLS]</div>
                        <div class="bs-node">company</div>
                        <div class="bs-node bs-node-hot">risk</div>
                        <div class="bs-node">[MASK]</div>
                    </div>
                    <div class="bs-node-row">
                        <div class="bs-node">Layer 1</div>
                        <div class="bs-node">Head 2</div>
                        <div class="bs-node">Head 3</div>
                        <div class="bs-node">Layer 2</div>
                    </div>
                    <div class="bs-node-row">
                        <div class="bs-node">earnings</div>
                        <div class="bs-node">revenue</div>
                        <div class="bs-node bs-node-hot">uncertainty</div>
                        <div class="bs-node">growth</div>
                    </div>
                </div>
            </div>
            <div class="bs-hero-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    if "active_pane" not in st.session_state:
        st.session_state["active_pane"] = "Dashboard"

    with st.sidebar:
        st.header("BERTokenScope")
        st.metric("Deployment Mode", "Offline-safe demo")
        st.caption("Live model serving is available through the FastAPI backend.")
        st.markdown("---")
        if st.button("Dashboard", width="stretch"):
            st.session_state["active_pane"] = "Dashboard"
        if st.button("Run History", width="stretch"):
            st.session_state["active_pane"] = "Run History"
        view = str(st.session_state["active_pane"])
        if view == "Dashboard":
            st.markdown("---")
            st.markdown(
                """
                <div class="bs-sidebar-nav">
                    <a href="#overview">Overview</a>
                    <a href="#masked-word-lab">Masked Word</a>
                    <a href="#attention-explorer">Attention</a>
                    <a href="#explainability-lab">Explainability</a>
                    <a href="#financial-nlp">Financial NLP</a>
                    <a href="#embedding-explorer">Embeddings</a>
                    <a href="#model-comparison">Model Comparison</a>
                </div>
                """,
                unsafe_allow_html=True,
            )
    return view


def open_section(anchor: str, title: str, note: str) -> None:
    st.markdown(
        f"""
        <div id="{anchor}" class="bs-anchor"></div>
        <div class="bs-section">
            <div class="bs-section-title">{title}</div>
            <div class="bs-section-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def attention_engine() -> AttentionEngine:
    return AttentionEngine(offline=True)


@st.cache_resource
def financial_analyzer() -> FinancialTextAnalyzer:
    return FinancialTextAnalyzer()


@st.cache_resource
def transcript_analyzer() -> TranscriptAnalyzer:
    return TranscriptAnalyzer()


@st.cache_resource
def embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()


@st.cache_resource
def comparison_engine() -> ModelComparisonEngine:
    return ModelComparisonEngine()


@st.cache_resource
def explainability_engine() -> ExplainabilityEngine:
    return ExplainabilityEngine(offline=True)


@st.cache_resource
def run_tracker() -> RunTracker:
    return RunTracker()


def render_overview() -> None:
    open_section(
        "overview",
        "Executive Overview",
        "A public dashboard for transformer interpretability and finance-aware NLP intelligence.",
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Architecture", "BERT-style", delta="Transformer")
    metric_cols[1].metric("Primary Task", "Masked LM", delta="Top-k")
    metric_cols[2].metric("Explainability", "Attention", delta="Token-level")
    metric_cols[3].metric("Runtime", "Fallback-ready", delta="Public demo")
    st.markdown(
        """
        <div class="bs-callout">
            BERTokenScope is deployed as a reliable Streamlit demo while the repository
            preserves the enterprise FastAPI, observability, security, governance,
            and model-serving architecture.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_masked_word_lab() -> None:
    open_section(
        "masked-word-lab",
        "Masked Word Lab",
        "Inspect BERT-style masked-token predictions and probability distributions.",
    )
    text = st.text_area(
        "Input text",
        value="The company reported record [MASK] this quarter.",
        height=120,
        key="masked_text",
    )
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, key="masked_top_k")
    if st.button("Run prediction", type="primary", width="stretch"):
        try:
            result = attention_engine().analyze_mask(text, top_k=top_k)
            st.caption(f"Model: {result.model_name}")
            st.plotly_chart(prediction_bar_chart(result.predictions), width="stretch")
            st.dataframe(
                [prediction.model_dump() for prediction in result.predictions],
                hide_index=True,
                width="stretch",
            )
        except Exception as exc:
            st.error(str(exc))


def render_attention_explorer() -> None:
    open_section(
        "attention-explorer",
        "Attention Explorer",
        "Explore token-to-token attention, attention rollout, and head diagnostics.",
    )
    sample = st.text_input(
        "Sentence",
        value="The bank was muddy because the river overflowed near the [MASK].",
    )
    controls = st.columns([1, 1, 1])
    requested_layer = controls[0].number_input("Layer", min_value=1, value=1, step=1)
    requested_head = controls[1].number_input("Head", min_value=1, value=1, step=1)
    export_diagram = controls[2].checkbox("Export PNG diagram", value=False)

    if st.button("Analyze attention", type="primary", width="stretch"):
        try:
            result = attention_engine().explore_attention(
                sample,
                layer=int(requested_layer),
                head=int(requested_head),
                export_diagram=export_diagram,
            )
            st.caption(
                f"Model: {result.model_name} | Layers: {result.layer_count} | "
                f"Heads per layer: {result.head_count}"
            )
            st.code(" ".join(result.tokens))

            heatmap_col, rollout_col = st.columns(2)
            heatmap_col.plotly_chart(
                attention_heatmap(
                    result.tokens,
                    result.attention_matrix,
                    title=f"Layer {result.selected_layer}, Head {result.selected_head}",
                ),
                width="stretch",
            )
            rollout_col.plotly_chart(
                attention_heatmap(
                    result.tokens,
                    result.rollout_matrix,
                    title="Attention Rollout",
                ),
                width="stretch",
            )

            link_col, summary_col = st.columns(2)
            link_col.markdown("#### Strongest Token Links")
            link_col.dataframe(
                [link.model_dump() for link in result.strongest_links],
                hide_index=True,
                width="stretch",
            )
            summary_col.markdown("#### Head Diagnostics")
            summary_col.dataframe(
                [summary.model_dump() for summary in result.head_summaries],
                hide_index=True,
                width="stretch",
            )
            if result.diagram_path:
                st.success(f"Diagram exported to {result.diagram_path}")
        except Exception as exc:
            st.error(str(exc))


def render_explainability_lab() -> None:
    open_section(
        "explainability-lab",
        "Explainability Lab",
        "Generate token attribution, rationales, and counterfactual diagnostics.",
    )
    explanation_task = st.radio(
        "Task",
        ["financial-sentiment", "masked-language"],
        horizontal=True,
    )
    default_explain_text = (
        "Management reported strong revenue growth, but noted inflation risk "
        "and uncertain guidance."
        if explanation_task == "financial-sentiment"
        else "The company reported record [MASK] this quarter."
    )
    explain_text = st.text_area(
        "Text to explain", value=default_explain_text, height=140
    )
    explain_cols = st.columns(2)
    explain_layer = explain_cols[0].number_input(
        "Explain layer", min_value=1, value=1, step=1
    )
    explain_head = explain_cols[1].number_input(
        "Explain head", min_value=1, value=1, step=1
    )
    if st.button("Explain prediction", type="primary", width="stretch"):
        try:
            if explanation_task == "masked-language":
                report = explainability_engine().explain_masked_language(
                    explain_text,
                    layer=int(explain_layer),
                    head=int(explain_head),
                )
            else:
                report = explainability_engine().explain_financial_text(explain_text)

            metric_cols = st.columns(3)
            metric_cols[0].metric("Prediction", report.prediction)
            metric_cols[1].metric("Confidence", round(report.confidence, 4))
            metric_cols[2].metric("Model", report.model_name)
            st.plotly_chart(token_attribution_chart(report), width="stretch")
            st.plotly_chart(counterfactual_impact_chart(report), width="stretch")
            st.markdown("#### Rationale")
            for item in report.rationale:
                st.write(f"- {item}")
            st.markdown("#### Token Attribution Table")
            st.dataframe(
                [item.model_dump() for item in report.token_attributions],
                hide_index=True,
                width="stretch",
            )
            if report.counterfactuals:
                st.markdown("#### Counterfactual Table")
                st.dataframe(
                    [item.model_dump() for item in report.counterfactuals],
                    hide_index=True,
                    width="stretch",
                )
        except Exception as exc:
            st.error(str(exc))


def render_financial_nlp() -> None:
    open_section(
        "financial-nlp",
        "Financial NLP Intelligence",
        "Analyze sentiment, risk language, uncertainty, optimism, and transcript drift.",
    )
    finance_tabs = st.tabs(["Single Text", "Transcript Drift"])
    with finance_tabs[0]:
        finance_text = st.text_area(
            "Financial text",
            value=(
                "Management reported strong revenue growth and improved demand, "
                "but noted inflation risk and uncertain guidance for the next quarter."
            ),
            height=160,
        )
        if st.button("Analyze financial language", type="primary", width="stretch"):
            analysis = financial_analyzer().analyze(finance_text)
            cols = st.columns(4)
            cols[0].metric("Sentiment", analysis.sentiment.label)
            cols[1].metric("Risk", analysis.risk_score)
            cols[2].metric("Uncertainty", analysis.uncertainty_score)
            cols[3].metric("Optimism", analysis.optimism_score)
            st.caption(f"Model: {analysis.model_name}")
            st.plotly_chart(financial_signal_chart(analysis), width="stretch")
            st.write(analysis.sentiment.explanation)
            st.json(analysis.matched_terms)

    with finance_tabs[1]:
        company = st.text_input("Company", value="Example Corp")
        q1 = st.text_area(
            "Earlier period transcript",
            value=(
                "Q1 revenue growth was strong and demand improved across enterprise customers. "
                "Management expressed confidence in operating momentum."
            ),
            height=130,
        )
        q2 = st.text_area(
            "Latest period transcript",
            value=(
                "Q2 revenue expanded, but management noted inflation risk, uncertain guidance, "
                "and possible headwinds in discretionary spending."
            ),
            height=130,
        )
        if st.button("Analyze transcript drift", type="primary", width="stretch"):
            report = transcript_analyzer().analyze_report(
                [
                    PeriodDocument(period="Earlier", text=q1),
                    PeriodDocument(period="Latest", text=q2),
                ],
                company=company,
            )
            st.markdown("#### Executive Summary")
            for item in report.executive_summary:
                st.write(f"- {item}")
            st.plotly_chart(tone_drift_chart(report), width="stretch")
            st.plotly_chart(transcript_chunk_chart(report), width="stretch")
            st.dataframe(
                [point.model_dump() for point in report.tone_drift],
                hide_index=True,
                width="stretch",
            )


def parse_documents(raw_documents: str) -> list[EmbeddingDocument]:
    documents = []
    for index, line in enumerate(raw_documents.splitlines(), start=1):
        if not line.strip():
            continue
        if "|" in line:
            label, text = [part.strip() for part in line.split("|", 1)]
        else:
            label, text = f"Document {index}", line.strip()
        documents.append(EmbeddingDocument(label=label, text=text))
    return documents


def render_embedding_explorer() -> None:
    open_section(
        "embedding-explorer",
        "Embedding Explorer",
        "Map documents into semantic space for clustering and similarity analysis.",
    )
    sample_documents = st.text_area(
        "Documents",
        value=(
            "Northwind Q1 | Revenue growth was strong and enterprise demand improved.\n"
            "Northwind Q2 | Inflation risk and uncertain guidance created headwinds.\n"
            "Contoso Q1 | Cloud demand expanded with strong customer retention.\n"
            "Fabrikam Q1 | Litigation risk and recession concerns pressured outlook."
        ),
        height=180,
    )
    cluster_count = st.slider("Clusters", min_value=1, max_value=6, value=3)
    if st.button("Generate embedding map", type="primary", width="stretch"):
        try:
            report = embedding_engine().analyze(
                parse_documents(sample_documents),
                cluster_count=cluster_count,
                similarity_limit=20,
            )
            st.caption(f"Model: {report.model_name} | Dimensions: {report.dimensions}")
            st.plotly_chart(embedding_scatter_chart(report), width="stretch")
            st.plotly_chart(similarity_heatmap(report), width="stretch")
            st.markdown("#### Closest Document Pairs")
            st.dataframe(
                [pair.model_dump() for pair in report.similarities],
                hide_index=True,
                width="stretch",
            )
        except Exception as exc:
            st.error(str(exc))


def render_model_comparison() -> None:
    open_section(
        "model-comparison",
        "Model Comparison",
        "Compare transformer-family behavior across NLP tasks and runtime metrics.",
    )
    compare_masked_text = st.text_input(
        "Masked language prompt",
        value="The company reported record [MASK] this quarter.",
    )
    compare_financial_text = st.text_area(
        "Financial text",
        value=(
            "Management reported strong revenue growth, but noted inflation risk "
            "and uncertain guidance."
        ),
        height=120,
    )
    compare_documents = st.text_area(
        "Embedding comparison documents",
        value=(
            "A | Revenue growth and enterprise demand improved.\n"
            "B | Strong revenue growth and customer demand continued.\n"
            "C | Litigation risk and recession concerns pressured guidance."
        ),
        height=140,
    )
    if st.button("Compare models", type="primary", width="stretch"):
        try:
            report = comparison_engine().compare(
                masked_text=compare_masked_text,
                financial_text=compare_financial_text,
                documents=parse_documents(compare_documents),
            )
            st.plotly_chart(model_latency_chart(report), width="stretch")
            st.plotly_chart(model_confidence_chart(report), width="stretch")
            st.markdown("#### Masked Language Runs")
            st.dataframe(
                [run.model_dump() for run in report.masked_language.runs],
                hide_index=True,
                width="stretch",
            )
            st.markdown("#### Financial Sentiment Runs")
            st.dataframe(
                [run.model_dump() for run in report.financial_sentiment.runs],
                hide_index=True,
                width="stretch",
            )
            st.markdown("#### Embedding Runs")
            st.dataframe(
                [run.model_dump() for run in report.embeddings.runs],
                hide_index=True,
                width="stretch",
            )
            st.markdown("#### Prediction Overlap")
            st.json(report.masked_language.prediction_overlap)
            st.markdown("#### Recommendation")
            for item in report.recommendation:
                st.write(f"- {item}")
        except Exception as exc:
            st.error(str(exc))


def render_run_history() -> None:
    open_section(
        "run-history",
        "Run History",
        "Dedicated in-app pane for persisted analysis artifacts and experiment metadata.",
    )
    history_limit = st.slider("Runs to show", min_value=5, max_value=100, value=20)
    records = run_tracker().list_runs(limit=history_limit)
    if not records:
        st.info(
            "No persisted runs yet. API-triggered workflows are recorded automatically."
        )
        return
    st.dataframe(
        [record.model_dump() for record in records],
        hide_index=True,
        width="stretch",
    )
    selected_run = st.selectbox("Artifact", [record.run_id for record in records])
    if selected_run:
        try:
            st.json(run_tracker().read_artifact(selected_run))
        except Exception as exc:
            st.error(str(exc))


def render_dashboard_page() -> None:
    render_overview()
    render_masked_word_lab()
    render_attention_explorer()
    render_explainability_lab()
    render_financial_nlp()
    render_embedding_explorer()
    render_model_comparison()


def main() -> None:
    configure_page()
    render_hero()
    view = render_sidebar()
    if view == "Run History":
        render_run_history()
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
