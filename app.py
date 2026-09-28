"""Streamlit dashboard: KPI overview, anomaly review, a grounded AI chat, and a
"How it works" tab that shows the pipeline, the guardrail, and live detection
accuracy against the synthetic data's answer key.

Chart design follows a validated categorical/sequential/diverging palette
(fixed hue order, colorblind-checked) rather than default chart colors.
Colors are defined once below and reused everywhere so the whole dashboard
reads as one system.
"""

import time

import altair as alt
import pandas as pd
import streamlit as st

from src.anomalies import (
    BUDGET_OVERRUN_THRESHOLD,
    DUPLICATE_DATE_WINDOW_DAYS,
    PROCESSING_TIME_IQR_MULTIPLIER,
    detect_all,
)
from src.answers import BRIEFING_QUESTION, SUGGESTED_QUESTIONS, answer
from src.evaluation import detection_scorecard, load_ground_truth
from src.insights import GEMINI_MODELS
from src.kpis import (
    avg_processing_days,
    budget_variance_extremes,
    load_budgets,
    load_invoices,
    spend_by_category,
    spend_by_department,
    spend_by_month,
    total_spend,
)
from src.retrieval import enrich_anomalies_with_dims, retrieve

st.set_page_config(page_title="Finance Ops AI Insights", page_icon="📊", layout="wide")

# --- Design tokens (validated palette) -----------------------------------

SERIES_1 = "#2a78d6"       # blue — single-hue bars/area (magnitude, one series)
DIVERGE_UNDER = "#2a78d6"  # blue — under budget
DIVERGE_OVER = "#e34948"   # red — over budget
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT_SECONDARY = "#52514e"

ANOMALY_LABELS = {
    "duplicate_payment": "Duplicate Payment",
    "processing_time_outlier": "Processing Delay",
    "budget_overrun": "Budget Overrun",
}
ANOMALY_COLORS = {
    "duplicate_payment": "#2a78d6",        # categorical slot 1
    "processing_time_outlier": "#eb6834",  # categorical slot 2
    "budget_overrun": "#1baf7a",           # categorical slot 3
}
ANOMALY_RULES = {
    "duplicate_payment": f"Same vendor and same amount, paid within {DUPLICATE_DATE_WINDOW_DAYS} days of each other.",
    "processing_time_outlier": (
        f"Submit-to-pay time beyond the extreme-outlier fence (Q3 + {PROCESSING_TIME_IQR_MULTIPLIER:g} × IQR)."
    ),
    "budget_overrun": (
        f"A department's monthly spend in a category exceeds {BUDGET_OVERRUN_THRESHOLD:.0%} of its budget."
    ),
}

st.markdown(
    """
    <style>
    .block-container, [data-testid="stMainBlockContainer"] { padding-top: 2.5rem; }
    .pill {
        display: inline-block;
        padding: 4px 12px;
        margin: 0 8px 6px 0;
        border-radius: 999px;
        background: rgba(42, 120, 214, 0.10);
        color: #184f95;
        font-size: 12.5px;
        font-weight: 600;
    }
    .step-num {
        display: inline-block;
        width: 24px;
        height: 24px;
        line-height: 24px;
        text-align: center;
        border-radius: 50%;
        background: #2a78d6;
        color: #fff;
        font-size: 13px;
        font-weight: 700;
        margin-right: 8px;
    }
    .step-title { font-weight: 700; font-size: 15px; }
    .step-body { color: #52514e; font-size: 13.5px; margin-top: 6px; }
    .anomaly-card { border-left: 4px solid; padding: 2px 0 2px 12px; }
    .anomaly-count { font-size: 30px; font-weight: 700; line-height: 1.2; }
    .anomaly-rule { color: #52514e; font-size: 13px; }
    div[data-testid="stMetricLabel"] { font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Charts ---------------------------------------------------------------

def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=GRID,
            domainColor=AXIS,
            tickColor=AXIS,
            labelColor=TEXT_SECONDARY,
            titleColor=TEXT_SECONDARY,
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_legend(labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY)
    )


def hbar_chart(df: pd.DataFrame, category_field: str) -> alt.Chart:
    # Horizontal bars so long labels ("Software & IT Services") stay readable.
    chart = (
        alt.Chart(df)
        .mark_bar(color=SERIES_1, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=20)
        .encode(
            y=alt.Y(f"{category_field}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            x=alt.X("total_spend:Q", title="Spend (EUR)", axis=alt.Axis(format="~s")),
            tooltip=[
                alt.Tooltip(f"{category_field}:N", title=category_field.title()),
                alt.Tooltip("total_spend:Q", title="Spend (EUR)", format=",.0f"),
                alt.Tooltip("invoice_count:Q", title="Invoices"),
            ],
        )
        .properties(height=alt.Step(34))
    )
    return style_chart(chart)


def trend_chart(df: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(df).encode(x=alt.X("month:O", title=None))
    area = base.mark_area(color=SERIES_1, opacity=0.12).encode(
        y=alt.Y("total_spend:Q", title="Spend (EUR)", axis=alt.Axis(format="~s"))
    )
    line = base.mark_line(color=SERIES_1, strokeWidth=2).encode(y="total_spend:Q")
    points = base.mark_circle(color=SERIES_1, size=45).encode(
        y="total_spend:Q",
        tooltip=[
            alt.Tooltip("month:O", title="Month"),
            alt.Tooltip("total_spend:Q", title="Spend", format=",.0f"),
            alt.Tooltip("invoice_count:Q", title="Invoices"),
        ],
    )
    return style_chart((area + line + points).properties(height=260))


def diverging_variance_chart(df: pd.DataFrame) -> alt.Chart:
    df = df.copy()
    df["direction"] = df["variance_pct"].apply(lambda v: "Over budget" if v > 0 else "Under budget")
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=3, size=18)
        .encode(
            y=alt.Y(
                "label:N",
                sort=alt.EncodingSortField(field="variance_pct", order="descending"),
                title=None,
                axis=alt.Axis(labelLimit=260),
            ),
            x=alt.X("variance_pct:Q", title="Variance vs. budget (%)", axis=alt.Axis(format="+.0f")),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Under budget", "Over budget"], range=[DIVERGE_UNDER, DIVERGE_OVER]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Department · Category (Month)"),
                alt.Tooltip("variance_pct:Q", title="Variance", format="+.1f"),
                alt.Tooltip("actual_amount:Q", title="Actual", format=",.0f"),
                alt.Tooltip("budget_amount:Q", title="Budget", format=",.0f"),
            ],
        )
        .properties(height=alt.Step(26))
    )
    return style_chart(chart)


# --- Data (cached: deterministic, computed once per process) --------------

@st.cache_data
def get_invoices():
    return load_invoices()


@st.cache_data
def get_budgets():
    return load_budgets()


@st.cache_data
def get_anomalies():
    invoices, budgets = get_invoices(), get_budgets()
    return enrich_anomalies_with_dims(detect_all(invoices, budgets), invoices)


@st.cache_data
def get_scorecard():
    return detection_scorecard(detect_all(get_invoices(), get_budgets()), load_ground_truth())


@st.cache_resource
def live_briefing_store():
    return {}


def get_briefing():
    # A *live* briefing is shared across all visitors for a day, so the landing
    # page costs one Gemini call per day, not one per visitor. Pre-generated
    # fallbacks are never stored here, so the next click tries live again.
    store = live_briefing_store()
    cached = store.get("briefing")
    if cached and time.time() - cached["at"] < 24 * 3600:
        return cached["facts"], cached["result"]
    facts = retrieve(BRIEFING_QUESTION, get_invoices(), get_budgets())
    result = answer(facts)
    if result["source"] == "live":
        store["briefing"] = {"at": time.time(), "facts": facts, "result": result}
    return facts, result


def source_caption(result):
    if result["source"] == "live":
        return f"Live answer · {result['model']}"
    return (
        f"Gemini's free tier is busy, so this is a pre-generated answer ({result['model']}, "
        f"{result['generated_on']}) for exactly these facts."
    )


def filter_by_dims(df, department, category):
    if department != "All":
        df = df[df["department"] == department]
    if category != "All":
        df = df[df["category"] == category]
    return df


invoices = get_invoices()
budgets = get_budgets()
anomalies = get_anomalies()

# --- Header ----------------------------------------------------------------

st.title("Finance Operations — AI Insights")
st.markdown(
    "Vendor-invoice analytics where **pandas computes every number** and **Gemini only narrates them** — "
    "so every AI answer can be checked against the exact facts it was given."
)
st.markdown(
    f'<span class="pill">{len(invoices):,} synthetic invoices</span>'
    f'<span class="pill">Rule-based anomaly detection · {get_scorecard()["recall"].min():.0%} recall</span>'
    '<span class="pill">LLM narrates, never calculates</span>',
    unsafe_allow_html=True,
)

step_cols = st.columns(3)
steps = [
    ("Compute", "KPIs and anomaly rules run in plain pandas — deterministic and unit-tested. No AI involved."),
    ("Retrieve", "Your question selects only the relevant precomputed facts. Raw invoice rows never leave this step."),
    ("Narrate", "Gemini turns those facts into plain English, under instructions that forbid calculating anything."),
]
for i, (col, (title, body)) in enumerate(zip(step_cols, steps), start=1):
    with col, st.container(border=True):
        st.markdown(
            f'<span class="step-num">{i}</span><span class="step-title">{title}</span>'
            f'<div class="step-body">{body}</div>',
            unsafe_allow_html=True,
        )

with st.container(border=True):
    brief_col, button_col = st.columns([4, 1], vertical_alignment="center")
    brief_col.markdown("**AI briefing** — a one-click executive summary of the whole dataset.")
    if button_col.button("Generate briefing", type="primary", width="stretch"):
        try:
            with st.spinner("Asking Gemini..."):
                st.session_state.briefing = get_briefing()
        except Exception as e:
            st.session_state.briefing = None
            st.warning(f"Couldn't generate the briefing: {e}")
    if st.session_state.get("briefing"):
        briefing_facts, briefing_result = st.session_state.briefing
        st.markdown(briefing_result["text"])
        st.caption(source_caption(briefing_result))
        with st.expander("Facts Gemini was given (every number above comes from here)"):
            st.json(briefing_facts)

# --- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    department = st.selectbox("Department", ["All"] + sorted(invoices["department"].unique()))
    category = st.selectbox("Category", ["All"] + sorted(invoices["category"].unique()))
    st.caption(
        "Filters apply to the Overview and Anomalies tabs. In **Ask the AI**, just name a "
        "department or category in your question."
    )
    st.divider()
    st.markdown("**About this demo**")
    st.caption(
        "All data is synthetic (generated with Faker), with anomalies deliberately planted so "
        "detection accuracy can be measured. See the **How it works** tab."
    )

scoped_invoices = filter_by_dims(invoices, department, category)
scoped_anomalies = filter_by_dims(anomalies, department, category)
flagged_invoice_share = (
    scoped_anomalies["invoice_id"].nunique() / len(scoped_invoices) * 100 if len(scoped_invoices) else 0.0
)

overview_tab, anomalies_tab, chat_tab, how_tab = st.tabs(
    ["Overview", "Anomalies", "Ask the AI", "How it works"]
)

# --- Overview --------------------------------------------------------------

with overview_tab:
    col1, col2, col3, col4 = st.columns(4)
    with col1, st.container(border=True):
        st.metric("Total Spend", f"€{total_spend(scoped_invoices):,.0f}", help="Sum of all invoice amounts in scope.")
    with col2, st.container(border=True):
        st.metric("Invoices", f"{len(scoped_invoices):,}")
    with col3, st.container(border=True):
        st.metric(
            "Avg Processing Time",
            f"{avg_processing_days(scoped_invoices):.1f} days",
            help="Average days from submission to payment. Invoices with no submission date are excluded.",
        )
    with col4, st.container(border=True):
        st.metric(
            "Anomalies Flagged",
            f"{len(scoped_anomalies):,}",
            delta=f"{flagged_invoice_share:.1f}% of invoices affected",
            delta_color="off",
            delta_arrow="off",
            help="Duplicate payments and processing delays (per invoice) plus budget overruns "
            "(per department, category and month). Details in the Anomalies tab.",
        )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1, st.container(border=True):
        st.subheader("Spend by Department")
        st.altair_chart(hbar_chart(spend_by_department(scoped_invoices), "department"), width="stretch")
    with chart_col2, st.container(border=True):
        st.subheader("Spend by Category")
        st.altair_chart(hbar_chart(spend_by_category(scoped_invoices), "category"), width="stretch")

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3, st.container(border=True):
        st.subheader("Monthly Spend Trend")
        # The dataset starts mid-month; plotting that stub month would look like a spend collapse.
        first_day = invoices["invoice_date"].min()
        monthly = spend_by_month(scoped_invoices)
        if first_day.day != 1:
            monthly = monthly[monthly["month"] != first_day.strftime("%Y-%m")]
            st.caption(f"Data starts {first_day:%d %b %Y}; that partial first month is omitted.")
        st.altair_chart(trend_chart(monthly), width="stretch")
    with chart_col4, st.container(border=True):
        st.subheader("Largest Budget Variances")
        st.caption("Biggest over- and under-spends vs. budget, company-wide (not affected by filters).")
        st.altair_chart(diverging_variance_chart(budget_variance_extremes(invoices, budgets)), width="stretch")

# --- Anomalies ---------------------------------------------------------------

with anomalies_tab:
    st.caption(
        "Three simple, explainable rules — no machine learning, so every flag can be justified to an auditor."
    )
    card_cols = st.columns(3)
    for col, (anomaly_type, label) in zip(card_cols, ANOMALY_LABELS.items()):
        count = int((scoped_anomalies["anomaly_type"] == anomaly_type).sum())
        with col, st.container(border=True):
            st.markdown(
                f'<div class="anomaly-card" style="border-color:{ANOMALY_COLORS[anomaly_type]}">'
                f'<div class="step-title">{label}</div>'
                f'<div class="anomaly-count">{count}</div>'
                f'<div class="anomaly-rule">{ANOMALY_RULES[anomaly_type]}</div></div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.subheader("Flagged Records")
        anomaly_type_filter = st.multiselect(
            "Show types",
            options=list(ANOMALY_LABELS),
            default=list(ANOMALY_LABELS),
            format_func=ANOMALY_LABELS.get,
        )
        display_anomalies = scoped_anomalies[scoped_anomalies["anomaly_type"].isin(anomaly_type_filter)].copy()
        display_anomalies["anomaly_type"] = display_anomalies["anomaly_type"].map(ANOMALY_LABELS)
        display_anomalies = display_anomalies.rename(
            columns={
                "anomaly_type": "Type",
                "invoice_id": "Invoice ID",
                "department": "Department",
                "category": "Category",
                "month": "Month",
                "detail": "Why it was flagged",
            }
        ).fillna("—")

        label_to_hex = {v: ANOMALY_COLORS[k] for k, v in ANOMALY_LABELS.items()}

        def tint_type(val):
            hex_color = label_to_hex.get(val)
            return f"background-color: {hex_color}22; color: {hex_color}; font-weight: 600;" if hex_color else ""

        st.dataframe(display_anomalies.style.map(tint_type, subset=["Type"]), width="stretch", hide_index=True)

# --- Ask the AI --------------------------------------------------------------

with chat_tab:
    st.caption(
        "Ask in plain English. Your question picks the relevant precomputed facts; Gemini narrates only those. "
        "Open **Facts Gemini was given** under any answer to check every number."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    suggestion_cols = st.columns(len(SUGGESTED_QUESTIONS) + 1)
    for col, suggestion in zip(suggestion_cols, SUGGESTED_QUESTIONS):
        if col.button(suggestion, width="stretch"):
            st.session_state.pending_question = suggestion
    if st.session_state.messages and suggestion_cols[-1].button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    # Messages render into this container, which sits above the input box, so
    # the conversation reads top-to-bottom with the input always last.
    conversation = st.container()
    question = st.chat_input("e.g. Which department went most over budget?")
    if not question and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    with conversation:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("source_caption"):
                    st.caption(msg["source_caption"])
                if msg.get("facts") is not None:
                    with st.expander("Facts Gemini was given"):
                        st.json(msg["facts"])

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            facts, caption = None, None
            with st.chat_message("assistant"):
                try:
                    facts = retrieve(question, invoices, budgets)
                    with st.spinner("Asking Gemini..."):
                        result = answer(facts)
                    reply, caption = result["text"], source_caption(result)
                except Exception as e:
                    reply = f"Couldn't generate an answer: {e}"
                st.markdown(reply)
                if caption:
                    st.caption(caption)
                if facts is not None:
                    with st.expander("Facts Gemini was given"):
                        st.json(facts)

            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "source_caption": caption, "facts": facts}
            )

# --- How it works ------------------------------------------------------------

with how_tab:
    st.subheader("Pipeline")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR; bgcolor="transparent";
            node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6",
                  fontname="sans-serif", fontsize=11, margin="0.15,0.08"];
            edge [color="#8a8980", arrowsize=0.7];
            gen   [label="1. Generate\\nsynthetic invoices\\n+ planted anomalies"];
            clean [label="2. Clean\\npandas notebook"];
            calc  [label="3. Compute\\nKPIs + anomaly rules"];
            ret   [label="4. Retrieve\\nonly relevant facts"];
            llm   [label="5. Narrate\\nGemini", fillcolor="#fdf0e9", color="#eb6834"];
            ui    [label="6. Display\\nthis app"];
            gen -> clean -> calc -> ret -> llm -> ui;
            calc -> ui [style=dashed, label=" charts", fontsize=10, fontcolor="#52514e"];
        }
        """,
        width="stretch",
    )
    st.caption("Data flows one way. Nothing after step 3 ever touches a raw invoice row.")

    guard_col, score_col = st.columns(2)
    with guard_col, st.container(border=True):
        st.subheader("The AI guardrail")
        st.markdown(
            "- **Structural, not just a prompt.** Gemini's request contains only a small JSON of "
            "precomputed facts — there are no raw rows in it to misread.\n"
            "- **No arithmetic.** The system instruction forbids calculating, estimating or inventing "
            "figures; every number must be copied from the facts.\n"
            "- **Auditable.** Every answer shows the exact facts it was given, so anyone can check it.\n"
            "- **Graceful degradation.** Tries "
            + " → ".join(f"`{m}`" for m in GEMINI_MODELS)
            + " live (30-second budget). If all are busy, suggested questions fall back to answers "
            "pre-generated for the exact same facts — always labelled as such."
        )
    with score_col, st.container(border=True):
        st.subheader("Detection accuracy")
        scorecard = get_scorecard()
        st.dataframe(
            pd.DataFrame(
                {
                    "Anomaly": scorecard["anomaly_type"].map(ANOMALY_LABELS),
                    "Planted": scorecard["planted"],
                    "Flagged": scorecard["flagged"],
                    "Recall": scorecard["recall"].map("{:.0%}".format),
                    "Precision": scorecard["precision"].map("{:.0%}".format),
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Scored live against the data generator's answer key, which the detection code never reads. "
            "Recall = share of planted anomalies caught; precision = share of flags that were real."
        )

    with st.container(border=True):
        st.subheader("Honest limitations")
        st.markdown(
            "- Synthetic data: realistic patterns, but not real transactions.\n"
            "- Thresholds were tuned on this synthetic answer key; in production they'd be set with finance stakeholders.\n"
            "- Retrieval is keyword-based (no vector database) — right-sized for a few structured tables, "
            "but it won't understand every phrasing.\n"
            "- Gemini free tier is often overloaded: typed questions may ask you to retry, while the "
            "briefing and suggested questions fall back to clearly labelled pre-generated answers."
        )
