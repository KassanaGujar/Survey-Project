import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Initialize session state for light mode
if 'light_mode' not in st.session_state:
    st.session_state.light_mode = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Survey Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Instrument+Sans:wght@300;400;500;600&display=swap');

[data-testid="collapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
  position: fixed !important;
  top: 0.5rem !important;
  left: 0.5rem !important;
  z-index: 999999 !important;
}
            
html, body, [class*="css"] {
  font-family: 'Instrument Sans', sans-serif;
  background: #f5f2eb !important;
  color: #1a1a1a !important;
}
.main .block-container { padding: 2rem 2.5rem; max-width: 1500px; }

[data-testid="stSidebar"] { background: #1a1a1a !important; }
[data-testid="stSidebar"] * { color: #d4d0c8 !important; }
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSelectbox label {
  font-size: 0.68rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: #666 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: 'Bebas Neue', sans-serif !important;
  color: #f5f2eb !important;
}
[data-testid="stSidebar"] hr { border-color: #2e2e2e !important; }

.page-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  margin-bottom: 2rem;
  padding-bottom: 1.2rem;
  border-bottom: 2px solid #1a1a1a;
}
.page-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 3.5rem;
  letter-spacing: 0.04em;
  line-height: 1;
  color: #e63947;
}
.page-meta { font-size: 0.75rem; color: #888; letter-spacing: 0.08em; text-align: right; }

.stat-row { display: flex; gap: 0.8rem; margin-bottom: 2rem; flex-wrap: wrap; }
.stat-pill {
  background: #1a1a1a; color: #f5f2eb;
  border-radius: 100px; padding: 0.5rem 1.2rem;
  font-size: 0.8rem; letter-spacing: 0.04em;
}
.stat-pill strong { color: #e8c547; margin-right: 0.3rem; }

.sec-label {
  font-size: 0.65rem; letter-spacing: 0.18em;
  text-transform: uppercase; color: #999; margin-bottom: 0.2rem; margin-top: 2rem;
}
.sec-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.8rem; color: #e63947; margin-bottom: 1rem;
}

.card {
  background: #fff; border: 1px solid #e0ddd6;
  border-radius: 12px; padding: 1.2rem 1.4rem 0.6rem; margin-bottom: 1rem;
}
.card-title { font-size: 0.85rem; font-weight: 600; color: #333; margin-bottom: 0.1rem; line-height: 1.3; }
.card-sub { font-size: 0.72rem; color: #aaa; margin-bottom: 0.6rem; }

[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 2px solid #1a1a1a !important; }
[data-baseweb="tab"] {
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 1rem !important; letter-spacing: 0.06em !important;
  padding: 0.6rem 1.4rem !important; color: #aaa !important;
}
[aria-selected="true"][data-baseweb="tab"] { color: #aaa !important; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR7PRKvQbJgVMNPXwdfoep2xbJISE8C_eUHwThVCbU5WZtCk5pztx_ddv_1qjJO4GaVKf1uVmsqrmpE/pub?output=csv"

@st.cache_data(ttl=120)
def load_data():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()
    return df

df_raw = load_data()
all_cols = df_raw.columns.tolist()

# ── Column detection ──────────────────────────────────────────────────────────
age_col       = next((c for c in all_cols if "age"       in c.lower()), None)
party_col     = next((c for c in all_cols if "party"     in c.lower() or "political" in c.lower()), None)
community_col = next((c for c in all_cols if "community" in c.lower()), None)

meta_cols     = {c for c in [age_col, party_col, community_col, all_cols[0]] if c}
question_cols = [c for c in all_cols[1:] if c not in meta_cols]

AGREE_VOCAB = {"agree", "disagree", "strongly agree", "strongly disagree"}

def is_agree_col(series):
    vals = set(series.dropna().str.lower().unique())
    return len(vals & AGREE_VOCAB) / max(len(vals), 1) > 0.4

agree_cols = [c for c in question_cols if df_raw[c].dtype == object and is_agree_col(df_raw[c])]

# ── Constants ─────────────────────────────────────────────────────────────────
AGREE_COLORS = {
    "strongly agree":    "#2d6a4f",
    "agree":             "#52b788",
    "disagree":          "#e76f51",
    "strongly disagree": "#c1121f",
}
ANSWER_ORDER    = ["strongly agree", "agree", "disagree", "strongly disagree"]
AGE_ORDER       = ["Under 18", "18-29", "29-39", "39-49", "49-59", "60+"]
PARTY_ORDER     = ["Democratic Party", "Republican Party", "Green Party",
                   "Libertarian Party", "Prefer not to say/Don't know", "None of these"]
COMMUNITY_ORDER = ["Urban", "Suburbs", "Rural"]

def norm(v):
    return str(v).strip().lower() if pd.notna(v) else "unknown"

def get_color(val):
    return AGREE_COLORS.get(norm(val), "#aaa")

# ── Sidebar ───────────────────────────────────────────────────────────────────
selected_communities = []

with st.sidebar:
    st.markdown("## Filters")
    st.markdown("---")

    light_mode_toggle = st.toggle("Light mode", value=st.session_state.light_mode)
    st.session_state.light_mode = light_mode_toggle

    if age_col:
        age_opts = [a for a in AGE_ORDER if a in df_raw[age_col].dropna().unique().tolist()]
        all_ages = st.checkbox("Select all age groups", value=True, key="all_ages")
        sel_ages = age_opts if all_ages else st.multiselect("Age Group", age_opts, default=age_opts)
    else:
        sel_ages = []

    if party_col:
        party_opts = [p for p in PARTY_ORDER if p in df_raw[party_col].dropna().unique().tolist()]
        all_parties = st.checkbox("Select all parties", value=True, key="all_parties")
        sel_parties = party_opts if all_parties else st.multiselect("Political Party", party_opts, default=party_opts)
    else:
        sel_parties = []

    if community_col:
        comm_opts = [c for c in COMMUNITY_ORDER if c in df_raw[community_col].dropna().unique().tolist()]
        all_comms = st.checkbox("Select all community types", value=True, key="all_comms")
        selected_communities = comm_opts if all_comms else st.multiselect("Community Type", comm_opts, default=comm_opts)
    
    st.markdown("---")
    st.markdown("## Questions")
    sel_qs = st.multiselect(
        "Show questions", agree_cols,
        default=agree_cols[:6] if len(agree_cols) > 6 else agree_cols,
    )
    if not sel_qs:
        sel_qs = agree_cols

    st.markdown("---")
    show_pct    = st.toggle("Show percentages", value=True)
    orientation = st.radio("Bar orientation", ["Horizontal", "Vertical"], index=0)

# Apply light mode CSS if enabled
if st.session_state.light_mode:
    st.markdown("""
    <style>
html, body, [class*="css"] {
  background: #eef0f2 !important;
  color: #1a1a1a !important;
}
    .main { background: #eef0f2 !important; }
    .main .block-container { background: #eef0f2 !important; }
    [data-testid="stAppViewContainer"] { background: #eef0f2 !important; }
    [data-testid="stApp"] { background: #eef0f2 !important; }
    .page-title { color: #87CEEB !important; }
    .page-meta { color: #888 !important; }
    .sec-title { color: #87CEEB !important; }
    .sec-label { color: #888888 !important; }
    .stat-pill { background: #888 !important; color: #1a1a1a !important; }
    .card { background: #1e1e1e !important; border-color: #2a2a2a !important; }
    .card-title { color: #fff !important; }
    .card-sub { color: #999 !important; }
    .page-header { border-bottom-color: #333 !important; }
    [data-baseweb="tab-list"] { border-bottom-color: #333 !important; }
    [data-baseweb="tab"] { color: #555 !important; }
    [aria-selected="true"][data-baseweb="tab"] { color: #333 !important; }
    [data-testid="stSidebar"] { background: #1a1a1a !important; }
    </style>
    """, unsafe_allow_html=True)

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_raw.copy()
if age_col       and sel_ages:              df = df[df[age_col].isin(sel_ages)]
if party_col     and sel_parties:           df = df[df[party_col].isin(sel_parties)]
if community_col and selected_communities:  df = df[df[community_col].isin(selected_communities)]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <div class="page-title">Survey<br>Analysis</div>
  <div class="page-meta">Political Opinion Dashboard<br>Live · Google Sheets sync</div>
</div>
<div class="stat-row">
  <div class="stat-pill"><strong>{len(df)}</strong> responses shown</div>
  <div class="stat-pill"><strong>{len(df_raw)}</strong> total</div>
  {"<div class='stat-pill'><strong>" + str(df[party_col].nunique()) + "</strong> parties</div>" if party_col else ""}
  {"<div class='stat-pill'><strong>" + str(df[age_col].nunique()) + "</strong> age groups</div>" if age_col else ""}
  <div class="stat-pill"><strong>{len(sel_qs)}</strong> questions</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Party Comparison", "Age Groups", "Communities"])

# ── stacked_bar helper ────────────────────────────────────────────────────────

label_color = "#1a1a1a" if st.session_state.light_mode else "#ffffff"

def stacked_bar(dq, x_col, x_order, height=260):
    dq = dq.copy()
    # Normalize answer to lowercase so colors match
    dq["Answer"] = dq["Answer"].str.strip().str.lower()
    # Apply categorical ordering BEFORE plotly sees the data
    dq[x_col]    = pd.Categorical(dq[x_col], categories=x_order, ordered=True)
    dq["Answer"] = pd.Categorical(dq["Answer"], categories=ANSWER_ORDER, ordered=True)
    dq = dq.sort_values([x_col, "Answer"])

    fig = px.bar(
        dq, x=x_col, y="Pct", color="Answer",
        color_discrete_map={a: get_color(a) for a in ANSWER_ORDER},
        barmode="stack", text="Pct",
        category_orders={x_col: x_order, "Answer": ANSWER_ORDER},
    )
    fig.update_traces(
        texttemplate="%{text:.0f}%", textposition="inside",
        textfont=dict(size=10, color="white"),
        width=0.5,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans"),
        margin=dict(l=4, r=4, t=10, b=10), height=height,
        xaxis=dict(title="", tickfont=dict(size=12, color = label_color), categoryorder="array", categoryarray=x_order),
        yaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.45, font=dict(size=10, color = label_color), title=""),
        bargap=0.3,
    )
    return fig



# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-label">All filtered responses</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">How did people respond?</div>', unsafe_allow_html=True)

    for pair in [sel_qs[i:i+2] for i in range(0, len(sel_qs), 2)]:
        cols = st.columns(len(pair))
        for col, q in zip(cols, pair):
            with col:
                counts = df[q].dropna().apply(norm).value_counts()
                counts = counts.reindex(ANSWER_ORDER, fill_value=0)
                vals   = (counts / counts.sum() * 100).round(1) if show_pct else counts
                colors = [get_color(v) for v in vals.index]
                suffix = "%" if show_pct else ""
                labels = [f"{v}{suffix}" for v in vals.values]

                if orientation == "Horizontal":
                    fig = go.Figure(go.Bar(
                        x=vals.values, y=vals.index, orientation='h',
                        marker_color=colors, text=labels, textposition='outside',
                        textfont=dict(size=11, color= label_color),
                    ))
                    fig.update_layout(
                        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                        yaxis=dict(showgrid=False, tickfont=dict(size=12, color=label_color)),
                        height=max(160, len(counts) * 42),
                    )
                else:
                    fig = go.Figure(go.Bar(
                        x=vals.index, y=vals.values,
                        marker_color=colors, text=labels, textposition='outside',
                        textfont=dict(size=11, color=label_color),
                    ))
                    fig.update_layout(
                        xaxis=dict(tickfont=dict(size=11, color=label_color), showgrid=False),
                        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                        height=260,
                    )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Instrument Sans"),
                    margin=dict(l=4, r=60, t=8, b=8), showlegend=False,
                )

                top     = counts.index[0] if len(counts) else "—"
                top_pct = int(counts.iloc[0] / counts.sum() * 100) if counts.sum() > 0 else 0

                st.markdown(f"""
                <div class="card">
                  <div class="card-title">{q}</div>
                  <div class="card-sub">Top: <strong>{top}</strong> · {top_pct}%</div>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

# ── Tab 2: Party Comparison ───────────────────────────────────────────────────
with tab2:
    if not party_col:
        st.warning("No political party column detected.")
    else:
        st.markdown('<div class="sec-label">By Political Party</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Where do parties agree & diverge?</div>', unsafe_allow_html=True)

        party_x = [p for p in PARTY_ORDER if p in df[party_col].dropna().unique().tolist()]

        for q in sel_qs:
            rows = []
            for p in party_x:
                sub = df[df[party_col] == p][q].dropna().apply(norm)
                if len(sub) == 0: continue
                for ans, pct in (sub.value_counts(normalize=True) * 100).items():
                    rows.append({"Party": p, "Answer": ans, "Pct": round(pct, 1)})
            if not rows: continue

            st.markdown(f'<div class="card"><div class="card-title">{q}</div>', unsafe_allow_html=True)
            st.plotly_chart(stacked_bar(pd.DataFrame(rows), "Party", party_x),
                            width='stretch', config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

# ── Tab 3: Age Groups ─────────────────────────────────────────────────────────
with tab3:
    if not age_col:
        st.warning("No age column detected.")
    else:
        st.markdown('<div class="sec-label">By Age Group</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">How do generations differ?</div>', unsafe_allow_html=True)

        age_x = [a for a in AGE_ORDER if a in df[age_col].dropna().unique().tolist()]

        for q in sel_qs:
            rows = []
            for a in age_x:
                sub = df[df[age_col] == a][q].dropna().apply(norm)
                if len(sub) == 0: continue
                for ans, pct in (sub.value_counts(normalize=True) * 100).items():
                    rows.append({"Age": a, "Answer": ans, "Pct": round(pct, 1)})
            if not rows: continue

            st.markdown(f'<div class="card"><div class="card-title">{q}</div>', unsafe_allow_html=True)
            st.plotly_chart(stacked_bar(pd.DataFrame(rows), "Age", age_x),
                            width='stretch', config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

# ── Tab 4: Communities ────────────────────────────────────────────────────────
with tab4:
    if not community_col:
        st.warning("No community column detected.")
    else:
        st.markdown('<div class="sec-label">By Community</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">How does area affect responses?</div>', unsafe_allow_html=True)

        comm_x = [c for c in COMMUNITY_ORDER if c in df[community_col].dropna().unique().tolist()]

        for q in sel_qs:
            rows = []
            for c in comm_x:
                sub = df[df[community_col] == c][q].dropna().apply(norm)
                if len(sub) == 0: continue
                for ans, pct in (sub.value_counts(normalize=True) * 100).items():
                    rows.append({"Community": c, "Answer": ans, "Pct": round(pct, 1)})
            if not rows: continue

            st.markdown(f'<div class="card"><div class="card-title">{q}</div>', unsafe_allow_html=True)
            st.plotly_chart(stacked_bar(pd.DataFrame(rows), "Community", comm_x),
                            width='stretch', config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
