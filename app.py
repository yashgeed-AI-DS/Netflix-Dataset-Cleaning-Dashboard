"""
Netflix Dataset Cleaning Dashboard
===================================
A professional Streamlit web application that showcases a complete,
real-world data-cleaning pipeline for a Netflix movies dataset.

Author : Yash Geed (Artificial Intelligence & Data Science Student)
Stack  : Python, Pandas, NumPy, Streamlit, Matplotlib, Seaborn, Plotly

Run locally:
    streamlit run app.py
"""

import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu

from cleaning import run_pipeline



# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Netflix Dataset Cleaning Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(APP_DIR, "assets")
DEFAULT_DATA_PATH = os.path.join(APP_DIR, "netflix_movies_dirty.csv")
CSS_PATH = os.path.join(ASSETS_DIR, "style.css")


def load_css(path: str):
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css(CSS_PATH)

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#0e1117",
                                     "figure.facecolor": "#0e1117",
                                     "axes.edgecolor": "#31333F",
                                     "axes.labelcolor": "#FAFAFA",
                                     "xtick.color": "#FAFAFA",
                                     "ytick.color": "#FAFAFA",
                                     "text.color": "#FAFAFA",
                                     "grid.color": "#31333F"})

NETFLIX_RED = "#E50914"
PLOTLY_TEMPLATE = "plotly_dark"


# --------------------------------------------------------------------------- #
# Data loading & caching
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_raw_data(file_bytes: bytes | None) -> pd.DataFrame:
    """Load either the bundled sample dataset or a user-uploaded CSV."""
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_csv(DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def get_cleaned_data(raw_df: pd.DataFrame):
    """Run the full cleaning pipeline once and cache the result."""
    cleaned_df, step_results, overall = run_pipeline(raw_df)
    return cleaned_df, step_results, overall


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Small reusable UI components
# --------------------------------------------------------------------------- #
def metric_card_row(items):
    """items: list of (label, value) tuples rendered as metric cards."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def section_title(text: str, emoji: str = ""):
    st.markdown(f'<div class="section-title">{emoji} {text}</div>', unsafe_allow_html=True)


def hero_banner(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-banner">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="app-footer">
            Built with ❤️ using Streamlit &nbsp;|&nbsp;
            <b>Created by Yash Geed</b> &nbsp;|&nbsp;
            Artificial Intelligence &amp; Data Science Student
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar — navigation + data source
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🎬 Netflix Cleaning")
    st.caption("End-to-end data cleaning dashboard")

    page = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Dataset Overview",
            "Missing Values Analysis",
            "Data Cleaning Steps",
            "Before vs After Comparison",
            "Final Dataset",
            "Download Cleaned Dataset",
            "About Project",
        ],
        icons=[
            "house", "table", "search", "tools",
            "arrow-left-right", "check2-square", "download", "info-circle",
        ],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": NETFLIX_RED, "font-size": "16px"},
            "nav-link": {"font-size": "14.5px", "text-align": "left", "margin": "2px 0",
                         "--hover-color": "rgba(229,9,20,0.12)"},
            "nav-link-selected": {"background-color": NETFLIX_RED, "color": "white"},
        },
    )

    st.markdown("---")
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload your own dirty Netflix CSV (optional)", type=["csv"]
    )
    if uploaded_file is not None:
        st.success("Using your uploaded dataset ✅")
    else:
        st.info("Using bundled sample dataset")

    st.markdown("---")
    st.caption("💡 Tip: use the sidebar to explore every stage of the pipeline.")


# --------------------------------------------------------------------------- #
# Load data once, run the cleaning pipeline once (cached)
# --------------------------------------------------------------------------- #
file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
raw_df = load_raw_data(file_bytes)
cleaned_df, step_results, overall = get_cleaned_data(raw_df)


# --------------------------------------------------------------------------- #
# PAGE: Home
# --------------------------------------------------------------------------- #
def page_home():
    hero_banner(
        "🎬 Netflix Dataset Cleaning Dashboard",
        "An interactive, end-to-end walkthrough of a real-world data-cleaning "
        "pipeline for a Netflix movies &amp; TV shows dataset — from raw, messy "
        "CSV data to an analysis-ready DataFrame.",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        section_title("Objective of the Project", "🎯")
        st.markdown(
            """
This application demonstrates a **complete data-cleaning pipeline** built with
Python and Pandas, applied to a deliberately messy Netflix movies dataset.

The goals of the project are to:

- Load and inspect a real-world, messy dataset
- Systematically detect every category of data-quality issue
  (missing values, duplicates, inconsistent formatting, outliers, mixed types)
- Apply reusable cleaning functions to fix each issue
- Validate the final dataset and make it analysis-ready
- Present the entire pipeline through an interactive, professional dashboard

Use the sidebar to move through **Dataset Overview → Missing Values →
Cleaning Steps → Before/After Comparison → Final Dataset → Download**.
            """
        )
    with col2:
        section_title("Technologies Used", "🛠️")
        techs = ["🐍 Python", "🐼 Pandas", "🔢 NumPy", "📊 Streamlit",
                 "📈 Matplotlib", "🎨 Seaborn", "⚡ Plotly"]
        st.markdown("".join(f'<span class="tech-pill">{t}</span>' for t in techs),
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Pipeline Snapshot", "⚙️")
        metric_card_row([
            ("Raw Rows", f'{overall["before"]["rows"]:,}'),
            ("Cleaned Rows", f'{overall["after"]["rows"]:,}'),
        ])

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("What This Dashboard Covers", "🧭")
    feature_cols = st.columns(4)
    features = [
        ("📋", "Dataset Overview", "Shape, dtypes, sample rows & summary stats"),
        ("🔍", "Missing Values", "Tables, percentages, bar charts & heatmaps"),
        ("🧹", "16 Cleaning Steps", "Explanation, code, and result for every fix"),
        ("⬇️", "Download", "Export the final cleaned CSV in one click"),
    ]
    for col, (icon, title, desc) in zip(feature_cols, features):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div style="font-size:1.6rem;">{icon}</div>
                    <b>{title}</b>
                    <p style="opacity:.8; font-size:.85rem; margin-top:.3rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# PAGE: Dataset Overview
# --------------------------------------------------------------------------- #
def page_dataset_overview():
    section_title("Dataset Overview", "📋")
    st.write(
        "A first look at the **raw, uncleaned** dataset — exactly as it was "
        "loaded, before any cleaning has been applied."
    )

    metric_card_row([
        ("Rows", f"{raw_df.shape[0]:,}"),
        ("Columns", f"{raw_df.shape[1]:,}"),
        ("Missing Cells", f"{int(raw_df.isnull().sum().sum()):,}"),
        ("Duplicate Rows", f"{int(raw_df.duplicated().sum()):,}"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔎 Preview (df.head())", "📐 Shape & Columns", "🧬 Data Types (df.info())", "📊 Summary (df.describe())"]
    )

    with tab1:
        n_rows = st.slider("Rows to preview", 5, 50, 10)
        st.dataframe(raw_df.head(n_rows), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**df.shape**")
            st.code(f"{raw_df.shape}", language="python")
            st.markdown(f"➡️ **{raw_df.shape[0]:,} rows** and **{raw_df.shape[1]} columns**")
        with c2:
            st.markdown("**Column names**")
            st.dataframe(pd.DataFrame({"Column": raw_df.columns}), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("**df.info()**")
        buf = io.StringIO()
        raw_df.info(buf=buf)
        st.code(buf.getvalue(), language="text")

        dtype_counts = raw_df.dtypes.astype(str).value_counts().reset_index()
        dtype_counts.columns = ["Data Type", "Count"]
        fig = px.pie(dtype_counts, names="Data Type", values="Count",
                     title="Column Data Type Distribution", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("**df.describe()**")
        st.dataframe(raw_df.describe(include="all").transpose(), use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE: Missing Values Analysis
# --------------------------------------------------------------------------- #
def page_missing_values():
    section_title("Missing Values Analysis", "🔍")
    st.write("Understanding *where* and *how much* data is missing in the raw dataset.")

    missing_count = raw_df.isnull().sum()
    missing_pct = (missing_count / len(raw_df) * 100).round(2)
    missing_table = pd.DataFrame({
        "Column": missing_count.index,
        "Missing Values": missing_count.values,
        "Missing %": missing_pct.values,
    }).sort_values("Missing Values", ascending=False).reset_index(drop=True)

    metric_card_row([
        ("Total Missing Cells", f'{int(missing_count.sum()):,}'),
        ("Columns With Missing Data", f'{(missing_count > 0).sum()}'),
        ("Worst Column", missing_table.iloc[0]["Column"]),
        ("Worst Column %", f'{missing_table.iloc[0]["Missing %"]}%'),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Table", "📊 Bar Chart", "🌡️ Heatmap"])

    with tab1:
        st.dataframe(
            missing_table.style.background_gradient(subset=["Missing %"], cmap="Reds"),
            use_container_width=True, hide_index=True,
        )

    with tab2:
        fig = px.bar(
            missing_table[missing_table["Missing Values"] > 0],
            x="Column", y="Missing Values", color="Missing %",
            color_continuous_scale="Reds", template=PLOTLY_TEMPLATE,
            title="Missing Values by Column",
        )
        fig.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.heatmap(raw_df.isnull(), cbar=False, cmap="Reds", yticklabels=False, ax=ax)
        ax.set_title("Missing Value Heatmap (raw dataset)", color="#FAFAFA")
        st.pyplot(fig, use_container_width=True)
        st.caption("Each red streak marks a missing cell — vertical bands reveal which columns are most affected.")


# --------------------------------------------------------------------------- #
# PAGE: Data Cleaning Steps
# --------------------------------------------------------------------------- #
def page_cleaning_steps():
    section_title("Data Cleaning Steps", "🧹")
    st.write(
        "Every transformation applied to the dataset, in order — with the "
        "problem, the explanation, the exact code used, and the measurable result."
    )

    total_steps = len(step_results)
    progress = st.progress(0, text="Cleaning pipeline progress")
    for i in range(total_steps):
        progress.progress((i + 1) / total_steps, text=f"Step {i+1} of {total_steps} complete")
        time.sleep(0.01)

    st.markdown("<br>", unsafe_allow_html=True)

    for i, step in enumerate(step_results, start=1):
        rows_delta = step["before"]["rows"] - step["after"]["rows"]
        missing_delta = step["before"]["missing"] - step["after"]["missing"]

        with st.expander(f"**{step['name']}**", expanded=(i == 1)):
            st.markdown(f"**Explanation:** {step['explanation']}")
            st.markdown("**Python Code Used:**")
            st.code(step["code"], language="python")

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows Removed", f"{rows_delta:,}")
            c2.metric("Missing Cells Change", f"{missing_delta:+,}".replace("+-", "-"))
            c3.metric("Rows Remaining", f'{step["after"]["rows"]:,}')

            st.success(f"✅ {step['note']}")

    st.balloons_shown = True
    st.success(f"🎉 All {total_steps} cleaning steps completed successfully!")


# --------------------------------------------------------------------------- #
# PAGE: Before vs After Comparison
# --------------------------------------------------------------------------- #
def page_before_after():
    section_title("Before vs After Comparison", "⚖️")
    st.write("A side-by-side look at how the dataset improved through the cleaning pipeline.")

    before = overall["before"]
    after = overall["after"]

    compare_df = pd.DataFrame({
        "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
        "Before Cleaning": [before["rows"], before["columns"], before["missing"], before["duplicates"]],
        "After Cleaning": [after["rows"], after["columns"], after["missing"], after["duplicates"]],
    })
    compare_df["Change"] = compare_df["After Cleaning"] - compare_df["Before Cleaning"]

    st.dataframe(
        compare_df.style.background_gradient(subset=["Change"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    card_cols = st.columns(4)
    metrics_display = [
        ("Rows", before["rows"], after["rows"]),
        ("Columns", before["columns"], after["columns"]),
        ("Missing Values", before["missing"], after["missing"]),
        ("Duplicate Rows", before["duplicates"], after["duplicates"]),
    ]
    for col, (label, b, a) in zip(card_cols, metrics_display):
        with col:
            st.metric(label, f"{a:,}", delta=f"{a - b:,}", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📊 Grouped Bar Chart", "📉 Missing Values Trend"])

    with tab1:
        melted = compare_df.melt(id_vars="Metric",
                                  value_vars=["Before Cleaning", "After Cleaning"],
                                  var_name="Stage", value_name="Value")
        fig = px.bar(melted, x="Metric", y="Value", color="Stage", barmode="group",
                     template=PLOTLY_TEMPLATE, text="Value",
                     color_discrete_map={"Before Cleaning": "#6b7280", "After Cleaning": NETFLIX_RED},
                     title="Dataset Metrics: Before vs After Cleaning")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        trend_df = pd.DataFrame({
            "Step": ["Raw"] + [s["name"].split("—")[0].strip() for s in step_results],
            "Missing Values": [before["missing"]] + [s["after"]["missing"] for s in step_results],
        })
        fig = px.line(trend_df, x="Step", y="Missing Values", markers=True,
                      template=PLOTLY_TEMPLATE, title="Missing Values Across the Cleaning Pipeline")
        fig.update_traces(line_color=NETFLIX_RED)
        fig.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE: Final Dataset
# --------------------------------------------------------------------------- #
def page_final_dataset():
    section_title("Final Cleaned Dataset", "✅")
    st.write("Explore the fully cleaned, analysis-ready dataset — search, filter, and sort freely.")

    metric_card_row([
        ("Rows", f"{cleaned_df.shape[0]:,}"),
        ("Columns", f"{cleaned_df.shape[1]:,}"),
        ("Missing Values", f'{int(cleaned_df.isnull().sum().sum()):,}'),
        ("Duplicate Rows", f'{int(cleaned_df.duplicated().sum()):,}'),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        search_term = st.text_input("🔎 Search (Title / Director / Cast)", "")
    with c2:
        genre_options = ["All"] + sorted(cleaned_df["Genre"].dropna().unique().tolist())
        genre_filter = st.selectbox("🎭 Filter by Genre", genre_options)
    with c3:
        country_options = ["All"] + sorted(cleaned_df["Country"].dropna().unique().tolist())
        country_filter = st.selectbox("🌍 Filter by Country", country_options)

    c4, c5 = st.columns([2, 2])
    with c4:
        sort_col = st.selectbox("↕️ Sort by column", cleaned_df.columns.tolist(),
                                 index=cleaned_df.columns.get_loc("IMDb_Rating")
                                 if "IMDb_Rating" in cleaned_df.columns else 0)
    with c5:
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

    filtered = cleaned_df.copy()
    if search_term:
        mask = pd.Series(False, index=filtered.index)
        for col in ["Title", "Director", "Cast"]:
            if col in filtered.columns:
                mask |= filtered[col].astype(str).str.contains(search_term, case=False, na=False)
        filtered = filtered[mask]
    if genre_filter != "All":
        filtered = filtered[filtered["Genre"] == genre_filter]
    if country_filter != "All":
        filtered = filtered[filtered["Country"] == country_filter]

    filtered = filtered.sort_values(sort_col, ascending=(sort_order == "Ascending"))

    st.caption(f"Showing **{len(filtered):,}** of **{len(cleaned_df):,}** rows")
    st.dataframe(filtered, use_container_width=True, height=480)


# --------------------------------------------------------------------------- #
# PAGE: Download Cleaned Dataset
# --------------------------------------------------------------------------- #
def page_download():
    section_title("Download Cleaned Dataset", "⬇️")
    st.write("Export the fully cleaned dataset as a CSV file for your own analysis or projects.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Preview of file to be downloaded:**")
        st.dataframe(cleaned_df.head(15), use_container_width=True)

    with col2:
        st.markdown("**File Summary**")
        metric_card_row([
            ("Rows", f"{cleaned_df.shape[0]:,}"),
            ("Columns", f"{cleaned_df.shape[1]:,}"),
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = df_to_csv_bytes(cleaned_df)
        st.download_button(
            label="⬇️ Download cleaned_netflix_data.csv",
            data=csv_bytes,
            file_name="cleaned_netflix_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.success("✅ Dataset is clean, validated, and ready for download.")
        st.caption(f"File size: {len(csv_bytes) / 1024:.1f} KB")


# --------------------------------------------------------------------------- #
# PAGE: About Project
# --------------------------------------------------------------------------- #
def page_about():
    section_title("About This Project", "ℹ️")

    st.markdown(
        """
This dashboard turns a Jupyter Notebook data-cleaning exercise into a fully
interactive **Streamlit web application** — a portfolio-ready demonstration
of real-world data wrangling skills.

**What was cleaned:**
- Duplicate Movie IDs and near-duplicate rows
- Messy titles (whitespace, casing, punctuation, overly long strings)
- Inconsistent genre formatting and multiple separators
- Unrealistic release years, durations, IMDb ratings
- Numbers stored as text (commas, `$` signs)
- Four different date formats
- Country and language name variants
- Missing values, handled column-by-column with the right strategy
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        section_title("Tech Stack", "🛠️")
        for t in ["Python", "Pandas", "NumPy", "Streamlit", "Matplotlib", "Seaborn", "Plotly"]:
            st.markdown(f"- {t}")
    with c2:
        section_title("Project Links", "🔗")
        st.markdown(
            """
- 📓 Original notebook: `netflix_cleaning.ipynb`
- 💻 Source code: `app.py` + `utils/cleaning.py`
- 📊 Dashboard: this Streamlit app
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="step-card" style="text-align:center;">
            <h3 style="margin-bottom:.2rem;">👨‍💻 Yash Geed</h3>
            <p style="opacity:.85; margin:0;">Artificial Intelligence &amp; Data Science Student</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
PAGES = {
    "Home": page_home,
    "Dataset Overview": page_dataset_overview,
    "Missing Values Analysis": page_missing_values,
    "Data Cleaning Steps": page_cleaning_steps,
    "Before vs After Comparison": page_before_after,
    "Final Dataset": page_final_dataset,
    "Download Cleaned Dataset": page_download,
    "About Project": page_about,
}

PAGES[page]()
footer()
