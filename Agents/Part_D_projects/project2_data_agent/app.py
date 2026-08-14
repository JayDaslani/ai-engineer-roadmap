import streamlit as st
import pandas as pd
import plotly.express as px
import os
import tempfile
from data_agent import run_data_analysis

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout='wide'
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1rem;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📊 AI Data Analysis Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by LangGraph + '
    'Groq + Llama 3.3 70B</div>',
    unsafe_allow_html=True
)

st.divider()

if "result" not in st.session_state:
    st.session_state.result = None
if "df" not in st.session_state:
    st.session_state.df = None

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "📁 Upload CSV File",
        type=["csv"],
        help="Upload any CSV file for AI analysis"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("📏 Rows", df.shape[0])
        with m2:
            st.metric("📊 Columns", df.shape[1])
        with m3:
            st.metric(
                "✅ Complete",
                f"{int((1 - df.isnull().mean().mean())*100)}%"
            )
        with m4:
            num_cols_count = len(df.select_dtypes(include='number').columns)
            st.metric("🔢 Numeric Cols", num_cols_count)
        st.divider()

        if st.button(
            "🚀 Run AI Analysis",
            use_container_width=True,
            type="primary"
        ):
            with st.spinner(
                "AI Agent analyzing data..."
            ):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".csv"
                ) as tmp:
                    df.to_csv(tmp.name, index=False)
                    tmp_path=tmp.name

                result = run_data_analysis(tmp_path)
                st.session_state.result = result
                os.unlink(tmp_path)

            st.success("✅ Analysis Complete!")

if st.session_state.result:
    result = st.session_state.result
    df = st.session_state.df

    st.divider()
    st.markdown("## 📈 Analysis Results")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visualizations",
        "🔍 Analysis",
        "💡 Insights",
        "🎯 Recommendations"
    ])

    with tab1:
        st.subheader("📊 Data Visualizations")

        all_numeric = df.select_dtypes(include='number').columns.tolist()
        valuable_numeric = [c for c in all_numeric if not c.lower().endswith(('_id', 'id', 'index'))]
        if not valuable_numeric:
            valuable_numeric = all_numeric

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        c1, c2 = st.columns(2)

        # ─── Chart 1: Metric Distribution / Trend ───
        with c1:
            st.markdown("#### 📉 Metric Distribution / Histogram")
            selected_num = st.selectbox("Select Numeric Metric:", valuable_numeric, index=0, key="dist_metric")
            
            fig1 = px.histogram(
                df,
                x=selected_num,
                nbins=30,
                marginal="box",
                title=f"Distribution of {selected_num}",
                template="plotly_dark",
                color_discrete_sequence=["#00d2ff"]
            )
            st.plotly_chart(fig1, use_container_width=True)

        # ─── Chart 2: Aggregated Categorical Breakdown ───
        with c2:
            st.markdown("#### 📊 Categorical Performance (Aggregated)")
            if categorical_cols and valuable_numeric:
                
                non_date_cats = [c for c in categorical_cols if "date" not in c.lower()]
                default_cat = non_date_cats[0] if non_date_cats else categorical_cols[0]

                selected_cat = st.selectbox("Select Category (X-Axis):", categorical_cols, index=categorical_cols.index(default_cat), key="cat_col")
                selected_y = st.selectbox("Select Value (Y-Axis):", valuable_numeric, index=0, key="y_col")

                
                agg_df = df.groupby(selected_cat)[selected_y].sum().reset_index().sort_values(by=selected_y, ascending=False).head(10)

                fig2 = px.bar(
                    agg_df,
                    x=selected_cat,
                    y=selected_y,
                    title=f"Total {selected_y} by {selected_cat} (Top 10)",
                    template="plotly_dark",
                    color=selected_y,
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No categorical columns found for comparison.")

        
        if len(valuable_numeric) >= 2:
            st.markdown("#### 🔗 Correlation Heatmap (Inter-metric Relations)")
            corr_matrix = df[valuable_numeric].corr()
            fig3 = px.imshow(
                corr_matrix,
                text_auto=".2f",
                title="Correlation Heatmap",
                template="plotly_dark",
                color_continuous_scale="Blues",
                aspect="auto"
            )
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("🔍 Statistical Analysis")
        st.markdown(result["analysis"])

    with tab3:
        st.subheader("💡 Key Insights")
        st.markdown(result["insights"])

    with tab4:
        st.subheader("🎯 Strategic Recommendations")
        st.markdown(result["recommendations"])

        full_report = f"""
        # Data Analysis Report
        
        ## Analysis
        {result['analysis']}

        ## Key Insights
        {result['insights']}

        ## Recommendations
        {result['recommendations']}

        """
        st.download_button(
            "📥 Download Full Report",
            data=full_report,
            file_name="analysis_report.md",
            mime="text/markdown",
            use_container_width=True
        )



    



