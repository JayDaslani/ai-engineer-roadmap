import os
import streamlit as st
from research_agent import run_research

st.set_page_config(
    page_title="AI Smart Research Assistant",
    page_icon="🤖",
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🔍 Smart AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by LangGraph, DuckDuckGo & Groq (Llama-3.3-70b)</div>', unsafe_allow_html=True)

st.sidebar.title("📁 Reports History")
reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')

if os.path.exists(reports_dir):
    report_files = sorted(
        [f for f in os.listdir(reports_dir) if f.endswith(".md")],
        reverse=True
    )
else:
    report_files = []

selected_file = None
if report_files:
    st.sidebar.caption("Click to view previous generated reports:")
    selected_file = st.sidebar.selectbox("Select Report", ["-- Choose a Report --"] + report_files)
    
    if selected_file and selected_file != "-- Choose a Report --":
        file_path = os.path.join(reports_dir, selected_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        st.sidebar.download_button(
            label="📥 Download Selected (.md)",
            data=file_content,
            file_name=selected_file,
            mime="text/markdown",
            use_container_width=True
        )
else:
    st.sidebar.info("No saved reports found yet.")

st.subheader("🚀 Generate New Research")

with st.form('research_form'):
    topic = st.text_input(
        "Enter Research Topic:",
        placeholder="e.g., Quantitative Trading Models, AI Agents in Fintech, High Frequency Trading..."
    )
    submit_btn = st.form_submit_button("Start Research Pipeline ⚡", use_container_width=True)

if submit_btn:
    if not topic.strip():
        st.warning("⚠️ Please enter a research topic first!")
    else:
        status_box = st.status("🔄 Executing Research Graph Nodes...", expanded=True)

        try:
            status_box.write("🌐 [Node 1] Searching web via DuckDuckGo...")
            final_state = run_research(topic.strip())

            status_box.write("📝 [Node 2] Summarizing search data via LLM...")
            status_box.write("📊 [Node 3] Generating structured markdown report...")
            status_box.update(label="✅ Research Complete & Saved to Disk!", state="complete", expanded=False)

            st.success("🎉 Report generated successfully!")

            tab1, tab2, tab3 = st.tabs(["📄 Final Report", "📝 Executive Summary", "🔍 Raw Search Results"])

            with tab1:
                st.markdown(final_state['report'])
                st.download_button(
                    label="📥 Download Report (.md)",
                    data=final_state['report'],
                    file_name=f"{topic.lower().replace(' ', '_')}_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with tab2:
                st.info(final_state['summary'])

            with tab3:
                with st.expander("Click to view collected search data"):
                    st.text(final_state["search_results"])

        except Exception as e:
            status_box.update(label="❌ Error in Pipeline Execution", state="error")
            st.error(f"Execution Error: {str(e)}")


if selected_file and selected_file != "-- Choose a Report --" and not submit_btn:
    st.divider()
    st.subheader(f"📖 Previewing Saved Report: `{selected_file}`")
    file_path = os.path.join(reports_dir, selected_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        st.markdown(f.read())

