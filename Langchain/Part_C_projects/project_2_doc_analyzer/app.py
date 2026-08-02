# app.py

import streamlit as st
from analyzer import (
    load_documents,
    create_chunks,
    create_vectorstore,
    generate_summary,
    extract_key_points,
    compare_documents,
    ask_question
)

# Page config
st.set_page_config(
    page_title="Document Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Document Analyzer")
st.write("Multiple documents upload karo aur analyze karo!")

# Session state
if "documents" not in st.session_state:
    st.session_state.documents = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "key_points" not in st.session_state:
    st.session_state.key_points = None
if "comparison" not in st.session_state:
    st.session_state.comparison = None

# Sidebar
with st.sidebar:
    st.header("📁 Documents Upload")

    uploaded_files = st.file_uploader(
        "PDFs choose karo",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process Documents"):
            with st.spinner("Documents process ho rahe hain..."):

                # Load
                docs = load_documents(uploaded_files)
                st.session_state.documents = docs
                st.success(f"✅ {len(docs)} pages loaded!")

                # Chunks
                chunks = create_chunks(docs)
                st.success(f"✅ {len(chunks)} chunks!")

                # Vectorstore
                vs = create_vectorstore(chunks)
                st.session_state.vectorstore = vs

                # Reset
                st.session_state.summary = None
                st.session_state.key_points = None
                st.session_state.comparison = None
                st.session_state.chat_history = []

                st.success("✅ Ready!")

    # Files list
    if st.session_state.documents:
        st.subheader("📄 Loaded Files:")
        files = list(set([
            doc.metadata.get('source_file', 'Unknown')
            for doc in st.session_state.documents
        ]))
        for f in files:
            st.write(f"→ {f}")

    # Clear button
    if st.button("🗑️ Clear All"):
        st.session_state.documents = None
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.summary = None
        st.session_state.key_points = None
        st.session_state.comparison = None
        st.rerun()

# Main area
if st.session_state.documents is None:
    st.info("👈 Pehle PDFs upload karo aur Process karo")

else:
    # 4 Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Summary",
        "🎯 Key Points",
        "⚖️ Compare",
        "💬 Q&A Chat"
    ])

    # Tab 1 — Summary
    with tab1:
        st.header("📝 Document Summary")

        if st.button("Generate Summary"):
            with st.spinner("Thinking..."):
                summary = generate_summary(
                    st.session_state.documents
                )
                st.session_state.summary = summary

        if st.session_state.summary:
            st.write(st.session_state.summary)

    # Tab 2 — Key Points
    with tab2:
        st.header("🎯 Key Points")

        if st.button("Extract Key Points"):
            with st.spinner("Extracting Key points..."):
                key_points = extract_key_points(
                    st.session_state.documents
                )
                st.session_state.key_points = key_points

        if st.session_state.key_points:
            st.write(st.session_state.key_points)

    # Tab 3 — Compare
    with tab3:
        st.header("⚖️ Document Comparison")

        files = list(set([
            doc.metadata.get('source_file', 'Unknown')
            for doc in st.session_state.documents
        ]))

        if len(files) < 2:
            st.warning(
                "⚠️ Comparison ke liye "
                "2+ files upload karo!"
            )
        else:
            st.info(f"Comparing {len(files)} documents:")
            for f in files:
                st.write(f"→ {f}")

            if st.button("Compare Documents"):
                with st.spinner("Compare ho raha hai..."):
                    comparison = compare_documents(
                        st.session_state.documents
                    )
                    st.session_state.comparison = comparison

            if st.session_state.comparison:
                st.write(st.session_state.comparison)

    # Tab 4 — Q&A Chat
    with tab4:
        st.header("💬 Q&A Chat")

        # Chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                if chat["sources"]:
                    st.caption(
                        f"📄 Sources: "
                        f"{', '.join(chat['sources'])}"
                    )

        # Input
        question = st.chat_input(
            "Documents ke baare mein poocho..."
        )

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Soch raha hoon..."):
                    answer, sources = ask_question(
                        question,
                        st.session_state.vectorstore
                    )
                st.write(answer)
                if sources:
                    st.caption(
                        f"📄 Sources: "
                        f"{', '.join(sources)}"
                    )

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })