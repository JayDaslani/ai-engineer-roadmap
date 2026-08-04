import streamlit as st
import os
from knowledge_base import (
    get_or_create_vectorstore,
    add_new_note,
    ask_knowledge_base,
    generate_quiz,
    load_all_notes
)

st.set_page_config(
    page_title="Personal Knowledge Base",
    page_icon = "🧠",
    layout="wide"
)

st.title("🧠 Personal AI Knowledge Base")
st.write("Upload your personal notes (PDF & TXT) and let the AI assistant generate insights and a quiz!")


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_or_create_vectorstore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = None

with st.sidebar:
    st.header("📁 Add New Notes")

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        accept_multiple_files=False
    )

    if uploaded_file:
        if st.button("➕ Add to Knowledge Base"):
            with st.spinner("Processing & Indexing Note..."):
                msg, updated_vs = add_new_note(uploaded_file, st.session_state.vectorstore)
                st.session_state.vectorstore = updated_vs
                st.success(msg)
                st.rerun()

    st.markdown("---")

    st.subheader("📚 Saved Notes:")
    notes_dir = "Langchain/Part_C_projects/project_3_knowledge_base/notes"
    if os.path.exists(notes_dir):
        files = os.listdir(notes_dir)
        valid_files = [f for f in files if f.endswith('.pdf') or f.endswith('.txt')]
        if valid_files:
            for f in valid_files:
                st.write(f"📄 {f}")
        else:
            st.info("Koi notes saved nahi hain.")
    else:
        st.info("Notes directory khali hai.")

# -------------------------------------------------------------
# Main Tabs UI
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "ℹ️ Knowledge Base Info",
    "💬 Q&A Chat",
    "📝 Quiz Generator"
])

with tab1:
    st.header("ℹ️ Knowledge Base Overview")

    if st.session_state.vectorstore is None:
        st.warning("⚠️ Your knowledge base is currently empty! before sidebar add '.pdf' and '.txt' file.")
    else:
        st.success("✅ Knowledge Base Active aur ready on Persistence Mode")
        st.write("Your saved notes are safely stored on the persistent disk (./chroma_db). Your data will remain safe even if the app restarts.")

        if st.button("🔄 Sync & Reload All Notes"):
            with st.spinner("Knowledge base is reloading ..."):
                st.session_state.vectorstore = get_or_create_vectorstore(force_reload=False)
                st.success("Knowledge Base refreshed!")
                st.rerun()

# -------------------------------------------------------------
# Tab 2: Q&A Chat
# -------------------------------------------------------------
with tab2:
    st.header("💬 Chat with your Personal Notes")
    

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            if chat['sources']:
                st.caption(f"📖 Sources: {', '.join(chat['sources'])}")

    question = st.chat_input("Ask anything about notes ...")

    

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching for the answer in your notes..."):
                answer, sources = ask_knowledge_base(
                    question,
                    st.session_state.vectorstore
                )
            st.write(answer)
            if sources:
                st.caption(f"📖 Sources: {', '.join(sources)}")
            
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })
        st.rerun()


# -------------------------------------------------------------
# Tab 3: Quiz Generator
# -------------------------------------------------------------
with tab3:
    st.header("topic wise quiz Generator")
    st.write("Generate a 5-question quiz on any specefic topic from your notes.")

    topic = st.text_input("Enter a quiz topic (e.g. Linear Regression, Neural Networks, etc.):")

    if st.button("🎯 Generate Quiz"):
        if not topic.strip():
            st.warning("Please, first type a topic.")
        elif st.session_state.vectorstore is None:
            st.error("First, upload notes to the Knowledge Base.")
        else:
            with st.spinner(f"Quiz are creating on this {topic}"):
                quiz = generate_quiz(topic, st.session_state.vectorstore)
                st.session_state.quiz_result = quiz

    if st.session_state.quiz_result:
        st.markdown("---")
        st.markdown(st.session_state.quiz_result)


