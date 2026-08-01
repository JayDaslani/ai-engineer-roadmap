import streamlit as st
import os
from pdf_qa import load_pdf, create_chunks, create_vectorstore, get_answer

st.set_page_config(
    page_title="PDF Q&A System",
    page_icon="📄",
    layout='wide'
)

st.title("📄 PDF Q&A System")
st.write("Upload a PDF and ask questions!")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("📁 PDF Upload")

    uploaded_file = st.file_uploader(
        "Choose a pdf",
        type='pdf'
    )

    if uploaded_file:
        if st.session_state.pdf_name != uploaded_file.name:

            temp_path = f"temp_{uploaded_file.name}"

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner("PDF process ho rahi hai..."):

                docs = load_pdf(temp_path)
                st.success(f"Pages : {len(docs)}")

                chunks = create_chunks(docs)
                st.success(f"Chunks : {len(chunks)}")

                vs = create_vectorstore(chunks)
                st.session_state.vectorstore = vs
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []

                os.remove(temp_path)

            st.success("✅ PDF ready!")

    if st.session_state.pdf_name:
        st.info(f"📄 {st.session_state.pdf_name}")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if st.session_state.vectorstore is None:
    st.info("👈 Pehle PDF upload karo")
else:
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['question'])

        with st.chat_message("assistant"):
            st.write(chat['answer'])
            if chat['pages']:
                st.caption(
                    f"📖 Sources: Pages "
                    f"{chat['pages']}"
                )

    question = st.chat_input("Enter your question ...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                answer, pages = get_answer(
                    question,
                    st.session_state.vectorstore
                )

            st.write(answer)

            if pages:
                st.caption(
                    f"📖 Sources: Pages {pages}"
                )
        
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "pages": pages
        })
