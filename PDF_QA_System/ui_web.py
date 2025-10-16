import os
from pathlib import Path
from pdf_qa_system import PDFQASystem

# ---------------------------
# Web Interface (Streamlit)
# ---------------------------
def run_web_mode():
    import streamlit as st
    st.set_page_config(
        layout="wide",
        page_title="PDF QA",
        page_icon="📄"
    )
    
    # Custom CSS for scrollable PDF tree
    st.markdown("""
    <style>
    .scrollable-container {
        max-height: 70vh;
        overflow-y: auto;
        padding-right: 10px;
    }
    .scrollable-container::-webkit-scrollbar { width: 8px; }
    .scrollable-container::-webkit-scrollbar-thumb { background: #ccc; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Configuration
    PDF_DIR = "knowledge"
    MAX_THREADS = 16
    pdf_files = list(Path(PDF_DIR).rglob("*.pdf"))
    file_count = len(pdf_files)
    INDEX_PATH = f"artifacts/pdf{file_count}_qa_index.pkl"

    # Initialize system in session state
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = PDFQASystem(
            pdf_dir=PDF_DIR,
            model_name="all-MiniLM-L6-v2",
            max_threads=MAX_THREADS,
            is_web=True  # Enable web-specific behavior
        )
        if os.path.exists(INDEX_PATH):
            st.session_state.qa_system.load_index(INDEX_PATH)
        else:
            if file_count == 0:
                st.error(f"No PDFs found in {PDF_DIR}! Add PDFs to this folder.")
                return
            st.session_state.qa_system.process_all_pdfs().save_index(INDEX_PATH)

    qa_system = st.session_state.qa_system

    # Layout
    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("📂 PDF Files")
        st.text(f"Found {file_count} files in: {PDF_DIR}")
        st.divider()
        
        # Scrollable PDF tree
        with st.container():
            st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
            if pdf_files:
                dir_structure = {}
                for pdf in pdf_files:
                    rel_path = os.path.relpath(pdf, PDF_DIR)
                    parts = rel_path.split(os.sep)
                    current = dir_structure
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = None

                def display_tree(node, level=0):
                    for name, child in node.items():
                        if child is None:
                            st.text("`  " * level + f"📄 {name}")
                        else:
                            st.text("`  " * level + f"📁 {name}")
                            display_tree(child, level + 1)
                display_tree(dir_structure)
            else:
                st.warning("No PDF files detected")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.header("❓ PDF Question Answering")
        st.text(f"Processed {len(qa_system.text_chunks)} chunks from {file_count} PDFs")
        st.divider()

        question = st.text_input("Ask a question:")
        if st.button("Get Answer", type="primary") and question:
            with st.spinner("Generating answer..."):
                answer = qa_system.answer_question(question)
                st.subheader("📝 Answer")
                st.write(answer['answer'])
                st.subheader("🎯 Confidence")
                st.progress(min(1.0, answer['score']))
                st.text(f"Score: {answer['score']:.4f}")
                if answer['sources']:
                    st.subheader("📚 Sources")
                    for source in answer['sources']:
                        st.text(f"• {source}")