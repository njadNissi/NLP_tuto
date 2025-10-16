import os
from pathlib import Path
import streamlit as st
from pdf_qa_system import PDFQASystem
# ---------------------------
# CLI Interface
# ---------------------------
def run_cli_mode():
    # Configuration
    PDF_DIR = "knowledge"
    MAX_THREADS = 8
    pdf_files = list(Path(PDF_DIR).rglob("*.pdf"))
    file_count = len(pdf_files)
    INDEX_PATH = f"artifacts/pdf{file_count}_qa_index.pkl"

    # Initialize system
    qa_system = PDFQASystem(
        pdf_dir=PDF_DIR,
        model_name="all-MiniLM-L6-v2",
        max_threads=MAX_THREADS,
        is_web=False  # Enable CLI-specific behavior
    )

    # Load or create index
    if os.path.exists(INDEX_PATH):
        qa_system.load_index(INDEX_PATH)
    else:
        if file_count == 0:
            print(f"Error: No PDFs found in {PDF_DIR}! Add PDFs to this folder.")
            return
        qa_system.process_all_pdfs().save_index(INDEX_PATH)

    # Show PDF list
    print("\n--- Available PDFs ---")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"{i}. {os.path.relpath(pdf, PDF_DIR)}")

    # Interactive query loop
    print("\n--- Question Answering ---")
    while True:
        try:
            question = input("\nEnter your question (or 'quit' to exit): ").strip()
            if question.lower() == 'quit':
                print("Exiting...")
                break
            if not question:
                print("Please enter a question.")
                continue

            # Get and display answer
            answer = qa_system.answer_question(question)
            print("\nAnswer:", answer['answer'])
            print(f"Confidence: {answer['score']:.4f}")
            if answer['sources']:
                print("Sources:")
                for source in answer['sources']:
                    print(f"• {source}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")