"""

"""

import os
from pathlib import Path
import re
import PyPDF2
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import faiss
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed  # THREADING: Import thread tools


class PDFQASystem:
    def __init__(self, pdf_dir, 
                 model_name="all-MiniLM-L6-v2", 
                 qa_model_name="distilbert-base-uncased-distilled-squad",
                 max_threads=4):  # THREADING: Add thread count config
        self.pdf_dir = pdf_dir
        self.model = self._load_sentence_model(model_name)
        self.qa_pipeline = self._load_qa_model(qa_model_name)
        self.index = None
        self.text_chunks = []  # Stores (chunk, full_path, relative_path)
        self.embeddings = None
        self.max_threads = max_threads  # THREADING: Save max concurrent threads


    def _load_sentence_model(self, model_name):
        try:
            full_model_name = f"sentence-transformers/{model_name}"
            print(f"Loading sentence transformer model: {full_model_name}")
            return SentenceTransformer(full_model_name)
        except Exception as e:
            print(f"Could not load {full_model_name}: {str(e)}")
            print(f"Trying to load base model: {model_name}")
            try:
                return SentenceTransformer(model_name)
            except Exception as e:
                print(f"Could not load {model_name}: {str(e)}")
                print("Falling back to default model")
                return SentenceTransformer("all-MiniLM-L6-v2")


    def _load_qa_model(self, model_name):
        try:
            print(f"Loading QA model: {model_name}")
            return pipeline("question-answering", model=model_name)
        except Exception as e:
            print(f"Could not load {model_name}: {str(e)}")
            print("Falling back to default QA model")
            return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")


    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Skip empty pages
                        text += page_text
                # Clean text: remove extra spaces/newlines and strip whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""


    def split_text_into_chunks(self, text, chunk_size=500, chunk_overlap=50):
        chunks = []
        if not text:
            return chunks
        
        # Adjust chunk size/overlap for short texts to avoid empty chunks
        effective_chunk_size = min(chunk_size, len(text))
        effective_overlap = min(chunk_overlap, effective_chunk_size // 2)
        
        # Generate chunks
        for i in range(0, len(text), effective_chunk_size - effective_overlap):
            chunk = text[i:i + effective_chunk_size].strip()
            if chunk:  # Skip empty chunks
                chunks.append(chunk)
        return chunks


    # THREADING: Helper method to process a single PDF (thread-safe)
    def _process_single_pdf(self, pdf_path):
        """Process one PDF: extract text → split into chunks. Returns valid chunks with sources."""
        rel_path = os.path.relpath(pdf_path, self.pdf_dir)
        text = self.extract_text_from_pdf(pdf_path)
        
        # Skip PDFs with no text
        if not text:
            return None, f"Warning: No text extracted from {rel_path} — skipping."
        
        # Split into chunks
        chunks = self.split_text_into_chunks(text)
        if not chunks:
            return None, f"Warning: No valid chunks from {rel_path} — skipping."
        
        # Return chunks with source info (ready to add to all_chunks)
        chunks_with_source = [(chunk, pdf_path, rel_path) for chunk in chunks]
        return chunks_with_source, f"Processed {rel_path} → {len(chunks)} chunks"


    def process_all_pdfs(self):
        print(f"Processing PDFs in {self.pdf_dir} (including subfolders) with {self.max_threads} threads...")
        
        # Get all PDF files recursively
        pdf_files = list(Path(self.pdf_dir).rglob("*.pdf"))
        pdf_files = [str(path) for path in pdf_files]
        
        if not pdf_files:
            print(f"Warning: No PDF files found in {self.pdf_dir} or its subfolders.")
            return self
        
        # THREADING: Use ThreadPoolExecutor to process PDFs in parallel
        all_chunks = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Map each PDF to a thread task
            future_to_pdf = {executor.submit(self._process_single_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
            
            # Track progress with tqdm (show 100% when all threads finish)
            for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc="Processing PDFs (parallel)"):
                pdf_path = future_to_pdf[future]
                try:
                    # Get results from the thread
                    chunks, message = future.result()
                    print(f"\n{message}")  # Print status (e.g., "Processed file.pdf → 10 chunks")
                    if chunks:
                        all_chunks.extend(chunks)
                except Exception as e:
                    print(f"\nError processing {pdf_path} in thread: {str(e)}")

        # Final filter: Keep only valid chunks (safety check)
        self.text_chunks = [entry for entry in all_chunks if isinstance(entry[0], str) and entry[0].strip()]
        
        if not self.text_chunks:
            print("Warning: No valid text chunks found across all PDFs. Embeddings cannot be created.")
            return self
        
        print(f"\nProcessed {len(pdf_files)} PDFs into {len(self.text_chunks)} valid text chunks (parallel processing done)")
        print("Creating embeddings (this may take a while)...") 
        
        # Create embeddings (CPU-bound, no threading needed here)
        print("Creating embeddings...")
        chunk_texts = [chunk[0] for chunk in self.text_chunks]
        self.embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        
        # Create FAISS index
        self._create_index()
        return self


    def _create_index(self):
        if self.embeddings is None or len(self.embeddings) == 0:
            print("Error: Cannot create FAISS index — no embeddings available.")
            return self
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        return self


    def save_index(self, index_path="artifacts/pdf_qa_index.pkl"):
        if not self.index or not self.text_chunks or self.embeddings is None:
            print("Error: Cannot save index — no valid data (index/chunks/embeddings missing).")
            return self
        data = {
            'index': self.index,
            'text_chunks': self.text_chunks,
            'embeddings': self.embeddings,
            'root_dir': self.pdf_dir,
            'max_threads': self.max_threads  # Save thread config for later
        }
        with open(index_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Index saved to {index_path}")
        return self


    def load_index(self, index_path="artifacts/pdf_qa_index.pkl"):
        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
            self.index = data['index']
            self.text_chunks = data['text_chunks']
            self.embeddings = data['embeddings']
            self.pdf_dir = data.get('root_dir', self.pdf_dir)
            self.max_threads = data.get('max_threads', 4)  # Restore thread config
            
            # Validate loaded chunks
            self.text_chunks = [entry for entry in self.text_chunks if isinstance(entry[0], str) and entry[0].strip()]
            print(f"Index loaded from {index_path} (found {len(self.text_chunks)} valid chunks)")
            return self
        except Exception as e:
            print(f"Error loading index from {index_path}: {str(e)}")
            return self


    def find_relevant_chunks(self, query, top_k=5):
        if not self.index or not self.text_chunks or self.embeddings is None:
            print("Error: Cannot find relevant chunks — index/data not ready.")
            return []
        
        query = str(query).strip()
        if not query:
            print("Error: Empty query — cannot search.")
            return []
        
        query_embedding = self.model.encode([query])
        top_k = min(top_k, len(self.text_chunks))  # Avoid index errors
        distances, indices = self.index.search(query_embedding, top_k)
        
        relevant_chunks = []
        for i in range(top_k):
            idx = indices[0][i]
            if 0 <= idx < len(self.text_chunks):
                relevant_chunks.append({
                    'chunk': self.text_chunks[idx][0],
                    'full_path': self.text_chunks[idx][1],
                    'relative_path': self.text_chunks[idx][2],
                    'distance': distances[0][i]
                })
        return relevant_chunks


    def answer_question(self, query, top_k=5):
        relevant_chunks = self.find_relevant_chunks(query, top_k)
        if not relevant_chunks:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'score': 0.0,
                'sources': [],
                'full_sources': []
            }
        
        # Combine chunks into context (truncate to avoid token overflow)
        context = "\n\n".join([chunk['chunk'] for chunk in relevant_chunks])[:2000].strip()
        
        try:
            result = self.qa_pipeline(question=str(query).strip(), context=context)
            # Remove duplicate sources
            unique_sources = []
            unique_full_sources = []
            seen = set()
            for chunk in relevant_chunks:
                if chunk['relative_path'] not in seen:
                    seen.add(chunk['relative_path'])
                    unique_sources.append(chunk['relative_path'])
                    unique_full_sources.append(chunk['full_path'])
            
            result['sources'] = unique_sources
            result['full_sources'] = unique_full_sources
            return result
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return {
                'answer': "I encountered an error while generating an answer.",
                'score': 0.0,
                'sources': [],
                'full_sources': []
            }


def main():
    # Configuration
    PDF_DIR = "knowledge"  # Update to your PDF folder
    MAX_THREADS = 12  # THREADING: Adjust based on your CPU (4–8 works for most systems)
    
    # Count total PDFs
    pdf_files = list(Path(PDF_DIR).rglob("*.pdf"))
    file_count = len(pdf_files)
    INDEX_PATH = f"artifacts/pdf{file_count}_qa_index.pkl"
    
    print(f"Found {file_count} PDF file(s) in {PDF_DIR} and subfolders.")
    
    # Initialize system with threading
    MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L3-v2"]
    qa_system = PDFQASystem(
        pdf_dir=PDF_DIR,
        model_name=MODELS[0],
        max_threads=MAX_THREADS  # Pass thread count to the system
    )
    
    # Load or create index
    if os.path.exists(INDEX_PATH):
        qa_system.load_index(INDEX_PATH)
    else:
        if file_count == 0:
            print("Error: No PDFs found — cannot create index.")
            return
        qa_system.process_all_pdfs().save_index(INDEX_PATH)
    
    # Interactive loop
    print("\nPDF Question Answering System")
    print("------------------------------")
    print(f"Ready to answer questions about {len(qa_system.text_chunks)} chunks from {file_count} PDFs.")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nAsk a question: ")
        if question.lower().strip() == 'quit':
            break
        if not question.strip():
            print("Please enter a valid question.")
            continue
        
        answer = qa_system.answer_question(question)
        print(f"\nAnswer: {answer['answer']}")
        print(f"Confidence: {answer['score']:.4f}")
        if answer['sources']:
            print(f"Sources: {', '.join(answer['sources'])}")


if __name__ == "__main__":
    main()