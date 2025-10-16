import os
import sys
from pathlib import Path
import re
import PyPDF2
from transformers import pipeline
import faiss
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_utils import get_cached_model
# ---------------------------
# Shared PDF QA System Core Logic
# ---------------------------
class PDFQASystem:
    def __init__(self, pdf_dir, 
                 model_name="all-MiniLM-L6-v2", 
                 qa_model_name="distilbert-base-uncased-distilled-squad",
                 max_threads=4,
                 is_web=False):  # New flag to adjust output for web/CLI
        self.pdf_dir = pdf_dir
        self.model = get_cached_model(model_name)
        self.qa_pipeline = self._load_qa_model(qa_model_name, is_web)
        self.index = None
        self.text_chunks = []  # (chunk, full_path, relative_path)
        self.embeddings = None
        self.max_threads = max_threads
        self.pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
        self.is_web = is_web  # Track mode for output formatting


    def _load_qa_model(self, model_name, is_web):
        """Load QA model with appropriate feedback (web/CLI)"""
        msg_func = self._web_msg if is_web else print
        try:
            msg_func(f"Loading QA model: {model_name}")
            return pipeline("question-answering", model=model_name)
        except Exception as e:
            msg_func(f"Could not load {model_name}: {str(e)}")
            msg_func("Falling back to default QA model")
            return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")


    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with error handling"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            msg = f"Error extracting text from {pdf_path}: {str(e)}"
            self._web_msg(msg) if self.is_web else print(msg)
            return ""


    def split_text_into_chunks(self, text, chunk_size=500, chunk_overlap=50):
        """Split text into overlapping chunks"""
        chunks = []
        if not text:
            return chunks
        
        effective_chunk_size = min(chunk_size, len(text))
        effective_overlap = min(chunk_overlap, effective_chunk_size // 2)
        
        for i in range(0, len(text), effective_chunk_size - effective_overlap):
            chunk = text[i:i + effective_chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks


    def _process_single_pdf(self, pdf_path):
        """Process individual PDF (used in parallel)"""
        rel_path = os.path.relpath(pdf_path, self.pdf_dir)
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            return None, f"Warning: No text extracted from {rel_path} — skipping."
        
        chunks = self.split_text_into_chunks(text)
        if not chunks:
            return None, f"Warning: No valid chunks from {rel_path} — skipping."
        
        return [(chunk, pdf_path, rel_path) for chunk in chunks], f"Processed {rel_path} → {len(chunks)} chunks"


    def process_all_pdfs(self):
        """Process all PDFs with parallel threads"""
        msg_func = self._web_msg if self.is_web else print
        msg_func(f"Processing PDFs in {self.pdf_dir} with {self.max_threads} threads...")
        
        pdf_files = [str(path) for path in self.pdf_files]
        if not pdf_files:
            msg_func(f"No PDF files found in {self.pdf_dir}")
            return self
        
        all_chunks = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_pdf = {executor.submit(self._process_single_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
            
            # Show progress bar in web mode, simple counter in CLI
            if self.is_web:
                import streamlit as st
                progress_bar = st.progress(0)
            
            for i, future in enumerate(as_completed(future_to_pdf)):
                if self.is_web:
                    progress_bar.progress((i + 1) / len(pdf_files))
                
                pdf_path = future_to_pdf[future]
                try:
                    chunks, message = future.result()
                    status_msg = f"[{i+1}/{len(pdf_files)}] {message}"
                    msg_func(status_msg)
                    if chunks:
                        all_chunks.extend(chunks)
                except Exception as e:
                    msg_func(f"Error processing {pdf_path}: {str(e)}")

        self.text_chunks = [entry for entry in all_chunks if isinstance(entry[0], str) and entry[0].strip()]
        
        if not self.text_chunks:
            msg_func("No valid text chunks found across all PDFs")
            return self
        
        msg_func(f"Processed {len(pdf_files)} PDFs into {len(self.text_chunks)} valid chunks")
        msg_func("Creating embeddings...")
        chunk_texts = [chunk[0] for chunk in self.text_chunks]
        self.embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        
        self._create_index()
        return self


    def _create_index(self):
        """Create FAISS index for similarity search"""
        if self.embeddings is None or len(self.embeddings) == 0:
            msg = "Cannot create index — no embeddings available"
            self._web_msg(msg) if self.is_web else print(msg)
            return self
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        return self


    def save_index(self, index_path="artifacts/pdf_qa_index.pkl"):
        """Save index to disk"""
        if not self.index or not self.text_chunks or self.embeddings is None:
            msg = "Cannot save index — missing data"
            self._web_msg(msg) if self.is_web else print(msg)
            return self
        Path("artifacts").mkdir(exist_ok=True)
        with open(index_path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'text_chunks': self.text_chunks,
                'embeddings': self.embeddings,
                'root_dir': self.pdf_dir,
                'max_threads': self.max_threads
            }, f)
        msg = f"Index saved to {index_path}"
        self._web_msg(msg) if self.is_web else print(msg)
        return self


    def load_index(self, index_path="artifacts/pdf_qa_index.pkl"):
        """Load previously saved index"""
        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
            self.index = data['index']
            self.text_chunks = data['text_chunks']
            self.embeddings = data['embeddings']
            self.pdf_dir = data.get('root_dir', self.pdf_dir)
            self.max_threads = data.get('max_threads', 4)
            self.pdf_files = list(Path(self.pdf_dir).rglob("*.pdf"))
            msg = f"Loaded index with {len(self.text_chunks)} chunks"
            self._web_msg(msg) if self.is_web else print(msg)
            return self
        except Exception as e:
            msg = f"Error loading index: {str(e)}"
            self._web_msg(msg) if self.is_web else print(msg)
            return self


    def find_relevant_chunks(self, query, top_k=5):
        """Find relevant chunks for a query"""
        if not self.index or not self.text_chunks or self.embeddings is None:
            msg = "Index not ready — cannot search"
            self._web_msg(msg) if self.is_web else print(msg)
            return []
        
        query = str(query).strip()
        if not query:
            msg = "Empty query — skipping"
            self._web_msg(msg) if self.is_web else print(msg)
            return []
        
        query_embedding = self.model.encode([query])
        top_k = min(top_k, len(self.text_chunks))
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
        """Generate answer to query"""
        relevant_chunks = self.find_relevant_chunks(query, top_k)
        if not relevant_chunks:
            return {
                'answer': "No relevant information found.",
                'score': 0.0,
                'sources': []
            }
        
        context = "\n\n".join([chunk['chunk'] for chunk in relevant_chunks])[:2000].strip()
        
        try:
            result = self.qa_pipeline(question=str(query).strip(), context=context)
            
            # Get unique sources
            unique_sources = []
            seen = set()
            for chunk in relevant_chunks:
                if chunk['relative_path'] not in seen:
                    seen.add(chunk['relative_path'])
                    unique_sources.append(chunk['relative_path'])
            
            result['sources'] = unique_sources
            return result
        except Exception as e:
            msg = f"Error generating answer: {str(e)}"
            self._web_msg(msg) if self.is_web else print(msg)
            return {
                'answer': "Error generating answer.",
                'score': 0.0,
                'sources': []
            }


    def _web_msg(self, message):
        """Helper for web mode messages (uses Streamlit)"""
        import streamlit as st
        st.text(message)
