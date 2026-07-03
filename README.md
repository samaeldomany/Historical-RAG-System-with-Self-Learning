# Historical RAG System with Self-Learning

## Overview
The Historical RAG System is a Context-Aware Retrieval-Augmented Generation pipeline designed to accurately answer user queries based on specific historical documents (e.g., lecture slides and PDFs regarding the Egyptian Sphinx). 

Unlike traditional RAG systems, this project features a **Self-Learning Layer** (Dynamic Knowledge Injection). If the AI generates an incorrect answer or lacks information, users can provide the correct information through the UI. This correction is immediately embedded and prioritized in the vector database, effectively teaching the model the correct answer for future queries without the computational cost of fine-tuning weights.

## Key Features
* **Context-Aware Retrieval:** Utilizes `BAAI/bge-base-en-v1.5` embeddings for high-performance semantic search against a local knowledge base.
* **Dynamic Self-Learning:** A feedback loop that converts user corrections into high-priority "ground truth" documents, patching database knowledge in real-time.
* **Explainable AI:** The user interface explicitly displays the retrieved evidence (raw text chunks and document sources) alongside the generated answer, ensuring transparency.
* **Efficient Inference:** Runs a 4-bit quantized Large Language Model (`TinyLlama-1.1B-Chat-v1.0`) via BitsAndBytes, optimized for consumer-grade hardware like Google Colab T4 GPUs.
* **Interactive Web UI:** Built with Gradio to provide a seamless, split-screen experience for querying and teaching the system.

## Tech Stack
* **Framework:** LangChain
* **LLM:** TinyLlama-1.1B-Chat-v1.0 (HuggingFace)
* **Embeddings:** BAAI/bge-base-en-v1.5
* **Vector Database:** ChromaDB
* **UI:** Gradio
* **Tools:** PyPDF, Transformers, BitsAndBytes, Accelerate

## Architecture
1. **Document Processing:** PDFs are loaded, split into chunks (500 characters, 50 overlap), and embedded into ChromaDB.
2. **Retrieval:** User queries are embedded, and the top-3 most similar document chunks are retrieved using approximate nearest neighbor search (HNSW).
3. **Generation:** Retrieved context and the user query are passed through a strict prompt template to the TinyLlama model to generate a fact-based answer.
4. **Correction Loop:** If a user submits a correction, it bypasses the standard retrieval, gets vectorized, and is immediately injected into ChromaDB to serve as the primary context for future identical queries.

## Setup & Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install langchain_community langchain_huggingface transformers sentence_transformers pypdf chromadb bitsandbytes accelerate gradio langchain_text_splitters
