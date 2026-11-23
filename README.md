# RAG-Powered Assistant for Machine Learning Lecture Notes

A Retrieval-Augmented Generation (RAG) assistant that answers questions from `lecturenotes.pdf` (*Lecture Notes on Machine Learning* by Miguel Á. Carreira-Perpiñán). It retrieves relevant sections from the PDF first and passes them into a language model so responses stay grounded in the lecture notes.

## Pipeline Overview

1. **Load and Split:** Loads `lecturenotes.pdf` page by page using `PyMuPDFLoader` (`pymupdf`) and splits the text into overlapping chunks (`chunk_size=600`, `chunk_overlap=100`) using `RecursiveCharacterTextSplitter`.
2. **Embed and Store:** Converts each chunk into a dense vector embedding using `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` and stores the vectors in a `FAISS` vector store.
3. **Retrieve:** Cleans minor query typos, embeds the question, and retrieves the top `k=4` matching chunks along with page numbers.
4. **Generate:** Places the retrieved context and question into a prompt template and generates a grounded response using a 4-bit quantized `HuggingFaceH4/zephyr-7b-beta` model via Hugging Face Transformers and LangChain.

## Tech Stack

- **Document Loading and Chunking:** `PyMuPDFLoader` (`pymupdf`), `RecursiveCharacterTextSplitter`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (`HuggingFaceEmbeddings`)
- **Vector Store:** `FAISS` (`faiss-cpu`)
- **LLM and Orchestration:** `LangChain`, `HuggingFaceH4/zephyr-7b-beta` (4-bit `bitsandbytes` quantization)
- **Interfaces:** CLI output in the notebook and an interactive `Gradio` web interface

## Repository Structure

```text
rag-lecture-notes-assistant/
├── README.md
├── rag_assistant.ipynb
├── lecturenotes.pdf
└── screenshots/
    ├── cli_outputs.png
    └── gradio_ui.png