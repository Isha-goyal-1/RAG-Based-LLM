# Rag_based_LLM

# RAG-Based LLM Semantic Legal Search

A Retrieval-Augmented Generation (RAG) application for semantic legal document search with AI-powered explanations
Built using Streamlit, Ollama, ChromaDB, Cross-Encoder reranking, and LLaMA 2 LLM.

---

## Overview

This project implements a semantic legal document search system featuring:

- **Ollama Embeddings:** Uses Ollama's `"nomic-embed-text"` model to convert legal text into semantic vectors.
- **ChromaDB ANN Search:** Employs ChromaDB for fast approximate nearest neighbor retrieval with optimized HNSW indexing.
- **Cross-Encoder Reranking:** Applies a dense cross-encoder (`ms-marco-MiniLM-L-6-v2`) for refining initial results.
- **LLaMA 2 LLM Explanation Generation:** Calls Ollama's LLM (`llama2:latest`) to generate concise, human-readable explanations for each search result on demand.
- **Streamlit UI:** Interactive, clean interface with query input, configurable top-k results, rerank scoring, and explanation display.

---

##  Features

- **Efficient Embedding & Index Caching:** Uses Streamlit's `@st.cache_resource` decorator to avoid repeated embedding computation and index rebuilding across sessions.
- **Batch Processing with Progress Bar:** Shows real-time progress during initial embedding and indexing.
- **Configurable Top-K Results:** Adjustable slider to control number of displayed and reranked results.
- **On-Demand Explanation Generation:** Explanations by LLM are generated only when users request them, ensuring responsiveness.
- **Streamlined UI with Styling:** Displays cosine distances, rerank scores, legal metadata (`act` and `section`), documents excerpts, and AI-generated explanations in a visually appealing manner.

---

##  Use Cases

- Legal researchers and practitioners seeking semantically relevant excerpts from legal text.
- Developers exploring how to combine semantic search, neural reranking, and LLM explanation generation.
- Organizations wanting explainable AI-powered legal search to enhance workflows.

---

## ⚙️ How It Works

1. **Data Loading:** Reads legal documents from `bns.csv` (with `description`, `act`, and `section` columns).
2. **Embedding Generation:** Uses Ollama’s embedding model to convert documents into vector representations.
3. **Indexing:** Builds an ANN index in ChromaDB optimized with HNSW parameters.
4. **Query Processing:** Retrieves candidate documents based on vector similarity.
5. **Reranking:** Applies a Cross-Encoder model to reorder candidate results by refined relevance score.
6. **Explanation Generation:** On user request, generates a concise relevance explanation per result using Ollama’s LLaMA 2 model.
7. **Display:** Presents results with metadata, similarity scores, reranking, and explanations.

---

##  Requirements

- Python 3.8 or higher
- Python packages:
  - `streamlit`
  - `pandas`
  - `chromadb`
  - `sentence-transformers`
  - `requests`
  - `numpy`
- Ollama server running locally or accessible remotely with:
  - `"nomic-embed-text"` embedding model available
  - `"llama2:latest"` LLM model available
- A CSV file (`bns.csv`) containing legal documents with the necessary columns.

---

##  Installation & Setup

1. Clone this repository:
    ```
    git clone https://github.com/dhaneshdalal2/Rag_based_LLM.git
    cd Rag_based_LLM
    ```

2. Install required Python packages:
    ```
    pip install streamlit pandas chromadb sentence-transformers requests numpy
    ```

3. Ensure Ollama server is running at the configured API URL (`http://localhost:11434/api` by default) with the required models loaded.

4. Place your `bns.csv` file in the root directory. The file should contain legal document descriptions and metadata (`act`, `section`, `description`).

---

##  Running the App

Launch the Streamlit app with:

streamlit run RAG_Based_LLM.py


- Enter your legal query in the input area.
- Adjust the "Number of top results" slider.
- Click "Run Semantic Search" to retrieve and rerank relevant legal documents.
- Click "Show Explanation" on any result to generate a concise AI explanation of relevance.

---

## Project Structure

- `RAG_Based_LLM.py`: Main Streamlit application implementing the full search pipeline.
- `bns.csv`: Legal documents dataset (user-provided).
- `logo.jpg`: Optional sidebar logo image.
- `README.md`: Documentation and usage instructions.

---

## Configuration

Adjustable parameters within `RAG_Based_LLM.py`:

- **OLLAMA_EMBED_MODEL:** Ollama embedding model name (default: `"nomic-embed-text"`).
- **OLLAMA_LLM_MODEL:** Ollama LLM model for explanations (default: `"llama2:latest"`).
- **OLLAMA_API_URL:** Base URL for Ollama API calls.
- **CHROMA_COLLECTION_NAME:** ChromaDB collection name.
- **BATCH_SIZE:** Number of documents processed per batch during indexing.
- **HNSW parameters:** Settings for efficient ANN index construction and search.
- **Top-K slider:** User-controlled display and rerank result count.

---

## Error Handling & Reliability

- Cached loading of embedding function and reranker models with Streamlit caching.
- User-friendly error messages for missing CSV files or Ollama API connectivity issues.
- Explanation generation with 60-second timeout and error fallback messages.
- Progress indicators and status text for batch embedding.

---

## Acknowledgements

This project leverages:

- [Ollama](https://ollama.com) for embeddings & language models
- [ChromaDB](https://chroma.com) for vector similarity search
- [Sentence Transformers](https://www.sbert.net) cross-encoder models for reranking
- [Streamlit](https://streamlit.io) for creating the web app UI

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contribution

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## Contact

For questions or support, please contact ishagoyal2040@gmail.com

---

Enjoy exploring and extending your legal search with AI-powered explanations!!






