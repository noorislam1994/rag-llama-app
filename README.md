
# Llama-RAG-Project

  

**Retrieval-Augmented Generation (RAG) Application** using:

  

-  **Lightweight Llama 3.2–3B** (licensed from Meta)

-  **Qdrant** for vector storage (requires free account & API key)

-  **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)

- A custom Flask API and chat-like UI

  

---


## Project Structure

  

```plaintext

rag-project/

├── app/

│ ├── __init__.py

│ ├── main.py # The main Flask application entry point

│ ├── templates/

│ │ └── index.html # The front-end HTML UI

│ ├── utils/

│ │ ├── ingestion.py # PDF ingestion, chunking, embedding, upsert to Qdrant

│ │ ├── retrieval.py # Reranking, expanded context filtering, etc.

│ │ └── __init__.py

├── RAG-Application.ipynb # Jupyter Notebook demonstrating ingestion/retrieval logic

├── requirements.txt # Python dependencies

├── .gitignore

└── README.md # (this file)
```

## Architecture Overview  
- **Ingestion:** PDF → Chunking → Embedding → Qdrant  
- **Retrieval:** User query → Embedding → Qdrant → Re-ranker → Expanded context  
- **Prompt:** Llama uses the final curated context + user query  
- **Memory:** Session-based chat  

![RAG Architecture](.Architecture.jpeg)


## Key Directories/Files

- **`app/main.py`**  
  Your main Flask server code. Defines endpoints for `/upload` (ingestion) and `/query` (retrieval).

- **`app/templates/index.html`**  
  The front-end interface for PDF upload and chat interaction.

- **`app/utils/ingestion.py`**  
  Loads PDFs (via `PyPDFLoader`), splits text, embeds chunks (using `HuggingFaceEmbeddings`), and upserts vectors into Qdrant.

- **`app/utils/retrieval.py`**  
  Performs cross-encoder re-ranking and expanded context logic after the initial Qdrant retrieval.

- **`RAG-Application.ipynb`**  
  A prototyping notebook that demonstrates ingestion, query embedding, and retrieval logic before finalizing in `main.py`.

- **`requirements.txt`**  
  Contains your Python libraries (Flask, Qdrant client, Transformers, etc.).

---

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/noorislam1994/rag-project.git
cd rag-project
```
### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### Obtain a Qdrant Account & API Key

1. Sign up for a free Qdrant account at [qdrant.tech](https://qdrant.tech/).  
2. Retrieve your **API key** and set it in your code or environment variables (where the code references `qdrant_client = QdrantClient(...)`).
3. If hosting Qdrant locally, ensure it’s running and accessible at the specified **URL**.

### Get the LLaMA Model License & Download

1. **Request a license** from Meta for the **Llama 3.2–3B-Instruct** model (or whichever Llama variant you prefer).
2. **Once approved**, you’ll receive access to the model or a download URL.
3. **Update** the `model_name` or local path in `main.py` to point to your downloaded Llama weights (e.g., `"meta-llama/Llama-3.2-3B-Instruct"`).

## Run the Flask App

```bash
cd app
python main.py
```
*The server typically runs on [http://127.0.0.1:5000](http://127.0.0.1:5000) by default; you can change host/port if needed.*

## Open the UI
Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.  
Upload a PDF, then ask questions in the chat for retrieval-augmented answers.  

## Usage
**Upload PDFs from the front-end.** The app:  
- Creates or updates the Qdrant collection.  
- Splits and embeds the PDF content.  
- Stores vectors in Qdrant with metadata.  

**Query in chat:**  
- Embeds your question using HuggingFaceEmbeddings (MiniLM).  
- Fetches top results from Qdrant.  
- Re-ranks them with the cross-encoder.  
- Potentially expands context with a similarity threshold.  
- Sends the final context to LLaMA for an answer.  
- Maintains conversation history by session ID.  

## Contributing
Pull requests are welcome for improvements:  

- Swapping out the LLaMA model for another open-source model  
- Enhancing the front-end UI or chat memory logic  
- Adding an alternative vector store or advanced re-ranking strategy  


