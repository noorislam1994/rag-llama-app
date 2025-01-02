import logging
import os
import tempfile
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

# code refactored out:
from app.utils.ingestion import ingest_pdf
from app.utils.retrieval import (
     rerank_results,
     expanded_context_filter,
     build_conversation_str
 )


logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates")
CORS(app)

conversation_history = {}
uploaded_file_path = None

@app.before_request
def log_request():
    logging.info(f"Request: {request.method} {request.path}")

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Error occurred: {e}")
    return jsonify({"error": str(e)}), 500

# Qdrant connection
qdrant_client = QdrantClient(
    url="https://bdead1de-cfbe-449a-912d-ba4babcb20a8.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="lvQFpd67BeWEpHWCTMouCOLnASclH9R7AaXZHSHm3MHm35E8d4mZZA"
)

collection_name = "hr_policy_docs"
if not qdrant_client.collection_exists(collection_name):
    logging.warning(f"Collection '{collection_name}' does not exist. Creating it now.")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# Embeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Reranker
reranker = pipeline(
    "text-classification",
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=0 if torch.cuda.is_available() else -1
)

# LLM details
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Route: Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Route: Upload Document
@app.route('/upload', methods=['POST'])
def upload_document():
    global uploaded_file_path

    file = request.files['file']
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            uploaded_file_path = temp_file.name
            file.save(uploaded_file_path)

        try:
            # Delete the existing collection to ensure no duplicate or stale data
            try:
                qdrant_client.delete_collection(collection_name=collection_name)
                logging.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                logging.warning(f"Could not delete collection (it might not exist): {e}")

            # Recreate the collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logging.info(f"Recreated collection: {collection_name}")

            ingest_pdf(uploaded_file_path, qdrant_client, collection_name, embedding_model)

            return jsonify({"message": "Document uploaded and indexed successfully."})
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return jsonify({"error": "Error processing file."}), 500

    return jsonify({"error": "No file uploaded."}), 400

@app.route('/query', methods=['POST'])
def query_rag():
    # 1. Read incoming JSON
    data = request.json
    query_text = data.get("query", "")
    session_id = data.get("session_id", "default")  # fallback to "default" if not provided

    # NEW: optional param to rewind conversation
    rewind_to_index = data.get("rewind_to_index", None)

    logging.info(f"Received query: {query_text} for session: {session_id}")

    # 2. Initialize or fetch conversation history for this session
    if session_id not in conversation_history:
        conversation_history[session_id] = []  # list of {role: "user"/"assistant", text: ...}

    # 2b. If a rewind is requested, slice conversation_history up to that index
    if rewind_to_index is not None:
        logging.info(f"Rewinding conversation to index {rewind_to_index}")
        conversation_history[session_id] = conversation_history[session_id][:rewind_to_index]

    # 3. Add the new user query to the conversation history
    conversation_history[session_id].append({"role": "user", "text": query_text})

    # 4. Generate query embedding
    query_embedding = embedding_model.embed_query(query_text)

    # 5. Perform similarity search in Qdrant
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=10
    )

    # 6. Early return if no results or top score is too low
    if not search_results:
        no_context_answer = "No relevant context found."
        conversation_history[session_id].append({"role": "assistant", "text": no_context_answer})
        return jsonify({
            "query": query_text,
            "response": no_context_answer,
            "references": []
        })

    if search_results[0].score < 0.2:
        no_context_answer = "The answer cannot be found in the document."
        conversation_history[session_id].append({"role": "assistant", "text": no_context_answer})
        return jsonify({
            "query": query_text,
            "response": no_context_answer,
            "references": []
        })

    # 7. Rerank + Expand context
    reranked_results = rerank_results(query_text, search_results, reranker)
    final_context_results = expanded_context_filter(
        reranked_results,
        embedding_model,
        num_top_chunks=5,
        max_expansion=3,
        enrichment_threshold=0.15
    )
    if not final_context_results:
        no_context_answer = "No relevant context found."
        conversation_history[session_id].append({"role": "assistant", "text": no_context_answer})
        return jsonify({"query": query_text, "response": no_context_answer, "references": []})

    # 8. Build final context
    context_str = "\n".join([res["text"] for res in final_context_results])
    references = []
    for res in final_context_results:
        payload = res["payload"] or {}
        references.append({
            "page": payload.get("page", "N/A"),
            "score": res["score"],
            "text": payload.get("text", "No text available")
        })

    # 9. Build the *conversation* string to include in the prompt
    conversation_str = build_conversation_str(conversation_history[session_id])

    prompt = (
        f"You are an expert AI assistant. Utilize the most relevant context sources provided to answer the user's question. "
        f"Each source is separated by section headings or line breaks.\n"
        f"If you did not find anything to answer with from the context provided, state that the answer cannot be found "
        f"in the document.\nAnswer the following question directly and informatively. Provide no more than one response.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query_text}\n\nAnswer:"
    )
    logging.info(f"Prompt sent to LLM: {prompt}")

    # 11. Generate response
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.3,
            top_p=0.9
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during LLM response generation: {e}")
        error_msg = "An error occurred while generating the response."
        conversation_history[session_id].append({"role": "assistant", "text": error_msg})
        return jsonify({"error": error_msg}), 500

    # 12. Store the LLM's answer in conversation history
    conversation_history[session_id].append({"role": "assistant", "text": answer})

    print("DEBUG conversation:", conversation_history[session_id])
    logging.info(f"Response from LLM: {answer}")

    return jsonify({"query": query_text, "response": answer, "references": references})


# Debug endpoint
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Server is running!"}), 200

# Route: List all routes
@app.route('/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "rule": rule.rule,
            "doc": app.view_functions[rule.endpoint].__doc__
        })
    return jsonify({"routes": routes})

@app.route('/get_pdf')
def get_pdf():
    global uploaded_file_path
    if uploaded_file_path and os.path.exists(uploaded_file_path):
        return send_file(uploaded_file_path, as_attachment=False)
    return jsonify({"error": "No uploaded document available."}), 404

def format_text_as_html(text):
    """Format plain text with HTML for better display."""
    text = text.replace("\n", "<br>")  # Line breaks
    text = text.replace("*", "•")     # Replace asterisks with bullet points
    return f"<div style='text-align:justify;'>{text}</div>"

if __name__ == '__main__':
    app.run(debug=True)
