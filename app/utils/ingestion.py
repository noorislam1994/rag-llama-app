import logging
import tempfile
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def ingest_pdf(uploaded_file_path, qdrant_client, collection_name, embedding_model):
    """
    Ingests a single PDF by splitting it into chunks, embedding, 
    and upserting to Qdrant. 
    Returns the number of chunks ingested.
    """
    try:
        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(uploaded_file_path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Generate embeddings for the documents
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        text_embeddings = embedding_model.embed_documents(texts)

        # Prepare points for upsert
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=text_embeddings[i],
                payload={"text": texts[i], **metadatas[i]}
            )
            for i in range(len(text_embeddings))
        ]

        # Upsert points into Qdrant
        qdrant_client.upsert(collection_name=collection_name, points=points)
        logging.info(f"Indexed {len(points)} chunks into the collection.")

        return len(points)
    except Exception as e:
        logging.error(f"Error ingesting PDF: {e}")
        raise
