import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

def rerank_results(query, results, reranker, max_length=512, batch_size=8):
    # same logic you used in main.py
    ranked_results = []
    batched_inputs = []
    original_data = []

    for result in results:
        text = result.payload.get("text", "")
        combined_input = f"{query} [SEP] {text}"
        batched_inputs.append(combined_input)
        original_data.append({
            "original_score": result.score,
            "text": text,
            "payload": result.payload
        })

    for i in range(0, len(batched_inputs), batch_size):
        batch = batched_inputs[i : i + batch_size]
        try:
            scores = reranker(batch, truncation=True, max_length=max_length)
            for j, score in enumerate(scores):
                ranked_results.append({
                    "score": score["score"],
                    "text": original_data[i + j]["text"],
                    "original_score": original_data[i + j]["original_score"],
                    "payload": original_data[i + j]["payload"],
                })
        except Exception as e:
            logging.error(f"Error reranking batch {i // batch_size}: {str(e)}")

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results


def expanded_context_filter(reranked_results, embedding_model, 
                            num_top_chunks=5, max_expansion=3, enrichment_threshold=0.85):
    if not reranked_results:
        return []

    top_chunks = reranked_results[:num_top_chunks]
    expanded_context = top_chunks[:]
    seen_texts = [chunk['text'] for chunk in top_chunks]

    initial_embeddings = embedding_model.embed_documents(seen_texts)

    for result in reranked_results[num_top_chunks:]:
        if len(expanded_context) >= num_top_chunks + max_expansion:
            break

        new_text = result['text']
        new_embedding = embedding_model.embed_query(new_text)

        similarities = cosine_similarity([new_embedding], initial_embeddings).flatten()
        if np.all(similarities < enrichment_threshold):
            expanded_context.append(result)
            initial_embeddings = np.vstack([initial_embeddings, new_embedding])

    expanded_context = sorted(expanded_context, key=lambda x: x["score"], reverse=True)
    return expanded_context


def build_conversation_str(conversation_list, max_turns=5):
    truncated_conversation = conversation_list[-(max_turns*2):]
    conversation_str = ""
    for msg in truncated_conversation:
        role = msg["role"]
        text = msg["text"]
        if role == "user":
            conversation_str += f"User: {text}\n"
        elif role == "assistant":
            conversation_str += f"Assistant: {text}\n"
    return conversation_str
