from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import CrossEncoder
import numpy as np
from query_service import generate_final_answer, generate_multi_query
from params.answer_params import AnswerParams
from chroma_service import add_documents_to_chroma
from pdf_service import load_and_split_pdf
import asyncio
import base64
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize global variables
MODEL = "llama3.1:8b"

chroma_client = chromadb.Client()
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@app.route("/initialize", methods=["POST"])
async def initialize():
    file_name = request.json.get("filename")
    print("File path: ", file_name)
    if not file_name:
        return jsonify({"error": "No selected file"}), 400

    try:
        texts = await asyncio.to_thread(load_and_split_pdf, file_name)
        collection_name = f"{file_name.replace('.pdf', '')}-collection"
        add_documents_to_chroma(chroma_client, collection_name, texts)
        
        return jsonify(
            {
                "message": "PDF processed successfully",
                "collection_name": collection_name,
            }
        ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process-document", methods=["POST"])
async def process_pdf():
    data = request.json
    file_name = data.get("filename")
    file_content = data.get("fileContent")

    if not file_name or not file_content:
        return jsonify({"error": "No file data provided"}), 400

    try:
        # Decode Base64 content
        pdf_content = base64.b64decode(file_content)

        # Process the PDF content
        texts = await asyncio.to_thread(load_and_split_pdf, pdf_content)
        collection_name = f"{file_name.replace('.pdf', '')}-collection"
        add_documents_to_chroma(chroma_client, collection_name, texts)
        
        return jsonify({
            "message": "PDF processed successfully",
            "collection_name": collection_name,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    if not data or "query" not in data or "collection_name" not in data:
        return jsonify({"error": "Missing query or collection_name"}), 400

    original_query = data["query"]
    collection_name = data["collection_name"]
    prompt_type = data["prompt_type"] if "prompt_type" in data else "default"
    count = data["count"] if "count" in data else 5
 
    print(f"Quiz count: {count}")
    chroma_collection = chroma_client.get_collection(collection_name)

    # # Generate multi queries related to original query
    # generated_queries = generate_multi_query(original_query)
    # queries = [original_query] + generated_queries

    # Retrieve documents with augmented queries
    results = chroma_collection.query(
        query_texts=original_query, n_results=5, include=["documents", "embeddings"]
    )

    # Get unique documents
    unique_documents = set()
    for documents in results["documents"]:
        for document in documents:
            unique_documents.add(document)

    unique_documents = list(unique_documents)

    pairs = [[original_query, doc] for doc in unique_documents]
    scores = cross_encoder.predict(pairs)

    # Get the top 5 documents based on scores
    top_indices = np.argsort(scores)[::-1][:3]
    top_documents = [unique_documents[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    # Generate final answer based on the top documents' context
    context = "\n\n".join(top_documents)
    params = AnswerParams(original_query, context, prompt_type, count)
    final_answer = generate_final_answer(params)

    # Prepare the response with top documents and their scores
    top_documents_with_scores = [
        {"document": doc, "score": float(score)}
        for doc, score in zip(top_documents, top_scores)
    ]

    return jsonify(
        {
            "query": original_query,
            "answer": final_answer,
            "top_documents": top_documents_with_scores,
        }
    ), 200


if __name__ == "__main__":
    app.run(debug=True)
