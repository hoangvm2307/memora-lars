from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import chromadb

# from sentence_transformers import CrossEncoder
import numpy as np
from query_service import generate_final_answer, generate_multi_query
from params.answer_params import AnswerParams
from chroma_service import add_documents_to_chroma
from pdf_service import load_and_split_pdf
import asyncio
import base64
import logging
from typing import Optional, Tuple

app = Flask(__name__)

# Load environment variables
load_dotenv()

chroma_client = chromadb.Client()


# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def decode_base64_content(
    base64_string: str,
) -> Tuple[bool, Optional[bytes], Optional[str]]:
    """
    Decode base64 string and handle padding issues
    Returns: (success, decoded_content, error_message)
    """
    try:
        # Ensure proper padding
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += "=" * (4 - missing_padding)

        # Try to decode
        decoded_content = base64.b64decode(base64_string)
        return True, decoded_content, None

    except Exception as e:
        logging.error(f"Base64 decode error: {str(e)}")
        return False, None, str(e)


@app.route("/initialize", methods=["POST"])
async def process_document():
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        file_content = data.get("fileContent")
        if not file_content:
            return jsonify({"error": "No file content provided"}), 400

        # Clean the base64 string
        # Remove potential data URL prefix
        if "," in file_content:
            file_content = file_content.split(",")[1]

        # Remove whitespace and newlines
        file_content = file_content.strip().replace("\n", "").replace("\r", "")

        # Decode base64
        success, decoded_content, error = decode_base64_content(file_content)
        if not success:
            return jsonify({"error": f"Base64 decode error: {error}"}), 400

        # Process the PDF content
        texts = await asyncio.to_thread(load_and_split_pdf, decoded_content)
        collection_name = (
            f"{data.get('filename', 'unnamed').replace('.pdf', '')}-collection"
        )

        add_documents_to_chroma(chroma_client, collection_name, texts)

        return jsonify(
            {
                "message": "PDF processed successfully",
                "collection_name": collection_name,
            }
        ), 200

    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
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

        return jsonify(
            {
                "message": "PDF processed successfully",
                "collection_name": collection_name,
            }
        ), 200
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
    # scores = cross_encoder.predict(pairs)

    # Get the top 5 documents based on scores
    # top_indices = np.argsort(scores)[::-1][:3]
    # top_documents = [unique_documents[i] for i in top_indices]
    # top_scores = [scores[i] for i in top_indices]

    # Generate final answer based on the top documents' context
    context = "\n\n".join(unique_documents)
    params = AnswerParams(original_query, context, prompt_type, count)
    final_answer = generate_final_answer(params)

    # Prepare the response with top documents and their scores
    # top_documents_with_scores = [
    #     {"document": doc, "score": float(score)}
    #     for doc, score in zip(top_documents, top_scores)
    # ]

    return jsonify(
        {
            "query": original_query,
            "answer": final_answer,
            # "top_documents": top_documents_with_scores,
        }
    ), 200

 