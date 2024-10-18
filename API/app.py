from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import chromadb
from flask_cors import CORS

# from sentence_transformers import CrossEncoder
from query_service import generate_final_answer
from params.answer_params import AnswerParams
from chroma_service import add_documents_to_chroma, delete_user_collections
from pdf_service import load_and_split_pdf
import asyncio
import base64
import re
import tempfile
from werkzeug.utils import secure_filename
import traceback
app = Flask(__name__)
CORS(app)
# Load environment variables
load_dotenv()

chroma_client = chromadb.Client()
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def decode_base64(data, altchars=b"+/"):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb"[^a-zA-Z0-9%s]+" % altchars, b"", data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b"=" * (4 - missing_padding)
    return base64.b64decode(data, altchars)


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
async def process_document():
    if 'file' not in request.files:
        return jsonify({"error": "Thiếu file"}), 400

    if 'userId' not in request.form:
        return jsonify({"error": "Thiếu userId"}), 400

    file = request.files['file']
    user_id = request.form['userId']

    if file.filename == '':
        return jsonify({"error": "Không có file nào được chọn"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            # Xóa tất cả các collection cũ của người dùng
            delete_user_collections(chroma_client, user_id)

            # Xử lý và thêm tài liệu mới
            texts = await asyncio.to_thread(load_and_split_pdf, temp_file_path)
            collection_name = f"{user_id}"
            add_documents_to_chroma(chroma_client, collection_name, texts)

            os.unlink(temp_file_path)

            return jsonify({
                "message": "PDF đã được xử lý thành công và các collection cũ đã được xóa",
                "collection_name": collection_name,
            }), 200
        except Exception as e:
            traceback.print_exc()
            os.unlink(temp_file_path)
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Loại file không được phép"}), 400


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf"}


@app.route("/quizzes", methods=["POST"])
def generate_quizzes():
    data = request.json
    if not data or "collection_name" not in data:
        return jsonify({"error": "Missing query or collection_name"}), 400

    user_id = data["userId"]
    original_query = ""
    collection_name = data["collection_name"]
    
    if not collection_name.startswith(f"{user_id}"):
        return jsonify({"error": "Unauthorized access to collection"}), 403
    
    prompt_type = data["prompt_type"] if "prompt_type" in data else "multiple_choice"
    if prompt_type != "multiple_choice" and prompt_type != "true_false":
        prompt_type = "multiple_choice"
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
        }
    ), 200


@app.route("/cards", methods=["POST"])
def generate_cards():
    data = request.json
    if not data  or "collection_name" not in data:
        return jsonify({"error": "Missing collection_name"}), 400

    user_id = data["userId"]
    original_query = ""
    collection_name = data["collection_name"]
    
    if not collection_name.startswith(f"{user_id}"):
        return jsonify({"error": "Unauthorized access to collection"}), 403
    
    prompt_type = "card"
    count = data["count"] if "count" in data else 5

    print(f"Quiz count: {count}")
    chroma_collection = chroma_client.get_collection(collection_name)

    # # Generate multi queries related to original query
    # generated_queries = generate_multi_query(original_query)
    # queries = [original_query] + generated_queries

    # Retrieve documents with augmented queries
    results = chroma_collection.query(
        query_texts=original_query, n_results=1000, include=["documents", "embeddings"]
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
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
