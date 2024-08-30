from flask import Flask, request, jsonify
from document_processor import load_documents, split_documents
from qa_system import QASystem
from conversation_memory import ConversationMemory

app = Flask(__name__)

qa_system = None
conversation_memory = None

@app.route('/initialize', methods=['POST'])
def initialize():
    global qa_system, conversation_memory
    
    file_paths = request.json.get('file_paths', [])
    if not file_paths:
        return jsonify({"error": "No file paths provided"}), 400
    
    try:
        documents = load_documents(file_paths)
        split_docs = split_documents(documents)
        
        qa_system = QASystem()
        qa_system.initialize(documents)
        
        conversation_memory = ConversationMemory()
        
        return jsonify({"message": "QA system initialized successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global qa_system, conversation_memory
    
    if not qa_system or not conversation_memory:
        return jsonify({"error": "QA system not initialized. Call /initialize first."}), 400
    
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        response, citations = qa_system.generate_response_with_citations(question, conversation_memory)
        formatted_response = qa_system.format_response_with_citations(response, citations)
        
        return jsonify({
            "question": question,
            "answer": response,
            "formatted_response": formatted_response,
            "citations": citations
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/conversation_history', methods=['GET'])
def get_conversation_history():
    global conversation_memory
    
    if not conversation_memory:
        return jsonify({"error": "QA system not initialized. Call /initialize first."}), 400
    
    return jsonify({
        "history": conversation_memory.history
    }), 200

if __name__ == '__main__':
    app.run(debug=True)