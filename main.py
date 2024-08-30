from document_processor import load_documents, split_documents
from qa_system import QASystem
from conversation_memory import ConversationMemory
from memora_api import app

def main():
    # Load and process documents
    file_paths = ["dotnet.pdf"]  # Add more file paths as needed
    documents = load_documents(file_paths)
    # split_docs = split_documents(documents)

    # Initialize QA system
    qa_system = QASystem()
    qa_system.initialize(documents)

    # Initialize conversation memory
    conversation_memory = ConversationMemory()

    # Example usage
    questions = [
        "What is the purpose of .NET?",
        "What is the environment of .NET?",
        "How does .NET handle errors?",
        "Do you have information about DLL?",
        "What is the purpose of .NET?",
    ]

    for question in questions:
        print(f"Question: {question}")
        response, citations = qa_system.generate_response_with_citations(
            question, conversation_memory
        )
        formatted_response = qa_system.format_response_with_citations(
            response, citations
        )
        print(formatted_response)
        print()


if __name__ == "__main__":
    # main()
    app.run(debug=True)
