import re
from typing import List, Dict, Tuple
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter
from langchain.schema import Document
from conversation_memory import ConversationMemory


class QASystem:
    def __init__(self, model_name: str = "llama3"):
        self.model = Ollama(model=model_name)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.parser = StrOutputParser()
        self.vectorstore = None
        self.retriever = None
        self.chain = None

    def initialize(self, documents: List[Document]):
        self.vectorstore = DocArrayInMemorySearch.from_documents(
            documents, embedding=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 5}
        )

        template = """
        Answer the question based on the context below and the conversation history. If you can't answer the question, reply "I don't know".
        When using information from the context, sources with the format [Citation X] must be included where X is the number of citation of each answer. 
        If answer come from the same source, reuse the same citation number.  

        Context: {context}

        Conversation History:
        {history}

        Question: {question}

        Answer:
        """

        prompt = PromptTemplate.from_template(template)

        self.chain = (
            {
                "context": itemgetter("question") | self.retriever,
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | prompt
            | self.model
            | self.parser
        )

    def create_citation(self, document: Document, relevant_text: str) -> Dict[str, any]:
        return {
            "document_name": document.metadata.get("source", "Unknown"),
            "page_number": document.metadata.get("page", 0) + 1,
            "text": relevant_text,
            "start_char": document.page_content.index(relevant_text),
            "end_char": document.page_content.index(relevant_text) + len(relevant_text),
        }

    def generate_response_with_citations(
        self, question: str, conversation_memory: ConversationMemory
    ) -> Tuple[str, List[Dict[str, any]]]:
        retrieved_docs = self.retriever.invoke(question)
        context = ""
        citations = []
        for i, doc in enumerate(retrieved_docs):
            relevant_text = doc.page_content
            citation = self.create_citation(doc, relevant_text)
            citations.append(citation)
            context += f"[Citation {i + 1}] {relevant_text}\n\n"

        history = conversation_memory.get_formatted_history()
        response = self.chain.invoke(
            {"context": context, "question": question, "history": history}
        )

        used_citations = []
        for match in re.finditer(r"\[Citation (\d+)\]", response):
            citation_num = int(match.group(1))
            if 1 <= citation_num <= len(citations):
                used_citations.append(citations[citation_num - 1])

        conversation_memory.add_interaction(question, response)
        return response, used_citations

    @staticmethod
    def format_response_with_citations(
        response: str, citations: List[Dict[str, any]]
    ) -> str:
        formatted_response = f"{response}\n\nCitations:\n"
        if not citations:
            formatted_response += "No citations available.\n"
        for i, citation in enumerate(citations):
            try:
                formatted_response += f"{i+1}. Document: {citation['document_name']}, Page: {citation['page_number']}\n"
                formatted_response += f"   Text: {citation['text'][:100]}...\n\n"
            except Exception as e:
                formatted_response += f"{i+1}. Error formatting citation: {str(e)}\n\n"
        return formatted_response
