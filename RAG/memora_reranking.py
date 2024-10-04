import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import numpy as np
import umap
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader

import fitz  # PyMuPDF library for PDF manipulation
import unicodedata
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader,
    UnstructuredHTMLLoader,
)
import matplotlib.pyplot as plt
# ### Setup environment
#

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "llama3.1:8b"

model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

parser = StrOutputParser()
chain = model | parser
 

def normalize_text(text):
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")


def load_documents(file_paths):
    documents = []
    for file_path in file_paths:
        _, file_extension = os.path.splitext(file_path.lower())
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".doc", ".docx", ".odt"]:
            loader = Docx2txtLoader(file_path)
        elif file_extension in [".rtf", ".txt"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension in [".xls", ".xlsx", ".ods", ".csv"]:
            loader = UnstructuredExcelLoader(file_path)
        elif file_extension in [".ppt", ".pptx", ".odp"]:
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_extension in [
            ".bmp",
            ".gif",
            ".jpg",
            ".jpeg",
            ".png",
            ".svg",
            ".tiff",
        ]:
            loader = UnstructuredImageLoader(file_path)
        elif file_extension == ".html":
            loader = UnstructuredHTMLLoader(file_path)
        else:
            print(f"Unsupported file format: {file_extension}")
            continue

        documents.extend(loader.load())

    return documents


def extract_highlighted_text(pdf_path, page_num, start_char, end_char):
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Get the rectangle coordinates for the text range
    start_rect = page.get_text("words")[start_char][:4]
    end_rect = page.get_text("words")[end_char - 1][:4]

    # Create a rectangle that encompasses the text range
    highlight_rect = fitz.Rect(start_rect[0], start_rect[1], end_rect[2], end_rect[3])

    # Extract the text within the rectangle
    highlighted_text = page.get_text("text", clip=highlight_rect)

    # Optionally, you can still add a highlight annotation if needed
    # page.add_highlight_annot(highlight_rect)

    doc.close()
    return highlighted_text


def create_citation(document, relevant_text):
    return {
        "document_name": document.metadata.get("source", "Unknown"),
        "page_number": document.metadata.get("page", 0) + 1,
        "text": relevant_text,
        "start_char": document.page_content.index(relevant_text),
        "end_char": document.page_content.index(relevant_text) + len(relevant_text),
    }


# ### Load and split documents
#
# Load PDF Content and return token_split_texts
reader = PdfReader("dotnet.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Add documents to Chroma collection
embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function([token_split_texts[10]]))

chroma_client = chromadb.Client()

chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
count = chroma_collection.count()

# Retrieve documents
query = "What was the total revenue for the year?"

results = chroma_collection.query(
    query_texts=query, n_results=10, include=["documents", "embeddings"]
)

retrieved_documents = results["documents"][0]

for document in results["documents"][0]:
    print(word_wrap(document))
    print("")

# from sentence_transformers import CrossEncoder

# Rank documents with cross-encoder
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# pairs = [[query, doc] for doc in retrieved_documents]
# scores = cross_encoder.predict(pairs)

# print("Scores:")
# for score in scores:
#     print(score)

# print("New Ordering:")
# for o in np.argsort(scores)[::-1]:
#     print(o + 1)

original_query = "What is the purpose of .NET"


def generate_multi_query(query, model=None):
    if model is None:
        model = Ollama(model=MODEL)

    prompt = """
    You are a knowledgeable software development assistant. 
    Your users are inquiring about software information. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
                """
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = model.invoke(messages)
    aug_queries = [q.strip() for q in response.split("\n") if q.strip()]
    return aug_queries

generated_queries = generate_multi_query(original_query)
print("Augmented Query ----------------------")
for query in generated_queries:
    print("\n", query)
# concatenate the original query with the generated queries
queries = [original_query] + generated_queries


results = chroma_collection.query(
    query_texts=queries, n_results=10, include=["documents", "embeddings"]
)

retrieved_documents = results["documents"]

# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

unique_documents = list(unique_documents)

pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])

# scores = cross_encoder.predict(pairs)

# print("Scores:")
# for score in scores:
#     print(score)

# print("New Ordering:")
# for o in np.argsort(scores)[::-1]:
#     print(o)
# ====
# top_indices = np.argsort(scores)[::-1][:5]
# top_documents = [unique_documents[i] for i in top_indices]

# Concatenate the top documents into a single context
context = "\n\n".join(unique_documents)


# Generate the final answer using the OpenAI model
def generate_final_answer(query, context, model=MODEL):
    model = Ollama(model=MODEL)
    prompt = """
    You are a knowledgeable software development assistant. 
    Your users are inquiring about information about software. 
    """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"based on the following context:\n\n{context}\n\nAnswer the query: '{query}'",
        },
    ]

    response = model.invoke(messages)
    aug_queries = [q.strip() for q in response.split("\n") if q.strip()]
    return aug_queries


res = generate_final_answer(query=original_query, context=context)
print("Final Answer:")
print(res)
