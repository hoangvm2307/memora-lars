from dotenv import load_dotenv
from langchain_google_vertexai import VertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from helper_utils import word_wrap
from pypdf import PdfReader
import vertexai
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor

# Setup environment
load_dotenv()

# Initialize Vertex AI
vertexai.init(project="memora-436413", location="asia-southeast1")

# Set up the VertexAI model
MODEL = "gemini-1.5-flash-001"
model = VertexAI(model_name=MODEL)

parser = StrOutputParser()
chain = model | parser

# Load and split documents
reader = PdfReader("dotnet.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], 
    chunk_size=750,  # Changed from 1000 to 750
    chunk_overlap=50  # Added overlap
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Embedding cache implementation
def get_embedding_cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_or_create_embedding(text, embedding_function):
    cache_key = get_embedding_cache_key(text)
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    embedding = embedding_function([text])[0]
    embedding_cache[cache_key] = embedding
    return embedding

# Parallel processing for embedding generation
def create_embedding_parallel(texts, embedding_function, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(executor.map(embedding_function, texts))
    return embeddings

# Add documents to Chroma collection
embedding_function = SentenceTransformerEmbeddingFunction()

# Use HNSW index
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db",
    anonymized_telemetry=False
))

chroma_collection = chroma_client.create_collection(
    "microsoft-collection", 
    embedding_function=embedding_function,
    hnsw_config={
        "M": 64,
        "efConstruction": 128,
        "efSearch": 128
    }
)

# Use embedding cache and parallel processing
embedding_cache = {}
embeddings = create_embedding_parallel(token_split_texts, embedding_function)

# Extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts, embeddings=embeddings)
count = chroma_collection.count()

# Retrieve documents
query = "What was the total revenue for the year?"

results = chroma_collection.query(
    query_texts=query, 
    n_results=3,  # Changed from 5 to 3
    include=["documents", "embeddings"]
)

retrieved_documents = results["documents"][0]

for document in results["documents"][0]:
    print(word_wrap(document))
    print("")

# from sentence_transformers import CrossEncoder

# Rank documents with cross-encoder
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [[query, doc] for doc in retrieved_documents]
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
        model = VertexAI(model_name=MODEL)

    prompt = """
    You are a knowledgeable software development assistant. 
    Your users are inquiring about software information. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
    """
    
    response = model.invoke(prompt + "\n\n" + query)
    aug_queries = [q.strip() for q in response.split("\n") if q.strip()]
    return aug_queries

results = chroma_collection.query(
    query_texts=original_query, 
    n_results=3,  # Changed from 5 to 3
    include=["documents", "embeddings"]
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

# top_indices = np.argsort(scores)[::-1][:3]
# top_documents = [unique_documents[i] for i in top_indices]

# Concatenate the top documents into a single context
context = "\n\n".join(unique_documents)

# Generate the final answer using the Vertex AI model
def generate_final_answer(query, context, model_name=MODEL):
    model = VertexAI(model_name=model_name)
    prompt = """
    You are a knowledgeable software development assistant. 
    Your users are inquiring about information about software. 
    Based on the following context:

    {context}

    Answer the query: '{query}'
    """
    
    formatted_prompt = prompt.format(context=context, query=query)
    response = model.invoke(formatted_prompt)
    return response

res = generate_final_answer(query=original_query, context=context)
print("Final Answer:")
print(res)

# Save embedding cache
with open('embedding_cache.pkl', 'wb') as f:
    pickle.dump(embedding_cache, f)