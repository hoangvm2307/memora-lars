from dotenv import load_dotenv
from langchain_google_vertexai import VertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from helper_utils import word_wrap
from pypdf import PdfReader

import vertexai
# Setup environment
load_dotenv()

# Initialize Vertex AI
vertexai.init(project="memora-436413", location="asia-southeast1")

# Set up the VertexAI model
MODEL = "gemini-1.5-flash-001"  # or any other appropriate Vertex AI model
model = VertexAI(model_name=MODEL)

parser = StrOutputParser()
chain = model | parser

# The rest of your functions (normalize_text, load_documents, extract_highlighted_text, create_citation) remain the same

# Load and split documents
reader = PdfReader("dotnet.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=750, chunk_overlap=30
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

chroma_client = chromadb.Client()

chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# Extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
count = chroma_collection.count()

# Retrieve documents
query = "What was the total revenue for the year?"

results = chroma_collection.query(
    query_texts=query, n_results=5, include=["documents", "embeddings"]
)

retrieved_documents = results["documents"][0]

for document in results["documents"][0]:
    print(word_wrap(document))
    print("")

from sentence_transformers import CrossEncoder

# Rank documents with cross-encoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o + 1)

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

# generated_queries = generate_multi_query(original_query)
# print("Augmented Query ----------------------")
# for query in generated_queries:
#     print("\n", query)

# # Concatenate the original query with the generated queries
# queries = [original_query] + generated_queries

results = chroma_collection.query(
    query_texts=original_query, n_results=5, include=["documents", "embeddings"]
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

scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o)

top_indices = np.argsort(scores)[::-1][:3]
top_documents = [unique_documents[i] for i in top_indices]

# Concatenate the top documents into a single context
context = "\n\n".join(top_documents)

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