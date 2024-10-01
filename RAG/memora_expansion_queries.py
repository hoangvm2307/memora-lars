import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
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

# ### Utility functions
#


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

query = "What was the total revenue for the year?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]


def generate_multi_query(query, model=None):

    if model is None:
        model = Ollama(model=MODEL)

    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
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


original_query = (
    "What details can you provide about the factors that led to revenue growth?"
)
aug_queries = generate_multi_query(original_query)

# 1. First step show the augmented queries
print("Augmented Query ----------------------")
for query in aug_queries:
    print("\n", query)

# 2. concatenate the original query with the augmented queries
joint_query = [ 
    original_query
] + aug_queries  # original query is in a list because chroma can actually handle multiple queries, so we add it in a list

# print("======> \n\n", joint_query)

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)


# output the results documents
for i, documents in enumerate(retrieved_documents):
    print(f"Query: {joint_query[i]}")
    print("")
    print("Results:")
    for doc in documents:
        print(word_wrap(doc))
        print("")
    print("-" * 100)

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# 4. We can also visualize the results in the embedding space
original_query_embedding = embedding_function([original_query])
augmented_query_embeddings = embedding_function(joint_query)


project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(
    augmented_query_embeddings, umap_transform
)

retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)



# Plot the projected query and retrieved documents in the embedding space
plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot
