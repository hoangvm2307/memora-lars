import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


chroma_collection = None


def init_chroma():
    global chroma_client, chroma_collection
    chroma_client = chromadb.Client()
    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_collection = chroma_client.create_collection(
        "microsoft-collection", embedding_function=embedding_function
    )


def delete_user_collections(chroma_client, user_id):
    collections = chroma_client.list_collections()
    for collection in collections:
        if collection.name.startswith(f"{user_id}-"):
            chroma_client.delete_collection(collection.name)


def query_chroma(queries, n_results=10):
    return chroma_collection.query(
        query_texts=queries, n_results=n_results, include=["documents", "embeddings"]
    )


def add_documents_to_chroma(chroma_client, collection_name, token_split_texts):
    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_collection = chroma_client.get_or_create_collection(
        collection_name, embedding_function=embedding_function
    )

    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts)

    return chroma_collection


def retrieve_documents(query, chroma_collection, n_results=10):
    results = chroma_collection.query(
        query_texts=query, n_results=n_results, include=["documents", "embeddings"]
    )
    return results["documents"]
