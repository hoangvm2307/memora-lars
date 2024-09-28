import requests
import json

# API endpoint URLs
BASE_URL = (
    "http://localhost:5000"  # Change this if your API is running on a different address
)
PROCESS_PDF_URL = f"{BASE_URL}/initialize"
QUERY_URL = f"{BASE_URL}/query"


def test_process_pdf():
    # Path to your test PDF file
    pdf_path = "english.pdf"
    data = {"filename": pdf_path}
    response = requests.post(PROCESS_PDF_URL, json=data)

    print("Process PDF Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}")

    return response.json().get("collection_name")


def test_query(collection_name):
    query_data = {
        "query": "",
        "collection_name": collection_name,
        "prompt_type": "multi_query",
        "count": 10,
    }

    response = requests.post(QUERY_URL, json=query_data)

    print("\nQuery Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}")


if __name__ == "__main__":
    # Test process_pdf endpoint
    collection_name = test_process_pdf()

    if collection_name:
        # Test query endpoint
        test_query(collection_name)
    else:
        print("Failed to process PDF. Cannot proceed with query test.")
