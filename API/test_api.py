import requests
import json
import os
import time

# API endpoint URLs
BASE_URL = (
    "http://localhost:5000"  # Change this if your API is running on a different address
)
PROCESS_PDF_URL = f"{BASE_URL}/initialize"
QUERY_URL = f"{BASE_URL}/query"
start_time = time.time()
def test_process_pdf():
    # Path to your test PDF file
    pdf_path = "english.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return None

    # Prepare the file for sending
    files = {'file': ('english.pdf', open(pdf_path, 'rb'), 'application/pdf')}
    
    # Send the request
    response = requests.post(PROCESS_PDF_URL, files=files)

    print("Process PDF Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}")

    end_time = time.time()
    print(f"Thời gian xử lý initialize: {end_time - start_time} giây")
    
    return response.json().get("collection_name")



def test_query(collection_name):
    query_data = {
        "query": "",
        "collection_name": collection_name,
        "prompt_type": "multiple_choice",
        "count": 10,
    }

    response = requests.post(QUERY_URL, json=query_data)

    print("\nQuery Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}")
    
    end_time = time.time()
    print(f"Thời gian xử lý query: {end_time - start_time} giây")

if __name__ == "__main__":
    # Test process_pdf endpoint
    collection_name = test_process_pdf()

    if collection_name:
        # Test query endpoint
        test_query(collection_name)
    else:
        print("Failed to process PDF. Cannot proceed with query test.")
