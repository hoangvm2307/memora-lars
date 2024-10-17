from prompts import prompts
from params.answer_params import AnswerParams
import vertexai
from langchain_google_vertexai import VertexAI
from google.cloud import aiplatform
from google.oauth2 import service_account
import os

CREDENTIALS_JSON = {
    "type": "service_account",
    "project_id": "memora-436413",
    "private_key_id": "e75054c62ee809e099affcfcaf66294be92c88a2",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDp6gkZo3cCfA9i\ncNL1zQ/LLWOcF+gT8AzSQdV83nLeUnfDZ33fbqITUYLmVROs6YmazZheN23KVtzF\nDAfdMzikOfpjbdqnjnhzNBWcXYY2mtdBzSVT70Yf6qKYT0M8s2GdDwJxHbRhqV3K\ndt0Axq3rVo1V9TIjub9q3yjfs+JViCtWeuVd0V8ayAZzAW8djnr16oieOWysJWAF\nZqKs16yh7biAPv1ITDPN4bbTSiBGePLymBQHDsXdT0OA88d7jGFdP0rqJsJseO9I\nfhnCkaazf5G0LgNUjaYkcR2mlwRvW58ix14atCzqN1TjwmE9CBhafcp8AJ5kBVCb\n5LgFZwhzAgMBAAECggEABVWLcJS8NYfLuAR3oxhSiGEYGOqopa+/MxpCOo8+EljS\nF1goVCyLMKKWuBbvlG1c8HVIyLk93TPe4/VyxnYREBmsXpEWP+TtVT82UPMHCP69\n8uexNrlJoZexroR8NYmnq7O4bAjv4JxCocfVIVvmgmveXDvzsUHenhrJrVRGUIn3\nveAMV09q/8sjkir1cINmAAFgEme/3OGE8rnh9g/3Fo76CxRYicsrqzVNZHo7tuaU\nAeqmPLkHFb/3SynVOrY5iMnxA1+Vs3PLksRMbz2sxtl15go2Yr6HSIyk7j/kIEuN\njV/A6Ewbb52gGDOv9Htja3CTbQJmwkyqtzi/XyOiVQKBgQD04IZ1cuLDXUA/FaHd\nUQRpuQNYpK32ME+W2o9ZeojIrE9o4giJ4fXmzIkDfDtAlN1xkohTg/sLQKSdZ142\nr8+ILXbxeft7vkX+U7wcwE89cejhYPx+QkfSUprDQKpwIoho6Fg6QHCiDLttr61e\nV3E0EF2B+c6NbNE8ry3j+eZSfQKBgQD0ighFNDOsO+wYVajzcRh/I9Rq2iSN7cQr\n9l477jsqcXQRxV9evB642ycX7YTc7hwp2DKrdvSWFCLN/UyG2mQpQ74bCG7svo7s\nEEXD4BE/HQ/sRy8MfL9FSgNSR6Kv2L3/Vz6zeXrUp6/a5loEJw0ELgjWwRcghOB7\nv5WUxVtJrwKBgQCe7ztV74MOme+hAkFUi8j5dYOefQQLzb9agfCYetdcp7nCsTIp\ni+c2LXqgMHmkqPoxRJIG4pqF6ybsorKbe/COyjNw92MqJYz4TRDC+G51ywEDhxda\nO1qyP7sDD22P6lnu/R6GcFyqUOk1f9heaxKmYBjQy3osgHwtjuWSGhhLYQKBgQCl\ns0KXYiABfTkl5CVvkBsBS84L+XT7lzlucKq6AVumDuqPgCZ3kxFeQWHkHNYCvnXn\nBNCQzzI837gVzKWmWyWzsGuI9dX0JcvCueQMLjCBi7fWawW+eGlDEjvd7RyX+04D\nT6L1CkPpBRdsRNqKJcv0IR1sJ7r8Fg3mzJMXFAQfkQKBgCuadtYg04Svk5uA1JxL\nMbD/bwiv4SASbtEl55J/0VOLKqyg7/69GPAGNTDQsmw3SUDNpr6SVUaRKuK4OYMC\ndRzHfHt9YSbmAMg0CSzCuBHm0T0Jz+E+TVYWQHEC0c6Tk/sS0+WVtfRjDYvsQr8i\nU9cCxpDDak3Ch322JzrtbAYE\n-----END PRIVATE KEY-----\n",
    "client_email": "admin-709@memora-436413.iam.gserviceaccount.com",
    "client_id": "118117741709208187681",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/admin-709%40memora-436413.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com",
}
credentials = service_account.Credentials.from_service_account_info(
    CREDENTIALS_JSON,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)


def generate_final_answer(config: AnswerParams):
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "memora-436413")

    location = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")

    aiplatform.init(project=project_id, location=location, credentials=credentials)

    MODEL = "gemini-1.5-flash-001"
    model = VertexAI(model_name=MODEL)
    prompt = prompts[config.prompt_type].format(
        count=config.count, context=config.context, query=config.query
    )

    response = model.invoke(prompt)

    response = [q.strip() for q in response.split("\n") if q.strip()]
    return response


def generate_multi_query(query, model=None):
    vertexai.init(project="memora-436413", location="asia-southeast1")
    MODEL = "gemini-1.5-flash-001"
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
