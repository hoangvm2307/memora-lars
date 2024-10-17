from prompts import prompts
from params.answer_params import AnswerParams
import vertexai
from langchain_google_vertexai import VertexAI
import os
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request

 
def generate_final_answer(config: AnswerParams):
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "memora-436413")
 
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")

    vertexai.init(project=project_id, location=location)

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
