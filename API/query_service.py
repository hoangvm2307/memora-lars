from langchain_community.llms import Ollama
from prompts import prompts
from params.answer_params import AnswerParams

MODEL = "llama3.1:8b"


def generate_final_answer(config: AnswerParams, model=MODEL):
    model = Ollama(model=MODEL)

    if config.prompt_type == "quiz":
        prompt = prompts[config.prompt_type].format(quiz_count=config.quiz_count)
    else:
        prompt = prompts.get(config.prompt_type, prompts["default"])
    print(f"Prompt: {prompt}")
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"based on the following context:\n\n{config.context}\n\nAnswer the query: '{config.query}'",
        },
    ]

    response = model.invoke(messages)
    aug_queries = [q.strip() for q in response.split("\n") if q.strip()]
    return aug_queries


def generate_multi_query(query, model=None):
    if model is None:
        model = Ollama(model=MODEL)

    # prompt = """
    # You are a knowledgeable software development assistant.
    # Your users are inquiring about software information.
    # For the given question, propose up to five related questions to assist them in finding the information they need.
    # Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic.
    # Ensure each question is complete and directly related to the original inquiry.
    # List each question on a separate line without numbering.
    # """
    prompt = prompts.get("multi_query", prompts["default"])
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
