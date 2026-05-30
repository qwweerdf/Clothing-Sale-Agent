import os
from dotenv import load_dotenv

load_dotenv()


def get_api_key(llm_type="OPENAI_API_KEY"):
    api_key = os.getenv(llm_type)
    os.environ[llm_type] = api_key
    return api_key
