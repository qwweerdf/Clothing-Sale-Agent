from langchain_qwq import ChatQwen

from utils.load_api_key import get_api_key

get_api_key(llm_type="DASHSCOPE_API_KEY")

qwen_llm = ChatQwen(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30,
)
