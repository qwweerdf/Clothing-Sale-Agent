# 1. 尺码/材质/颜色/价格问题，这些都是用户会发过来链接，问相关的问题，在这个商品里做rag就可以
# 2. 物流/配送/退换货政策相关的商品外问题，准备独立的物流相关的知识库
# 3. 商品推荐，库存做rag
# 需要做3个知识库，3个工具
from langchain_core.tools import tool
from langchain_qwq import ChatQwen

from sale_presale_retrieval import PresaleRetrieval

llm = ChatQwen(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=20
)

@tool
def presale_metadata(query: str, referred_item: str) -> str:
    """
    处理尺码/材质/颜色/价格等售前问题

    @:param
    query: 用户问题
    referred_item: 用户引用的衣服物品元数据信息

    :return 大模型返回
    """
    # 用referred_item做rag
    prompt_template = f"""
    你是一个衣物售前客服，请为用户提供衣物的尺码/材质/颜色/价格等售前问题答疑服务，
    【相关信息】中会记录用户闻讯的对应的物品信息，你需要参考这些信息对【用户问题】进行解答
    
    【相关信息】
    {referred_item}
    【用户问题】
    {query}
    """
    return llm.invoke(prompt_template).content

@tool
def presale_qa(query: str) -> str:
    """
    处理物流/配送/退换货政策相关的商品外问题

    @:param
    query: 用户问题
    referred_item: 用户引用的衣服物品元数据信息

    :return 大模型返回
    """
    presale_qa = PresaleRetrieval('presale-qa-index')
    context = presale_qa.invoke(query)
    prompt_template = f"""
    你是一个衣物售前客服，请为用户提供物流/配送/退换货政策相关的商品外问题等售前问题答疑服务，
    【相关信息】中会记录了可能相关的qa对供参考，你需要参考这些信息对【用户问题】进行解答
    如果【相关信息】中没有关于【用户问题】中提及的信息，请提示用户没有相关的条款信息

    【相关信息】
    {context}
    【用户问题】
    {query}
    """
    return llm.invoke(prompt_template).content

@tool
def presale_recommendation(query: str) -> str:
    """
    为用户进行商品推荐

    @:param
    query: 用户问题

    :return 大模型返回
    """
    context = PresaleRetrieval().invoke(query)
    prompt_template = f"""
    你是一个衣物售前客服，请根据用户问题进行回答，请参考【相关信息】对于【用户问题】进行答疑
    注意：【相关信息】里为衣物信息，如果你认为衣物没有和【用户问题】相关时，请提示用户没有相关的衣物信息
    【相关信息】
    {context}
    【用户问题】
    {query}
    """
    return llm.invoke(prompt_template).content
