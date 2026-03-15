from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel

from load_api_key import get_api_key
from sale_presale_tools import presale_metadata, presale_qa, presale_recommendation

api_key = get_api_key(llm_type="DASHSCOPE_API_KEY")

llm = ChatQwen(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30,
)

class State(BaseModel):
    messages: list
    referred_item: str
    user_intent: str
    llm_response: str

def intent_classifier(state: State):
    intent_prompt_template = f"""
    你是一个销售衣物的智能客服，你需要根据用户问题进行问题的分类
    
    详细规则：
    1. 【用户问题】中的意图，分为三种类型：闲聊、售前，售后
    2. 意图中英文对照表：闲聊（chat）、售前（presale）、售后（aftersale）
    3. 你需要输出【用户问题】中的意图，请用意图（闲聊/售前/售后）对应的英文输出
    4. 你只能根据意图输出一个单词：chat/presale/aftersale
    
    【用户问题】
    {state.messages[-1]}
    """
    intent_prompt = ChatPromptTemplate([HumanMessage(intent_prompt_template)])
    intent = llm.invoke(intent_prompt.messages).content

    return {"user_intent": intent}

def intent_router(state: State):
    return state.user_intent

def presale_agent(state: State):
    presale_agent = create_agent(
        llm,
        [presale_metadata,presale_qa, presale_recommendation]
    )
    msg = {
        "messages": [HumanMessage(state.referred_item + '\n\n' + state.messages[-1])]
    }
    resp = presale_agent.invoke(msg)
    resp_str = resp['messages'][-1].content
    return {"llm_response": resp_str}

def aftersale_agent(state: State):
    res = llm.invoke(f"你是一个衣物售后客服，请根据用户问题进行回答：{state.messages[-1]}").content
    return {"llm_response": res}

def chat_agent(state: State):
    res = llm.invoke(f"你是一个衣物客服，用户会和你进行闲聊，请根据用户问题进行回答：{state.messages[-1]}").content
    return {"llm_response": res}


builder = StateGraph(State)
builder.add_node("intent_classifier", intent_classifier)
builder.add_node("presale_agent", presale_agent)
builder.add_node("aftersale_agent", aftersale_agent)
builder.add_node("chat_agent", chat_agent)

builder.set_entry_point("intent_classifier")

builder.add_conditional_edges(
    "intent_classifier",
    intent_router,
    {
        "chat": "chat_agent",
        "presale": "presale_agent",
        "aftersale": "aftersale_agent",
    }
)

builder.add_edge("chat_agent", END)
builder.add_edge("presale_agent", END)
builder.add_edge("aftersale_agent", END)

graph = builder.compile()

initial_state = State(
    messages=['你好，我163cm，选择什么尺码合适？'],
    referred_item="""
        产品ID: TP001
        性别：女
        名称: 纯棉印花短袖T恤
        价格: ¥89
        颜色: 白色、黑色、樱花粉
        尺码: XS/S/M/L/XL
        尺码建议: XS(150-158cm/40-48kg)、S(155-163cm/45-53kg)、M(160-168cm/52-62kg)、L(165-173cm/60-70kg)、XL(168-176cm/68-80kg)
        材质: 100%棉
        适用季节: 夏
        风格: 休闲、简约
        库存: 白色(100件)、黑色(80件)、樱花粉(60件)
        特点: 纯棉透气，卡通印花可爱减龄，百搭款
        适用场合: 日常、出游、居家
    """,
    user_intent="",
    llm_response=""
)

result = graph.invoke(initial_state)

print(result)
