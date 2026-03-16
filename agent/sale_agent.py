from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel

from tools.presale_tools import presale_metadata, presale_qa, presale_recommendation
from models.llm import qwen_llm as llm

class State(BaseModel):
    messages: list
    referred_item: str
    user_intent: str
    llm_response: str


class ClothingSaleAgent:
    def __init__(self):
        builder = StateGraph(State)
        builder.add_node("intent_classifier", self.intent_classifier)
        builder.add_node("presale_agent", self.presale_agent)
        builder.add_node("aftersale_agent", self.aftersale_agent)
        builder.add_node("chat_agent", self.chat_agent)

        builder.set_entry_point("intent_classifier")

        builder.add_conditional_edges(
            "intent_classifier",
            self.intent_router,
            {
                "chat": "chat_agent",
                "presale": "presale_agent",
                "aftersale": "aftersale_agent",
            }
        )

        builder.add_edge("chat_agent", END)
        builder.add_edge("presale_agent", END)
        builder.add_edge("aftersale_agent", END)

        self.graph = builder.compile()

    def intent_classifier(self, state: State):
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

    def intent_router(self, state: State):
        return state.user_intent

    def presale_agent(self, state: State):
        agent = create_agent(
            llm,
            [presale_metadata,presale_qa, presale_recommendation]
        )
        msg = {
            "messages": [HumanMessage(state.referred_item + '\n\n' + state.messages[-1])]
        }
        resp = agent.invoke(msg)
        resp_str = resp['messages'][-1].content
        return {"llm_response": resp_str}

    def aftersale_agent(self, state: State):
        res = llm.invoke(f"你是一个衣物售后客服，请根据用户问题进行回答：{state.messages[-1]}").content
        return {"llm_response": res}

    def chat_agent(self, state: State):
        res = llm.invoke(f"你是一个衣物客服，用户会和你进行闲聊，请根据用户问题进行回答：{state.messages[-1]}").content
        return {"llm_response": res}

