from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
        builder.add_node("apply_sliding_window", self.apply_sliding_window)

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

        builder.add_edge("chat_agent", "apply_sliding_window")
        builder.add_edge("presale_agent", "apply_sliding_window")
        builder.add_edge("aftersale_agent", "apply_sliding_window")

        builder.add_edge("apply_sliding_window", END)

        self.graph = builder.compile()

    def intent_classifier(self, state: State):
        intent_prompt_template = f"""
        你是一个销售衣物的智能客服，你需要根据用户问题进行问题的分类
        
        详细规则：
        1. 【用户问题】中的意图，分为三种类型：闲聊、售前，售后
        2. 意图中英文对照表：闲聊（chat）、售前（presale）、售后（aftersale）
        3. 你需要输出【用户问题】中的意图，请用意图（闲聊/售前/售后）对应的英文输出
        4. 你只能根据意图输出一个单词：chat/presale/aftersale
        【最近对话】
        {self._build_context(state.messages)}
        【用户问题】
        {state.messages[-1].content}
        """
        intent_prompt = ChatPromptTemplate([HumanMessage(intent_prompt_template)])
        intent = llm.invoke(intent_prompt.messages).content

        return {"user_intent": intent}

    def intent_router(self, state: State):
        return state.user_intent

    def presale_agent(self, state: State):
        agent = create_agent(
            llm,
            [presale_metadata,presale_qa, presale_recommendation],
            system_prompt=SystemMessage(
                "你是一个售前智能助理主管，请选择合适的工具调用并且润色每个工具返回的内容。\n"
                f"最近的上下文：\n{self._build_context(state.messages)}"
                "请基于以上上下文提供连贯的回答。"
            )
        )
        msg = {
            "messages": [HumanMessage(state.referred_item + '\n\n' + state.messages[-1].content)]
        }
        resp = agent.invoke(msg)
        resp_str = resp['messages'][-1].content
        return {
            "llm_response": resp_str,
            "messages": state.messages + [AIMessage(resp_str)]
        }

    def aftersale_agent(self, state: State):
        res_str = llm.invoke(f"你是一个衣物售后客服，请根据用户问题进行回答：{state.messages[-1].content}").content
        return {
            "llm_response": res_str,
            "messages": state.messages + [AIMessage(res_str)]
        }

    def chat_agent(self, state: State):
        res_str = llm.invoke(f"你是一个衣物客服，用户会和你进行闲聊，请根据用户问题进行回答：{state.messages[-1]}").content
        return {
            "llm_response": res_str,
            "messages": state.messages + [AIMessage(res_str)]
        }

    def apply_sliding_window(self, state: State):
        WINDOW_SIZE = 3
        if len(state.messages) >= 2 * WINDOW_SIZE:
            return {"messages": state.messages[-WINDOW_SIZE*2:]}
        return {}

    def _build_context(self, messages: list) -> str:
        context = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context += f"用户：{msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"客服：{msg.content}\n"
        return context
