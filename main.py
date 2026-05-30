from ragas.messages import HumanMessage

from agent.sale_agent import State, ClothingSaleAgent


def run_conversation():
    state = State(
        messages=[],
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

    agent = ClothingSaleAgent()
    print("客服: 您好！我是您的衣物购物助手，有什么可以帮您的？")
    while True:
        user_input = input("\n用户：").strip()
        if user_input in ['exit', 'quit']:
            print('客服: 感谢您的咨询，再见！')
            break

        state.messages.append(HumanMessage(content=user_input))
        try:
            result = agent.graph.invoke(state)
            print(f"\n客服：{result['llm_response']}")
            state = result
        except Exception as e:
            print(f"处理出错：{e}")
            continue

if __name__ == '__main__':
    run_conversation()
