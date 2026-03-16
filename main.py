from agent.sale_agent import State, ClothingSaleAgent

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

agent = ClothingSaleAgent()
result = agent.graph.invoke(initial_state)

print(result)
