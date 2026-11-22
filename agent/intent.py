# -*- coding: utf-8 -*-
"""意图识别：先把问题分到五条分支，再决定走哪条链路

为什么用规则而不是模型：
    1. 零依赖、可离线、可解释——面试官问「为什么这里不用模型」，这就是答案：
       **意图分派是架构约束，不是语义理解**。要保证「急救类永远不进检索」，
       靠规则比靠模型稳定：模型可能把「胸痛」判成咨询，规则不会。
    2. 规则法只有 5 个桶，混淆面可控；用模型反而要处理 prompt 漂移、延迟、成本。

难点（面试会问）：一个句子可能同时命中多个意图怎么办？
    答：按「危险优先」排序——急救 > 用药 > 症状 > 就诊准备 > 闲聊。
    宁可把普通问题当急救处理（多给一条急救建议，没有害处），
    也不能把急救当普通咨询（漏掉危险信号，是事故）。
"""
import re

# 意图优先级（数字越小越优先；命中多个时取最小）
# 「挂什么科 / 要带什么」这类问题，主语是就诊流程而不是症状，
# 所以 preparation 要排在 symptom 前面，否则「头晕挂什么科」会被症状抢走。
PRIORITY = {
    "emergency": 0,
    "medication": 1,
    "preparation": 2,
    "symptom": 3,
    "chat": 4,
}

RULES = {
    "emergency": [
        "胸痛", "胸闷", "呼吸困难", "喘不上气", "意识不清", "昏迷", "晕倒", "抽搐",
        "大出血", "止不住血", "止不住", "流血不止", "剧烈头痛", "说话不清", "半身无力", "口角歪斜",
        "高烧不退", "持续高热", "误服", "中毒", "自杀", "割腕",
    ],
    "medication": [
        "吃什么药", "吃啥药", "用什么药", "能吃药", "能不能吃", "可以吃", "该吃什么",
        "剂量", "一天几次", "几片", "副作用", "禁忌", "抗生素", "消炎药", "退烧药",
        "布洛芬", "对乙酰氨基酚", "阿莫西林", "头孢", "连花清瘟", "中药", "西药",
        "怀孕能吃", "哺乳期能吃", "儿童能吃",
    ],
    "symptom": [
        "疼", "痛", "发烧", "发热", "咳嗽", "咳痰", "拉肚子", "腹泻", "呕吐", "恶心",
        "头晕", "头疼", "头痛", "失眠", "睡不着", "皮疹", "起疹子", "瘙痒", "过敏",
        "鼻塞", "流鼻涕", "嗓子", "咽痛", "乏力", "肿胀", "麻木", "心慌", "尿频",
        "尿急", "尿痛", "便秘", "便血", "耳鸣", "眼花",
    ],
    "preparation": [
        "去医院", "挂号", "看哪个科", "挂什么科", "要带什么", "准备什么", "空腹",
        "检查前", "抽血", "拍片", "医保", "报销", "复诊", "住院", "陪诊",
    ],
    "chat": [
        "你好", "谢谢", "你是谁", "你能做什么", "在吗", "早上好", "晚安",
    ],
}

# 编译成正则，避免每个关键词都写一遍
_COMPILED = {
    intent: [re.compile(re.escape(w)) for w in words]
    for intent, words in RULES.items()
}


def classify(query):
    """返回 {intent, matched, candidates, is_mixed}"""
    q = (query or "").strip()
    hits = {}
    for intent, patterns in _COMPILED.items():
        matched = [p.pattern for p in patterns if p.search(q)]
        if matched:
            hits[intent] = matched

    if not hits:
        # 没有关键词命中：不猜闲聊，一律当症状咨询处理，
        # 交给下游的置信度闸门决定要不要追问（闲聊只在命中闲聊词时才走 chat 分支）
        return {"intent": "symptom", "matched": [], "candidates": [], "is_mixed": False}

    ordered = sorted(hits, key=lambda i: PRIORITY[i])
    return {
        "intent": ordered[0],
        "matched": hits[ordered[0]],
        "candidates": ordered,
        "is_mixed": len(ordered) > 1,
    }


def describe(intent):
    return {
        "emergency": "急救判断",
        "medication": "用药咨询",
        "symptom": "症状咨询",
        "preparation": "就诊准备",
        "chat": "闲聊",
    }.get(intent, intent)
