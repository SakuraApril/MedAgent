# -*- coding: utf-8 -*-
"""护栏：不诊断、不开方、红旗优先、句句带免责。

这一层是医疗 Agent 与普通对话产品的分界线，也是本项目的核心。
"""
import re

DISCLAIMER = "以上为健康信息参考与就医提示，不构成诊断或用药建议；如有不适请以线下医生判断为准。"

# 被认定为「求诊断」的表达
DIAGNOSIS_PAT = [
    r"我(是不是|是不是得|是不是患)", r"是不是(癌|肿瘤|艾滋|白血病)", r"确诊",
    r"我得的?是什么病", r"是什么病", r"你(能)?(确定|判断|诊断)", r"帮我(看看|诊断)",
    r"判断(一下)?(是不是|是)", r"是不是(得|患)了?",
]

# 被认定为「求处方/剂量」的表达
PRESCRIPTION_PAT = [
    r"(该|应该|要)吃(什么|哪种)药", r"开(点|个)?药", r"吃几(片|粒|次)", r"剂量",
    r"(能|可以)不能吃", r"(能|可以)(吃|用).{0,8}吗", r"用什么药", r"输液",
    r"抗生素.*(吃|用)", r"开(点|个)?(抗生素|消炎药|药)", r"吃(什么|哪种|点).{0,8}药",
]

# 结论性表述（输出侧兜底，防止模型越界）
ABSOLUTE_PAT = [
    (r"你(就是|已经)?得了", "可能与以下情况相关"),
    (r"确诊为", "需要医生检查才能判断是否为"),
    (r"肯定是|一定是", "可能是"),
]


def detect_diagnosis(text):
    return any(re.search(p, text) for p in DIAGNOSIS_PAT)


def detect_prescription(text):
    return any(re.search(p, text) for p in PRESCRIPTION_PAT)


def soften(text):
    """输出侧兜底：把绝对化表述改成概率化表述。"""
    for pat, rep in ABSOLUTE_PAT:
        text = re.sub(pat, rep, text)
    return text


def emergency_answer(hits):
    """红旗命中时的固定应答：先救命，其他一律不谈。"""
    names = "、".join(h["name"] for h in hits)
    advice = hits[0]["advice"]
    lines = [
        "⚠️ 你描述的情况属于**需要立即就医**的信号：" + names,
        "",
        "**请立即拨打 120 或前往最近医院急诊**，不要在家观察、不要等待门诊。",
        "",
        "现场处置：" + advice,
        "",
        "在急救人员到达前，尽量保持患者安静、不要进食进水、记录症状开始时间。",
        "",
        DISCLAIMER,
    ]
    return "\n".join(lines)


def refuse_diagnosis():
    return (
        "我不能给你下诊断，也判断不了「是不是某个病」——这需要医生结合查体、化验和影像。\n\n"
        "我能做的是：根据你描述的症状，提示**该挂哪个科**、**哪些情况属于红旗信号**，以及**就诊前该准备什么**。\n\n"
        "你可以补充：症状部位、持续时间、诱因、有没有发热/呕吐、既往病史与正在吃的药。\n\n" + DISCLAIMER
    )


def refuse_prescription():
    return (
        "用药需要医生或药师结合你的病史、过敏史、肝肾功能和正在服用的其他药物来判断，我不能替你开药或给剂量。\n\n"
        "我可以提供**通用名药物的一般注意事项**（例如哪些人慎用、常见相互作用），但具体用不用、用多少，请咨询药师或医生。\n\n"
        + DISCLAIMER
    )


def with_disclaimer(text):
    return text if DISCLAIMER in text else text.rstrip() + "\n\n" + DISCLAIMER
