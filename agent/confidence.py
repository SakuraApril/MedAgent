# -*- coding: utf-8 -*-
"""置信度闸门：回答之前先判断「有没有资格回答」

为什么需要它：
    原来的护栏只管「命中红旗就短路」，那是「已知的危险」；
    但还有一类更隐蔽的风险——**检索没命中、症状没识别出来**，
    系统却照样给了一个科室建议。这类回答看起来正常，实际上没有依据。

三个信号 + 一个风险修正：
    E 证据分  检索 top1 相似度（0.35 视为满分参照）
    C 覆盖分  查询词在召回文档里的命中比例
    S 识别分  分诊是否识别到具体症状（没识别出来 = 没听懂）
    R 风险修正 涉及用药 / 求诊断时，门槛抬高（更保守）

三档动作：
    answer      正常回答
    caveat      回答 + 顶部提示「信息有限」
    ask_more    不给结论，改为追问或建议就医
"""
import re

from .guard import detect_diagnosis, detect_prescription

# 权重（合计 1.0）：识别出症状最重要，其次是检索证据
W_EVIDENCE = 0.25
W_COVERAGE = 0.20
W_SPECIFIC = 0.55

# 阈值
T_PASS = 0.55       # ≥ 正常回答
T_CAVEAT = 0.35     # ≥ 回答但加提示；< 则追问
RISK_PENALTY = 0.10  # 涉及用药/诊断时，实际门槛抬高

# 检索相似度的「满分参照」：top1 达到这个值算证据充分
EVIDENCE_REF = 0.35

# 计算覆盖分时忽略的高频虚词
STOP_WORDS = {"怎么办", "怎么", "什么", "可以", "能不能", "要不要", "如何", "一下", "最近", "感觉"}


def _query_terms(query):
    """把查询拆成用于覆盖度计算的词：中文 2-gram + 英文/数字词，去掉虚词"""
    q = re.sub(r"\s+", "", (query or "").lower())
    for w in STOP_WORDS:
        q = q.replace(w, "")
    terms = set()
    for seg in re.findall(r"[\u4e00-\u9fff]+", q):
        if len(seg) <= 2:
            terms.add(seg)
        else:
            for i in range(len(seg) - 1):
                terms.add(seg[i:i + 2])
    terms.update(re.findall(r"[a-z]+|\d+(?:\.\d+)?", q))
    return terms


def _coverage(terms, docs):
    if not terms or not docs:
        return 0.0
    blob = "".join((d.get("title", "") + d.get("text", "")) for d in docs).lower()
    hit = sum(1 for t in terms if t in blob)
    return hit / len(terms)


def evaluate(query, triage, docs, drugs):
    """返回 {total, parts, action, threshold}"""
    evidence = 0.0
    if docs:
        top = float(docs[0].get("score", 0.0))
        evidence = min(1.0, top / EVIDENCE_REF)

    coverage = _coverage(_query_terms(query), docs)
    specific = 1.0 if (triage and triage.get("matched")) else 0.0

    total = W_EVIDENCE * evidence + W_COVERAGE * coverage + W_SPECIFIC * specific

    # 两道封顶：没听懂症状，最多只到「加提示」；没有检索证据，最多只到「追问」
    if not specific:
        total = min(total, 0.34)      # 没听懂症状 → 只能追问
    elif not docs:
        total = min(total, 0.6)       # 听懂了但没检索到资料 → 加提示后回答

    risky = bool(drugs) or detect_diagnosis(query) or detect_prescription(query)
    threshold = T_PASS + (RISK_PENALTY if risky else 0.0)

    if total >= threshold:
        action = "answer"
    elif total >= T_CAVEAT:
        action = "caveat"
    else:
        action = "ask_more"

    return {
        "total": round(total, 3),
        "threshold": round(threshold, 3),
        "action": action,
        "risky": risky,
        "parts": {
            "evidence": round(evidence, 3),
            "coverage": round(coverage, 3),
            "specific": specific,
        },
    }


def ask_more_answer(missing_hint=""):
    """低置信度时的应答：不给结论，给「补什么信息」"""
    base = (
        "你描述的信息还不够，我不能凭这几句话给出科室建议——猜错比不说更麻烦。\n\n"
        "**请补充这几点，我再判断**\n"
        "· 具体哪里不舒服（部位、什么样的疼/痒/胀）\n"
        "· 从什么时候开始、持续多久了\n"
        "· 有没有发热、呕吐、皮疹、呼吸困难等伴随症状\n"
        "· 既往病史，以及正在吃的药\n"
    )
    if missing_hint:
        base += f"\n（另外：{missing_hint}）\n"
    base += "\n**如果症状明显、加重很快，或出现胸痛、呼吸困难、意识不清，请直接急诊或拨打 120。**"
    return base


def caveat_line(score):
    return (f"> ⚠ 信息有限（置信度 {score['total']:.2f}）：以下建议基于较少线索，"
            f"仅供就诊参考，请以医生现场判断为准。")
