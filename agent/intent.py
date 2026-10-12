REPORT_HINTS = (
    "使用报告",
    "生成报告",
    "我的报告",
    "月报",
    "使用记录",
    "个人报告",
    "查询报告",
    "出一份报告",
    "清洁报告",
    "保养报告",
)


def classify_intent(query: str) -> str:
    """规则分流：报告 / 知识问答。拿不准时走问答，避免误出报告。"""
    text = (query or "").replace(" ", "").replace("　", "")
    if any(hint in text for hint in REPORT_HINTS):
        return "report"
    if "报告" in text and any(word in text for word in ("我", "生成", "出", "查", "写")):
        return "report"
    return "qa"
