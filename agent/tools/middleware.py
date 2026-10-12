from typing import Callable
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import RemoveMessage, ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger
from utils.config_handler import agent_conf
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from utils.prompt_loader import load_system_prompts, load_report_prompts

QA_TOOLS = {"rag_summarize", "get_weather", "get_user_location"}
REPORT_TOOLS = {
    "get_user_id",
    "get_current_month",
    "fill_context_for_report",
    "fetch_external_data",
    "rag_summarize",
}


def _tool_message(request: ToolCallRequest, content: str) -> ToolMessage:
    tool_call = request.tool_call
    return ToolMessage(
        content=content,
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )


def _message_text(result: ToolMessage | Command) -> str:
    if isinstance(result, ToolMessage):
        return str(result.content or "")
    return ""


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    ctx = request.runtime.context
    name = request.tool_call["name"]
    args = request.tool_call.get("args", {})
    intent = ctx.get("intent", "qa")
    max_calls = int(agent_conf.get("max_tool_calls", 6))

    logger.info(f"[tool monitor]执行工具: {name}")
    logger.info(f"[tool monitor]传入参数: {args}")

    ctx["tool_count"] = int(ctx.get("tool_count", 0)) + 1
    if ctx["tool_count"] > max_calls:
        logger.warning(f"[tool monitor]超过工具次数上限 {max_calls}，拒绝 {name}")
        return _tool_message(request, f"工具调用已达上限（{max_calls}次），请基于已有信息直接回答用户，不要再调用工具。")

    allowed = REPORT_TOOLS if intent == "report" else QA_TOOLS
    if name not in allowed:
        logger.warning(f"[tool monitor]意图={intent}，拒绝工具 {name}")
        if intent == "report":
            return _tool_message(
                request,
                "当前是使用报告任务。请按「用户ID→月份→fill_context_for_report→fetch_external_data」执行，需要保养建议时再检索。",
            )
        return _tool_message(
            request,
            "当前是知识问答，不能查询个人使用记录或切换报告模式。请使用 rag_summarize，或在需要环境信息时使用定位/天气工具。",
        )

    if name == "fetch_external_data" and not ctx.get("report"):
        logger.warning("[tool monitor]未 fill_context 即 fetch，已拦截")
        return _tool_message(request, "请先调用 fill_context_for_report，再获取使用记录。")

    if intent == "report" and name == "rag_summarize" and not ctx.get("fetched_data"):
        logger.warning("[tool monitor]报告流程尚未拉到使用记录，拒绝检索")
        return _tool_message(request, "请先获取用户使用记录，再检索保养或使用建议。")

    if name == "rag_summarize" and ctx.get("rag_uncovered"):
        logger.warning("[tool monitor]知识库已未覆盖，拒绝再次检索")
        return _tool_message(request, "知识库已确认未覆盖，请直接根据已有信息回答，不要再次检索。")

    if name == "rag_summarize":
        query = str(args.get("query", "")).strip()
        seen = ctx.setdefault("rag_queries", [])
        if query and query in seen:
            logger.warning(f"[tool monitor]重复检索词，拒绝: {query}")
            return _tool_message(request, "该检索词已经查过，请基于已有资料作答，或更换完全不同的检索词。")
        if query:
            seen.append(query)

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{name}调用成功")

        if name == "fill_context_for_report":
            ctx["report"] = True

        if name == "fetch_external_data":
            ctx["fetched_data"] = True

        text = _message_text(result)
        if name == "rag_summarize" and "【知识库未覆盖】" in text:
            ctx["rag_uncovered"] = True

        return result
    except Exception as e:
        logger.error(f"工具{name}调用失败，原因: {str(e)}")
        raise e


@before_model
def trim_checkpoint_messages(state: AgentState, runtime: Runtime):
    """避免 checkpointer 把工具轨迹无限堆长。不持久化 report 开关。"""
    messages = state.get("messages") or []
    max_n = int(agent_conf.get("max_checkpoint_messages", 40))
    if len(messages) <= max_n:
        return None

    drop = list(messages[:-max_n])
    keep = list(messages[-max_n:])
    while keep and isinstance(keep[0], ToolMessage):
        drop.append(keep.pop(0))

    removed = [RemoveMessage(id=m.id) for m in drop if getattr(m, "id", None)]
    if not removed:
        return None
    logger.info(f"[trim]会话消息 {len(messages)} -> {len(keep)}，丢掉 {len(removed)} 条")
    return {"messages": removed}


@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime,
):
    last = state["messages"][-1]
    content = getattr(last, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    logger.debug(f"[log_before_model]{type(last).__name__} | {content.strip()}")
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()

    prompt = load_system_prompts()
    if request.runtime.context.get("intent") == "report":
        prompt += (
            "\n\n【本轮意图=使用报告】系统已判定用户要生成/查询个人使用报告。"
            "必须按「get_user_id→get_current_month（用户指定月份则用指定值）"
            "→fill_context_for_report→fetch_external_data」执行；拿到记录前不要检索知识库。"
        )
    return prompt
