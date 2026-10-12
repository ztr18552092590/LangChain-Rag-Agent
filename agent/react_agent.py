import random
import uuid
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.config_handler import agent_conf
from agent.intent import classify_intent
from agent.runtime_ctx import set_session, reset_session
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report,
                                     user_ids, cities)
from agent.checkpointer import get_checkpointer
from agent.tools.middleware import (
    monitor_tool,
    trim_checkpoint_messages,
    log_before_model,
    report_prompt_switch,
)


def build_input_messages(query: str, history: list[dict] | None = None, max_turns: int | None = None) -> list[dict]:
    """把界面历史收成 Agent 输入：只保留 user/assistant，截断窗口，当前问题只出现一次。"""
    if max_turns is None:
        max_turns = agent_conf.get("max_history_turns", 6)

    cleaned: list[dict] = []
    for item in history or []:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            cleaned.append({"role": role, "content": content})

    query = (query or "").strip()
    if cleaned and cleaned[-1]["role"] == "user" and cleaned[-1]["content"] == query:
        cleaned = cleaned[:-1]

    max_msgs = max(0, int(max_turns) * 2)
    if max_msgs and len(cleaned) > max_msgs:
        cleaned = cleaned[-max_msgs:]
        if cleaned and cleaned[0]["role"] == "assistant":
            cleaned = cleaned[1:]

    cleaned.append({"role": "user", "content": query})
    return cleaned


def _is_final_answer(message) -> bool:
    if isinstance(message, ToolMessage):
        return False
    if not isinstance(message, AIMessage):
        return False
    if getattr(message, "tool_calls", None):
        return False
    content = message.content
    return bool(content and str(content).strip())


class ReactAgent:
    def __init__(self):
        self.session_user_id = random.choice(user_ids)
        self.session_city = random.choice(cities)
        self.default_thread_id = str(uuid.uuid4())
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[
                monitor_tool,
                trim_checkpoint_messages,
                log_before_model,
                report_prompt_switch,
            ],
            checkpointer=get_checkpointer(),
        )

    def _checkpoint_size(self, thread_id: str) -> int:
        try:
            snapshot = self.agent.get_state({"configurable": {"thread_id": thread_id}})
            values = getattr(snapshot, "values", None) or {}
            return len(values.get("messages") or [])
        except Exception:
            return 0

    def execute_stream(
        self,
        query: str,
        history: list[dict] | None = None,
        thread_id: str | None = None,
    ):
        yielded_ids: set[str] = set()
        intent = classify_intent(query)
        token = set_session({
            "user_id": self.session_user_id,
            "city": self.session_city,
        })

        # checkpointer 要求每次都带 thread_id；只传当前句，避免和库里历史重复
        thread_id = thread_id or self.default_thread_id
        input_messages = [{"role": "user", "content": (query or "").strip()}]
        input_len = self._checkpoint_size(thread_id)
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # context 每轮重建，report 开关不写入 checkpointer
            for chunk in self.agent.stream(
                {"messages": input_messages},
                config=config,
                stream_mode="values",
                context={
                    "report": False,
                    "intent": intent,
                    "tool_count": 0,
                    "rag_uncovered": False,
                    "fetched_data": False,
                    "rag_queries": [],
                },
            ):
                for message in chunk["messages"][input_len:]:
                    if not _is_final_answer(message):
                        continue
                    message_id = getattr(message, "id", None) or str(id(message))
                    if message_id in yielded_ids:
                        continue
                    yielded_ids.add(message_id)
                    yield str(message.content).strip() + "\n"
        finally:
            reset_session(token)


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)