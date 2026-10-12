from contextvars import ContextVar, Token

_session: ContextVar[dict] = ContextVar("agent_session", default={})


def set_session(data: dict) -> Token:
    return _session.set(data)


def reset_session(token: Token) -> None:
    _session.reset(token)


def get_session() -> dict:
    return _session.get() or {}
