from agents.base import build_agent_executor


def run_agent(query: str, chat_history: list | None = None) -> str:
    """Execute the agent with a user query."""
    executor = build_agent_executor()
    result = executor.invoke(
        {
            "input": query,
            "chat_history": chat_history or [],
        }
    )
    return result["output"]
