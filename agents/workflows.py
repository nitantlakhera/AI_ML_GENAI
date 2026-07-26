from agents.executor import run_agent


def research_and_summarize(topic: str) -> str:
    """Multi-step workflow: research a topic and produce a summary."""
    research = run_agent(f"Research and list key points about: {topic}")
    summary = run_agent(f"Summarize these points in 3 sentences:\n{research}")
    return summary
