from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.tools import get_default_tools
from config.settings import AGENT_MAX_ITERATIONS
from rag.llm import get_llm


def build_agent_executor():
    """Build a LangChain agent with default tools."""
    llm = get_llm()
    tools = get_default_tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI agent. Use tools when needed."),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=AGENT_MAX_ITERATIONS,
        handle_parsing_errors=True,
    )
