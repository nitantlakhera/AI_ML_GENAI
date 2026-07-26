from chat.memory import get_memory
from rag.llm import get_llm


class ChatBot:
    """Simple conversational chatbot with memory."""

    def __init__(self):
        self.llm = get_llm()
        self.memory = get_memory()

    def chat(self, message: str) -> str:
        history = self.memory.load_memory_variables({})
        prompt = self._build_prompt(message, history.get("history", ""))
        response = self.llm.invoke(prompt)
        self.memory.save_context({"input": message}, {"output": response})
        return response

    def _build_prompt(self, message: str, history: str) -> str:
        return (
            "You are a friendly chatbot.\n"
            f"History:\n{history}\n\n"
            f"User: {message}\n"
            "Assistant:"
        )
