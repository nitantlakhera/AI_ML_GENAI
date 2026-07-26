from langchain_classic.memory import ConversationBufferMemory


def get_memory():
    return ConversationBufferMemory(
        memory_key="history",
        return_messages=False,
    )
