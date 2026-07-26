from config.settings import (
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    USE_API_LLM,
)


def get_llm():
    if USE_API_LLM:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=OPENAI_API_KEY, temperature=LLM_TEMPERATURE)

    from langchain_community.llms import LlamaCpp

    return LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_N_CTX,
        temperature=LLM_TEMPERATURE,
        verbose=False,
    )
