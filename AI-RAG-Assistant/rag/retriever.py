from config.settings import TOP_K


def get_retriever(vector_store):
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})
