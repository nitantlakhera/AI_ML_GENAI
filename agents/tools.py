from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression. Example: '2 + 2'."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Invalid expression."
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error: {exc}"


@tool
def word_count(text: str) -> str:
    """Count words in the given text."""
    return str(len(text.split()))


def get_default_tools():
    return [calculator, word_count]
