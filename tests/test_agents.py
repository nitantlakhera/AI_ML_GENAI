from agents.tools import calculator, word_count, get_default_tools


def test_calculator():
    assert calculator.invoke("2 + 2") == "4"


def test_word_count():
    assert word_count.invoke("hello world") == "2"


def test_default_tools_count():
    assert len(get_default_tools()) >= 2
