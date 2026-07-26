from mcp_server.tools.example_tool import format_document_summary


def test_format_document_summary():
    result = format_document_summary("Doc", "abcdefghij", max_length=5)
    assert result == "Doc: abcde..."
