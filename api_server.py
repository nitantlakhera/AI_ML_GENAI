"""Run the FastAPI server with Swagger UI.

Usage:
    uv run python api_server.py
    # or
    uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
