"""Entrypoint for rag-mentor-platform service.

This minimal `main.py` imports the FastAPI app from the API routes and
exposes it for Uvicorn. Adjust as needed after reviewing the scaffold.
"""

from rag_mentor_platform.api.routes.chat import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
