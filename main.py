"""
ChronoStream RAG — Entry Point.
Launches the FastAPI server with auto-starting continuous stream ingestion.
"""
import uvicorn
from app.config import settings


def main():
    uvicorn.run(
        "app.api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )


if __name__ == "__main__":
    main()
