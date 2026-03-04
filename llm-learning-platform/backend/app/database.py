"""
Database Connection and Session Management

Async PostgreSQL connection via SQLModel + asyncpg.
Falls back to SQLite for development without PostgreSQL.
"""

from sqlmodel import SQLModel, create_engine, Session

from app.config import get_settings


def get_engine():
    settings = get_settings()
    db_url = settings.database_url

    # For local dev: fall back to SQLite if no PostgreSQL configured
    if not db_url or db_url == "postgresql+asyncpg://postgres:postgres@db:5432/llm_platform":
        db_url = "sqlite:///./llm_platform.db"

    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(db_url, connect_args=connect_args)


engine = get_engine()


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
