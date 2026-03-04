"""
Database Connection and Session Management

Async PostgreSQL connection via SQLAlchemy + asyncpg.
Falls back to async SQLite (aiosqlite) for development without PostgreSQL.
"""

from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import get_settings


def _get_database_url() -> str:
    """Resolve the database URL, falling back to SQLite for local dev."""
    settings = get_settings()
    db_url = settings.database_url

    # For local dev: fall back to async SQLite if default Docker URL is used
    if not db_url or db_url == "postgresql+asyncpg://postgres:postgres@db:5432/llm_platform":
        db_url = "sqlite+aiosqlite:///./llm_platform.db"

    return db_url


engine = create_async_engine(
    _get_database_url(),
    echo=False,
    future=True,
)

async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def create_db_and_tables():
    """Create all tables from SQLModel metadata (async)."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session():
    """Async dependency that yields an AsyncSession."""
    async with async_session() as session:
        yield session
