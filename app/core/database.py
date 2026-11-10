"""Database engine, session management, and base models for SQLAlchemy 2.x and PostgreSQL."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy 2.x entities."""

    pass


class DatabaseSessionManager:
    """Manages SQLAlchemy 2.x async engine and session factory."""

    def __init__(self, database_url: str, echo: bool = False) -> None:
        # Standardize sync postgresql URLs to asyncpg/psycopg async driver if needed
        url = database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        self._engine: AsyncEngine = create_async_engine(
            url,
            echo=echo,
            pool_pre_ping=True,
        )
        self._sessionmaker = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        """Expose the factory to repositories, not individual shared sessions."""

        return self._sessionmaker

    async def close(self) -> None:
        await self._engine.dispose()


def get_database_manager(settings: Settings) -> DatabaseSessionManager:
    """Construct a DatabaseSessionManager from application settings."""
    return DatabaseSessionManager(
        database_url=settings.database_url,
        echo=settings.debug,
    )
