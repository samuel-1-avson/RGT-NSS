"""
Database Models

SQLModel-based ORM models for PostgreSQL persistence.
Covers users, sessions, progress, achievements, and model checkpoints.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlmodel import Field, SQLModel, Relationship, Column, JSON


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    username: str = Field(unique=True, index=True, max_length=50)
    display_name: str = Field(max_length=100)
    email: Optional[str] = Field(default=None, unique=True, max_length=255)
    xp: int = Field(default=0)
    level: int = Field(default=1)
    streak_days: int = Field(default=0)
    last_active: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    achievements: List["UserAchievement"] = Relationship(back_populates="user")
    progress: List["ModuleProgress"] = Relationship(back_populates="user")
    sessions: List["TrainingSession"] = Relationship(back_populates="user")


class UserAchievement(SQLModel, table=True):
    __tablename__ = "user_achievements"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id", index=True)
    achievement_id: str = Field(max_length=100)
    awarded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    user: Optional[User] = Relationship(back_populates="achievements")


class ModuleProgress(SQLModel, table=True):
    __tablename__ = "module_progress"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id", index=True)
    module_id: str = Field(max_length=100, index=True)
    status: str = Field(default="not_started", max_length=20)  # not_started, in_progress, completed
    score: Optional[float] = None
    time_spent_seconds: int = Field(default=0)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    user: Optional[User] = Relationship(back_populates="progress")


class TrainingSession(SQLModel, table=True):
    __tablename__ = "training_sessions"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: Optional[str] = Field(default=None, foreign_key="users.id", index=True)
    model_config_json: str = Field(default="{}", sa_column=Column(JSON))
    training_config_json: str = Field(default="{}", sa_column=Column(JSON))
    status: str = Field(default="created", max_length=20)
    current_step: int = Field(default=0)
    total_steps: int = Field(default=0)
    best_loss: float = Field(default=float("inf"))
    final_loss: Optional[float] = None
    metrics_summary: Optional[str] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    user: Optional[User] = Relationship(back_populates="sessions")


class ModelCheckpoint(SQLModel, table=True):
    __tablename__ = "model_checkpoints"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: str = Field(foreign_key="training_sessions.id", index=True)
    step: int = Field(default=0)
    loss: float
    checkpoint_path: str = Field(max_length=500)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
