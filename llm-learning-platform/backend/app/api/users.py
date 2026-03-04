"""
Users API

User management, XP/leveling system, achievements, and progress tracking.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory user store
_users: Dict[str, dict] = {}


# XP leveling curve: level N requires N * 100 XP
def _level_for_xp(xp: int) -> int:
    level = 1
    required = 100
    while xp >= required:
        xp -= required
        level += 1
        required = level * 100
    return level


def _xp_for_next_level(xp: int) -> dict:
    level = 1
    remaining = xp
    required = 100
    while remaining >= required:
        remaining -= required
        level += 1
        required = level * 100
    return {
        "current_level": level,
        "xp_in_level": remaining,
        "xp_for_next": required,
        "progress_pct": round(remaining / required * 100, 1),
    }


ACHIEVEMENTS = [
    {"id": "first-token", "title": "First Token", "description": "Tokenize your first text", "xp_bonus": 10},
    {"id": "embedding-explorer", "title": "Space Explorer", "description": "Explore embedding geometry", "xp_bonus": 20},
    {"id": "attention-master", "title": "Attention Master", "description": "Visualize all attention types", "xp_bonus": 50},
    {"id": "model-builder", "title": "Model Builder", "description": "Create your first model", "xp_bonus": 30},
    {"id": "training-complete", "title": "Training Complete", "description": "Complete a full training run", "xp_bonus": 100},
    {"id": "generation-wizard", "title": "Generation Wizard", "description": "Generate text with all sampling methods", "xp_bonus": 50},
    {"id": "speed-learner", "title": "Speed Learner", "description": "Complete 3 modules in one day", "xp_bonus": 200},
    {"id": "completionist", "title": "Completionist", "description": "Complete all modules", "xp_bonus": 500},
    {"id": "streak-7", "title": "Week Warrior", "description": "7-day streak", "xp_bonus": 150},
    {"id": "streak-30", "title": "Monthly Master", "description": "30-day streak", "xp_bonus": 500},
]


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    display_name: Optional[str] = None


class AwardXPRequest(BaseModel):
    xp: int = Field(..., ge=1, le=10000)
    reason: str = ""


class AwardAchievementRequest(BaseModel):
    achievement_id: str


class UpdateProgressRequest(BaseModel):
    module_id: str
    status: str = "completed"  # in_progress, completed
    score: Optional[float] = None


@router.post("")
async def create_user(request: CreateUserRequest):
    """Create a new user profile."""
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    user = {
        "user_id": user_id,
        "username": request.username,
        "display_name": request.display_name or request.username,
        "xp": 0,
        "level": 1,
        "achievements": [],
        "module_progress": {},
        "streak_days": 0,
        "last_active": now,
        "created_at": now,
    }
    _users[user_id] = user
    return user


@router.get("/{user_id}")
async def get_user(user_id: str):
    """Get user profile with level info."""
    if user_id not in _users:
        raise HTTPException(status_code=404, detail="User not found")
    user = _users[user_id]
    level_info = _xp_for_next_level(user["xp"])
    return {**user, "level_info": level_info}


@router.post("/{user_id}/xp")
async def award_xp(user_id: str, request: AwardXPRequest):
    """Award XP to a user."""
    if user_id not in _users:
        raise HTTPException(status_code=404, detail="User not found")
    user = _users[user_id]
    old_level = _level_for_xp(user["xp"])
    user["xp"] += request.xp
    new_level = _level_for_xp(user["xp"])
    user["level"] = new_level
    user["last_active"] = datetime.now(timezone.utc).isoformat()

    return {
        "xp_awarded": request.xp,
        "total_xp": user["xp"],
        "level": new_level,
        "leveled_up": new_level > old_level,
        "level_info": _xp_for_next_level(user["xp"]),
    }


@router.post("/{user_id}/achievements")
async def award_achievement(user_id: str, request: AwardAchievementRequest):
    """Award an achievement to a user."""
    if user_id not in _users:
        raise HTTPException(status_code=404, detail="User not found")

    achievement = next(
        (a for a in ACHIEVEMENTS if a["id"] == request.achievement_id), None
    )
    if not achievement:
        raise HTTPException(status_code=404, detail="Achievement not found")

    user = _users[user_id]
    if request.achievement_id in [a["id"] for a in user["achievements"]]:
        return {"already_awarded": True, "achievement": achievement}

    user["achievements"].append(
        {**achievement, "awarded_at": datetime.now(timezone.utc).isoformat()}
    )
    user["xp"] += achievement["xp_bonus"]
    user["level"] = _level_for_xp(user["xp"])

    return {
        "awarded": True,
        "achievement": achievement,
        "xp_bonus": achievement["xp_bonus"],
        "total_xp": user["xp"],
    }


@router.post("/{user_id}/progress")
async def update_progress(user_id: str, request: UpdateProgressRequest):
    """Update user's progress on a module."""
    if user_id not in _users:
        raise HTTPException(status_code=404, detail="User not found")

    user = _users[user_id]
    user["module_progress"][request.module_id] = {
        "status": request.status,
        "score": request.score,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    user["last_active"] = datetime.now(timezone.utc).isoformat()

    return {
        "module_id": request.module_id,
        "status": request.status,
        "total_completed": sum(
            1 for p in user["module_progress"].values() if p["status"] == "completed"
        ),
    }


@router.get("/{user_id}/progress")
async def get_progress(user_id: str):
    """Get all module progress for a user."""
    if user_id not in _users:
        raise HTTPException(status_code=404, detail="User not found")
    user = _users[user_id]
    return {
        "module_progress": user["module_progress"],
        "completed_count": sum(
            1 for p in user["module_progress"].values() if p["status"] == "completed"
        ),
    }


@router.get("/achievements/list")
async def list_achievements():
    """List all available achievements."""
    return {"achievements": ACHIEVEMENTS}
