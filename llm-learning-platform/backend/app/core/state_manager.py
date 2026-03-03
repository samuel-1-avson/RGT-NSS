"""
State management with Redis persistence.
Provides persistent storage for models, training sessions, and metadata.
"""

import pickle
import json
import logging
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from app.core.exceptions import StorageError, CacheError


logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Storage backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"


class StateManager(Protocol):
    """Protocol for state management implementations."""
    
    async def save_model(self, model_id: str, model: Any, metadata: Optional[Dict] = None) -> None: ...
    async def get_model(self, model_id: str) -> Optional[Any]: ...
    async def delete_model(self, model_id: str) -> bool: ...
    async def list_models(self, user_id: Optional[str] = None) -> List[Dict]: ...
    
    async def save_training_session(self, session_id: str, session_data: Dict) -> None: ...
    async def get_training_session(self, session_id: str) -> Optional[Dict]: ...
    async def delete_training_session(self, session_id: str) -> bool: ...
    async def list_training_sessions(self, user_id: Optional[str] = None) -> List[Dict]: ...
    
    async def set_key(self, key: str, value: Any, expire: Optional[int] = None) -> None: ...
    async def get_key(self, key: str) -> Optional[Any]: ...
    async def delete_key(self, key: str) -> bool: ...
    
    async def health_check(self) -> bool: ...


class MemoryStateManager:
    """In-memory state manager (development/testing only)."""
    
    def __init__(self):
        self._models: Dict[str, Dict] = {}
        self._training_sessions: Dict[str, Dict] = {}
        self._cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def save_model(self, model_id: str, model: Any, metadata: Optional[Dict] = None) -> None:
        """Save model to memory."""
        async with self._lock:
            self._models[model_id] = {
                "model": model,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
    
    async def get_model(self, model_id: str) -> Optional[Any]:
        """Get model from memory."""
        async with self._lock:
            entry = self._models.get(model_id)
            return entry["model"] if entry else None
    
    async def get_model_metadata(self, model_id: str) -> Optional[Dict]:
        """Get model metadata."""
        async with self._lock:
            entry = self._models.get(model_id)
            if entry:
                return {
                    "model_id": model_id,
                    **entry["metadata"],
                    "created_at": entry["created_at"],
                    "updated_at": entry["updated_at"]
                }
            return None
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model from memory."""
        async with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False
    
    async def list_models(self, user_id: Optional[str] = None) -> List[Dict]:
        """List all models."""
        async with self._lock:
            models = []
            for model_id, entry in self._models.items():
                if user_id is None or entry["metadata"].get("user_id") == user_id:
                    models.append({
                        "model_id": model_id,
                        **entry["metadata"],
                        "created_at": entry["created_at"],
                        "updated_at": entry["updated_at"]
                    })
            return models
    
    async def save_training_session(self, session_id: str, session_data: Dict) -> None:
        """Save training session."""
        async with self._lock:
            self._training_sessions[session_id] = {
                **session_data,
                "updated_at": datetime.utcnow().isoformat()
            }
    
    async def get_training_session(self, session_id: str) -> Optional[Dict]:
        """Get training session."""
        async with self._lock:
            return self._training_sessions.get(session_id)
    
    async def delete_training_session(self, session_id: str) -> bool:
        """Delete training session."""
        async with self._lock:
            if session_id in self._training_sessions:
                del self._training_sessions[session_id]
                return True
            return False
    
    async def list_training_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """List all training sessions."""
        async with self._lock:
            sessions = []
            for session_id, data in self._training_sessions.items():
                if user_id is None or data.get("user_id") == user_id:
                    sessions.append({
                        "session_id": session_id,
                        **data
                    })
            return sessions
    
    async def set_key(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """Set key-value pair."""
        async with self._lock:
            self._cache[key] = {
                "value": value,
                "expires_at": datetime.utcnow() + timedelta(seconds=expire) if expire else None
            }
    
    async def get_key(self, key: str) -> Optional[Any]:
        """Get value by key."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                if entry["expires_at"] and datetime.utcnow() > entry["expires_at"]:
                    del self._cache[key]
                    return None
                return entry["value"]
            return None
    
    async def delete_key(self, key: str) -> bool:
        """Delete key."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def health_check(self) -> bool:
        """Check health."""
        return True


class RedisStateManager:
    """Redis-based state manager for production use."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 model_ttl: int = 86400 * 7,  # 7 days
                 session_ttl: int = 86400 * 1):  # 1 day
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not installed. Install with: pip install redis")
        
        self.redis_url = redis_url
        self.model_ttl = model_ttl
        self.session_ttl = session_ttl
        self._client: Optional[redis.Redis] = None
    
    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=False)
        return self._client
    
    def _serialize(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        return pickle.dumps(obj)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to object."""
        return pickle.loads(data)
    
    async def save_model(self, model_id: str, model: Any, metadata: Optional[Dict] = None) -> None:
        """Save model to Redis."""
        try:
            client = await self._get_client()
            
            # Save model data
            model_key = f"model:{model_id}:data"
            await client.setex(model_key, self.model_ttl, self._serialize(model))
            
            # Save metadata
            meta_key = f"model:{model_id}:metadata"
            meta_data = {
                **(metadata or {}),
                "model_id": model_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            await client.setex(meta_key, self.model_ttl, json.dumps(meta_data))
            
            logger.info(f"Model {model_id} saved to Redis")
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise StorageError(f"Failed to save model: {e}", operation="save_model")
    
    async def get_model(self, model_id: str) -> Optional[Any]:
        """Get model from Redis."""
        try:
            client = await self._get_client()
            model_key = f"model:{model_id}:data"
            data = await client.get(model_key)
            
            if data:
                # Refresh TTL
                await client.expire(model_key, self.model_ttl)
                await client.expire(f"model:{model_id}:metadata", self.model_ttl)
                return self._deserialize(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            raise StorageError(f"Failed to get model: {e}", operation="get_model")
    
    async def get_model_metadata(self, model_id: str) -> Optional[Dict]:
        """Get model metadata."""
        try:
            client = await self._get_client()
            meta_key = f"model:{model_id}:metadata"
            data = await client.get(meta_key)
            
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get model metadata {model_id}: {e}")
            return None
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model from Redis."""
        try:
            client = await self._get_client()
            keys = [f"model:{model_id}:data", f"model:{model_id}:metadata"]
            deleted = await client.delete(*keys)
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    async def list_models(self, user_id: Optional[str] = None) -> List[Dict]:
        """List all models."""
        try:
            client = await self._get_client()
            pattern = "model:*:metadata"
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)
            
            models = []
            for key in keys:
                data = await client.get(key)
                if data:
                    meta = json.loads(data)
                    if user_id is None or meta.get("user_id") == user_id:
                        models.append(meta)
            
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def save_training_session(self, session_id: str, session_data: Dict) -> None:
        """Save training session."""
        try:
            client = await self._get_client()
            key = f"training:{session_id}"
            
            # Merge with existing data if present
            existing = await client.get(key)
            if existing:
                existing_data = self._deserialize(existing)
                existing_data.update(session_data)
                existing_data["updated_at"] = datetime.utcnow().isoformat()
                session_data = existing_data
            else:
                session_data["created_at"] = datetime.utcnow().isoformat()
                session_data["updated_at"] = datetime.utcnow().isoformat()
            
            await client.setex(key, self.session_ttl, self._serialize(session_data))
        except Exception as e:
            logger.error(f"Failed to save training session {session_id}: {e}")
            raise StorageError(f"Failed to save training session: {e}", operation="save_training_session")
    
    async def get_training_session(self, session_id: str) -> Optional[Dict]:
        """Get training session."""
        try:
            client = await self._get_client()
            key = f"training:{session_id}"
            data = await client.get(key)
            
            if data:
                return self._deserialize(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get training session {session_id}: {e}")
            return None
    
    async def delete_training_session(self, session_id: str) -> bool:
        """Delete training session."""
        try:
            client = await self._get_client()
            key = f"training:{session_id}"
            deleted = await client.delete(key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete training session {session_id}: {e}")
            return False
    
    async def list_training_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """List all training sessions."""
        try:
            client = await self._get_client()
            pattern = "training:*"
            sessions = []
            
            async for key in client.scan_iter(match=pattern):
                data = await client.get(key)
                if data:
                    session = self._deserialize(data)
                    if user_id is None or session.get("user_id") == user_id:
                        session["session_id"] = key.decode().split(":")[1] if isinstance(key, bytes) else key.split(":")[1]
                        sessions.append(session)
            
            return sessions
        except Exception as e:
            logger.error(f"Failed to list training sessions: {e}")
            return []
    
    async def set_key(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """Set key-value pair."""
        try:
            client = await self._get_client()
            serialized = self._serialize(value)
            if expire:
                await client.setex(key, expire, serialized)
            else:
                await client.set(key, serialized)
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            raise CacheError(f"Failed to set cache key: {e}")
    
    async def get_key(self, key: str) -> Optional[Any]:
        """Get value by key."""
        try:
            client = await self._get_client()
            data = await client.get(key)
            return self._deserialize(data) if data else None
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None
    
    async def delete_key(self, key: str) -> bool:
        """Delete key."""
        try:
            client = await self._get_client()
            deleted = await client.delete(key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Redis connection."""
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


class StateManagerFactory:
    """Factory for creating state managers."""
    
    @staticmethod
    def create(backend_type: BackendType = BackendType.MEMORY, **kwargs) -> StateManager:
        """Create state manager instance."""
        if backend_type == BackendType.REDIS:
            return RedisStateManager(**kwargs)
        elif backend_type == BackendType.MEMORY:
            return MemoryStateManager()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get global state manager instance."""
    global _state_manager
    if _state_manager is None:
        # Default to memory for development
        _state_manager = MemoryStateManager()
    return _state_manager


def set_state_manager(manager: StateManager) -> None:
    """Set global state manager instance."""
    global _state_manager
    _state_manager = manager
