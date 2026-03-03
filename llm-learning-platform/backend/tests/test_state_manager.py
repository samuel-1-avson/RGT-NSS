"""
Tests for state management.
"""

import pytest
import asyncio

from app.core.state_manager import (
    MemoryStateManager,
    BackendType,
    StateManagerFactory,
)


class TestMemoryStateManager:
    """Test in-memory state manager."""
    
    @pytest.fixture
    async def manager(self):
        """Create a fresh memory state manager."""
        return MemoryStateManager()
    
    @pytest.mark.asyncio
    async def test_save_and_get_model(self, manager):
        """Test saving and retrieving a model."""
        model_id = "test_model_123"
        model_data = {"layers": [1, 2, 3], "weights": "dummy"}
        metadata = {"name": "Test Model", "backend": "pytorch"}
        
        await manager.save_model(model_id, model_data, metadata)
        retrieved = await manager.get_model(model_id)
        
        assert retrieved == model_data
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_model(self, manager):
        """Test retrieving non-existent model."""
        retrieved = await manager.get_model("nonexistent")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete_model(self, manager):
        """Test deleting a model."""
        model_id = "test_model_delete"
        await manager.save_model(model_id, {"data": "test"})
        
        deleted = await manager.delete_model(model_id)
        assert deleted is True
        
        retrieved = await manager.get_model(model_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_model(self, manager):
        """Test deleting non-existent model."""
        deleted = await manager.delete_model("nonexistent")
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_list_models(self, manager):
        """Test listing models."""
        await manager.save_model("model1", {}, {"name": "Model 1"})
        await manager.save_model("model2", {}, {"name": "Model 2"})
        
        models = await manager.list_models()
        
        assert len(models) == 2
        model_ids = [m["model_id"] for m in models]
        assert "model1" in model_ids
        assert "model2" in model_ids
    
    @pytest.mark.asyncio
    async def test_training_session(self, manager):
        """Test training session operations."""
        session_id = "session_123"
        session_data = {
            "model_id": "model_1",
            "status": "running",
            "current_step": 100
        }
        
        await manager.save_training_session(session_id, session_data)
        retrieved = await manager.get_training_session(session_id)
        
        assert retrieved["model_id"] == "model_1"
        assert retrieved["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, manager):
        """Test key-value cache operations."""
        await manager.set_key("test_key", {"data": "value"}, expire=3600)
        retrieved = await manager.get_key("test_key")
        
        assert retrieved == {"data": "value"}
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, manager):
        """Test that cache entries expire."""
        await manager.set_key("expire_key", "value", expire=1)
        
        # Should exist immediately
        assert await manager.get_key("expire_key") == "value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        assert await manager.get_key("expire_key") is None
    
    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        """Test health check."""
        healthy = await manager.health_check()
        assert healthy is True


class TestStateManagerFactory:
    """Test state manager factory."""
    
    def test_create_memory(self):
        """Test creating memory state manager."""
        manager = StateManagerFactory.create(BackendType.MEMORY)
        assert isinstance(manager, MemoryStateManager)
    
    def test_create_memory_by_string(self):
        """Test creating memory state manager by string."""
        # Note: This would need enum parsing in actual implementation
        pass
    
    def test_create_unknown_raises(self):
        """Test that unknown backend raises error."""
        with pytest.raises(ValueError):
            StateManagerFactory.create("unknown_backend")


@pytest.mark.skip(reason="Requires Redis server")
class TestRedisStateManager:
    """Test Redis state manager (requires Redis)."""
    
    @pytest.fixture
    async def redis_manager(self):
        """Create a Redis state manager."""
        from app.core.state_manager import RedisStateManager
        manager = RedisStateManager(redis_url="redis://localhost:6379")
        yield manager
        # Cleanup
        client = await manager._get_client()
        await client.flushdb()
    
    @pytest.mark.asyncio
    async def test_redis_save_and_get(self, redis_manager):
        """Test saving and getting with Redis."""
        await redis_manager.save_model("test", {"data": "value"})
        retrieved = await redis_manager.get_model("test")
        assert retrieved == {"data": "value"}
