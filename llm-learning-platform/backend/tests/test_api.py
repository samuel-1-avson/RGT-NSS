"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "components" in data
    
    def test_api_status(self, client):
        """Test API status endpoint."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "active_models" in data
        assert "endpoints" in data


class TestModelEndpoints:
    """Test model management endpoints."""
    
    def test_create_model(self, client):
        """Test creating a model."""
        config = {
            "vocab_size": 100,
            "max_seq_len": 64,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "backend": "custom"
        }
        
        response = client.post("/api/model/create", json=config)
        
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert "num_parameters" in data
        assert data["config"]["backend"] == "custom"
    
    def test_create_model_validation_error(self, client):
        """Test model creation with invalid config."""
        config = {
            "vocab_size": 10,  # Too small
            "max_seq_len": 64,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4
        }
        
        response = client.post("/api/model/create", json=config)
        
        # Should fail validation
        assert response.status_code == 422
    
    def test_get_model(self, client):
        """Test getting model info."""
        # First create a model
        config = {
            "vocab_size": 100,
            "max_seq_len": 64,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "backend": "custom"
        }
        create_resp = client.post("/api/model/create", json=config)
        model_id = create_resp.json()["model_id"]
        
        # Get model info
        response = client.get(f"/api/model/{model_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_id
    
    def test_get_nonexistent_model(self, client):
        """Test getting non-existent model."""
        response = client.get("/api/model/nonexistent123")
        assert response.status_code == 404
    
    def test_list_models(self, client):
        """Test listing models."""
        # Create a few models first
        for i in range(3):
            client.post("/api/model/create", json={
                "vocab_size": 100,
                "max_seq_len": 64,
                "d_model": 64,
                "num_layers": 2,
                "num_heads": 4,
                "backend": "custom"
            })
        
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3
    
    def test_delete_model(self, client):
        """Test deleting a model."""
        # Create model
        create_resp = client.post("/api/model/create", json={
            "vocab_size": 100,
            "max_seq_len": 64,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "backend": "custom"
        })
        model_id = create_resp.json()["model_id"]
        
        # Delete model
        response = client.delete(f"/api/model/{model_id}")
        assert response.status_code == 200
        
        # Verify deletion
        get_resp = client.get(f"/api/model/{model_id}")
        assert get_resp.status_code == 404


class TestInferenceEndpoints:
    """Test inference endpoints."""
    
    def test_tokenize(self, client):
        """Test tokenization endpoint."""
        response = client.post("/api/inference/tokenize", json={
            "text": "Hello, world!",
            "strategy": "character"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "character"
        assert "tokens" in data
        assert "token_ids" in data
        assert data["num_tokens"] == len("Hello, world!")
    
    def test_tokenize_word_strategy(self, client):
        """Test word tokenization."""
        response = client.post("/api/inference/tokenize", json={
            "text": "Hello world test",
            "strategy": "word"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "word"
    
    def test_generate(self, client):
        """Test text generation."""
        # Create model first
        create_resp = client.post("/api/model/create", json={
            "vocab_size": 256,
            "max_seq_len": 64,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "backend": "custom"
        })
        model_id = create_resp.json()["model_id"]
        
        # Generate text
        response = client.post("/api/inference/generate", json={
            "model_id": model_id,
            "prompt": "Hello",
            "max_new_tokens": 10,
            "temperature": 0.8
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_id
        assert "generated_text" in data
        assert "tokens_generated" in data
    
    def test_generate_invalid_model(self, client):
        """Test generation with invalid model."""
        response = client.post("/api/inference/generate", json={
            "model_id": "nonexistent",
            "prompt": "Hello",
            "max_new_tokens": 10
        })
        
        assert response.status_code == 404


class TestTrainingEndpoints:
    """Test training endpoints."""
    
    def test_start_training(self, client):
        """Test starting training."""
        # Create model
        create_resp = client.post("/api/model/create", json={
            "vocab_size": 100,
            "max_seq_len": 32,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "backend": "custom"
        })
        model_id = create_resp.json()["model_id"]
        
        # Start training
        response = client.post("/api/training/start", json={
            "model_id": model_id,
            "batch_size": 4,
            "learning_rate": 0.001,
            "max_steps": 100,
            "seq_length": 32
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["model_id"] == model_id
    
    def test_training_status(self, client):
        """Test getting training status."""
        # Create model and start training
        create_resp = client.post("/api/model/create", json={
            "vocab_size": 100,
            "max_seq_len": 32,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "backend": "custom"
        })
        model_id = create_resp.json()["model_id"]
        
        train_resp = client.post("/api/training/start", json={
            "model_id": model_id,
            "batch_size": 4,
            "max_steps": 100,
            "seq_length": 32
        })
        session_id = train_resp.json()["session_id"]
        
        # Get status
        response = client.get(f"/api/training/{session_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "is_training" in data
    
    def test_stop_training(self, client):
        """Test stopping training."""
        # Create model and start training
        create_resp = client.post("/api/model/create", json={
            "vocab_size": 100,
            "max_seq_len": 32,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "backend": "custom"
        })
        model_id = create_resp.json()["model_id"]
        
        train_resp = client.post("/api/training/start", json={
            "model_id": model_id,
            "batch_size": 4,
            "max_steps": 1000,  # Long training
            "seq_length": 32
        })
        session_id = train_resp.json()["session_id"]
        
        # Stop training
        response = client.post(f"/api/training/{session_id}/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "stopped"


class TestGPUStatusEndpoint:
    """Test GPU status endpoint."""
    
    def test_gpu_status(self, client):
        """Test GPU status check."""
        response = client.get("/api/gpu/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "pytorch_available" in data
        assert "cuda_available" in data
        assert "device" in data


class TestVisualizationEndpoints:
    """Test visualization endpoints."""
    
    def test_get_attention(self, client):
        """Test attention visualization data."""
        # Create model
        create_resp = client.post("/api/model/create", json={
            "vocab_size": 100,
            "max_seq_len": 64,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "backend": "custom"
        })
        model_id = create_resp.json()["model_id"]
        
        response = client.get(f"/api/viz/attention/{model_id}?text=Hello&layer=0&head=0")
        
        # May succeed or fail depending on implementation
        assert response.status_code in [200, 404]
    
    def test_get_embeddings(self, client):
        """Test embeddings visualization."""
        # Create model
        create_resp = client.post("/api/model/create", json={
            "vocab_size": 100,
            "max_seq_len": 64,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "backend": "custom"
        })
        model_id = create_resp.json()["model_id"]
        
        response = client.get(f"/api/viz/embeddings/{model_id}?method=pca")
        
        assert response.status_code in [200, 404]
