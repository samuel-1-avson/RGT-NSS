import axios from 'axios';
import { GPTConfig } from '@/stores/modelStore';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Request interceptor for auth tokens if needed
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('api_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Model Management API
export const modelApi = {
  create: async (config: GPTConfig, name?: string) => {
    const response = await api.post('/model/create', { config, name });
    return response.data;
  },

  get: async (modelId: string) => {
    const response = await api.get(`/model/${modelId}`);
    return response.data;
  },

  list: async () => {
    const response = await api.get('/model/list');
    return response.data;
  },

  reset: async (modelId: string) => {
    const response = await api.post(`/model/${modelId}/reset`);
    return response.data;
  },

  delete: async (modelId: string) => {
    const response = await api.delete(`/model/${modelId}`);
    return response.data;
  },

  updateConfig: async (modelId: string, config: Partial<GPTConfig>) => {
    const response = await api.put(`/model/${modelId}/config`, config);
    return response.data;
  },
};

// Training API
export const trainingApi = {
  start: async (config: {
    model_id: string;
    dataset: string;
    batch_size: number;
    learning_rate: number;
    warmup_steps: number;
    max_steps: number;
    grad_clip: number;
  }) => {
    const response = await api.post('/training/start', config);
    return response.data;
  },

  stop: async (sessionId: string) => {
    const response = await api.post(`/training/${sessionId}/stop`);
    return response.data;
  },

  getStatus: async (sessionId: string) => {
    const response = await api.get(`/training/${sessionId}/status`);
    return response.data;
  },

  getHistory: async (sessionId: string) => {
    const response = await api.get(`/training/${sessionId}/history`);
    return response.data;
  },

  saveCheckpoint: async (sessionId: string, name?: string) => {
    const response = await api.post(`/training/${sessionId}/checkpoint`, { name });
    return response.data;
  },

  listCheckpoints: async (modelId: string) => {
    const response = await api.get(`/training/checkpoints/${modelId}`);
    return response.data;
  },
};

// Inference API
export const inferenceApi = {
  generate: async (request: {
    model_id: string;
    prompt: string;
    max_new_tokens: number;
    temperature: number;
    top_k?: number;
    top_p?: number;
  }) => {
    const response = await api.post('/inference/generate', request);
    return response.data;
  },

  tokenize: async (text: string) => {
    const response = await api.post('/inference/tokenize', { text });
    return response.data;
  },

  forward: async (modelId: string, inputIds: number[]) => {
    const response = await api.post('/inference/forward', {
      model_id: modelId,
      input_ids: inputIds,
    });
    return response.data;
  },
};

// Visualization API
export const visualizationApi = {
  getAttention: async (modelId: string, layer: number, head: number) => {
    const response = await api.get(`/viz/attention/${modelId}`, {
      params: { layer, head },
    });
    return response.data;
  },

  getEmbeddings: async (modelId: string, method: 'pca' | 'tsne' | 'umap' = 'pca') => {
    const response = await api.get(`/viz/embeddings/${modelId}`, {
      params: { method },
    });
    return response.data;
  },

  getGradients: async (modelId: string) => {
    const response = await api.get(`/viz/gradients/${modelId}`);
    return response.data;
  },

  getActivations: async (modelId: string, layer: number) => {
    const response = await api.get(`/viz/activations/${modelId}`, {
      params: { layer },
    });
    return response.data;
  },
};

// Datasets API
export const datasetApi = {
  list: async () => {
    const response = await api.get('/datasets');
    return response.data;
  },

  get: async (datasetId: string) => {
    const response = await api.get(`/datasets/${datasetId}`);
    return response.data;
  },

  upload: async (name: string, text: string) => {
    const response = await api.post('/datasets/upload', { name, text });
    return response.data;
  },
};

// Health check
export const healthApi = {
  check: async () => {
    const response = await api.get('/health');
    return response.data;
  },
  
  getGPUStatus: async () => {
    const response = await api.get('/health/gpu');
    return response.data;
  },
};

export default api;
