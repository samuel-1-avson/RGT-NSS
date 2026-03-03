import { GPTConfig } from '@/stores/modelStore';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ApiResponse<T> {
  data?: T;
  error?: string;
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  const url = `${API_URL}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return { 
        error: errorData.detail || `HTTP ${response.status}: ${response.statusText}` 
      };
    }

    const data = await response.json();
    return { data };
  } catch (error: any) {
    console.error(`API Error (${endpoint}):`, error);
    return { error: error.message || 'Network error' };
  }
}

// ============== Model Management API ==============

export interface ModelInfo {
  model_id: string;
  config: GPTConfig;
  num_parameters: number;
  created_at: string;
  status: string;
}

export const modelApi = {
  create: async (config: GPTConfig, name?: string): Promise<ApiResponse<ModelInfo>> => {
    return fetchApi('/api/model/create', {
      method: 'POST',
      body: JSON.stringify({ ...config, name }),
    });
  },

  get: async (modelId: string): Promise<ApiResponse<ModelInfo>> => {
    return fetchApi(`/api/model/${modelId}`);
  },

  list: async (): Promise<ApiResponse<ModelInfo[]>> => {
    return fetchApi('/api/models');
  },

  reset: async (modelId: string): Promise<ApiResponse<any>> => {
    return fetchApi(`/api/model/${modelId}/reset`, {
      method: 'POST',
    });
  },

  delete: async (modelId: string): Promise<ApiResponse<any>> => {
    return fetchApi(`/api/model/${modelId}`, {
      method: 'DELETE',
    });
  },
};

// ============== Training API ==============

export interface TrainingConfig {
  model_id: string;
  dataset: string;
  batch_size: number;
  learning_rate: number;
  min_learning_rate?: number;
  warmup_steps: number;
  max_steps: number;
  grad_clip: number;
  weight_decay?: number;
  optimizer?: 'adam' | 'adamw' | 'sgd';
}

export interface TrainingStatus {
  session_id: string;
  is_training: boolean;
  current_step: number;
  current_epoch: number;
  best_loss: number;
  progress: number;
}

export interface TrainingHistory {
  steps: number[];
  losses: number[];
  perplexities: number[];
  learning_rates: number[];
  grad_norms: number[];
}

export const trainingApi = {
  start: async (config: TrainingConfig): Promise<ApiResponse<{ session_id: string; status: string }>> => {
    return fetchApi('/api/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  stop: async (sessionId: string): Promise<ApiResponse<any>> => {
    return fetchApi(`/api/training/${sessionId}/stop`, {
      method: 'POST',
    });
  },

  getStatus: async (sessionId: string): Promise<ApiResponse<TrainingStatus>> => {
    return fetchApi(`/api/training/${sessionId}/status`);
  },

  getHistory: async (sessionId: string): Promise<ApiResponse<TrainingHistory>> => {
    return fetchApi(`/api/training/${sessionId}/history`);
  },

  saveCheckpoint: async (sessionId: string, name?: string): Promise<ApiResponse<any>> => {
    return fetchApi(`/api/training/${sessionId}/checkpoint`, {
      method: 'POST',
      body: JSON.stringify({ name }),
    });
  },
};

// ============== Inference API ==============

export interface GenerationRequest {
  model_id: string;
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
}

export interface GenerationResponse {
  generated_text: string;
  tokens_generated: number;
  prompt_tokens: number;
  total_tokens: number;
  generation_time: number;
}

export interface TokenizeResponse {
  tokens: number[];
  text_tokens: string[];
  token_count: number;
}

export const inferenceApi = {
  generate: async (request: GenerationRequest): Promise<ApiResponse<GenerationResponse>> => {
    return fetchApi('/api/inference/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  tokenize: async (text: string): Promise<ApiResponse<TokenizeResponse>> => {
    return fetchApi('/api/inference/tokenize', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  },

  forward: async (modelId: string, inputIds: number[]): Promise<ApiResponse<any>> => {
    return fetchApi('/api/inference/forward', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, input_ids: inputIds }),
    });
  },
};

// ============== Visualization API ==============

export interface AttentionData {
  layer: number;
  head: number;
  tokens: string[];
  attention_matrix: number[][];
}

export interface EmbeddingData {
  tokens: string[];
  coordinates: number[][];
  method: string;
}

export const visualizationApi = {
  getAttention: async (modelId: string, layer: number = 0, head: number = 0): Promise<ApiResponse<AttentionData>> => {
    return fetchApi(`/api/viz/attention/${modelId}?layer=${layer}&head=${head}`);
  },

  getEmbeddings: async (modelId: string, method: 'pca' | 'tsne' | 'umap' = 'pca'): Promise<ApiResponse<EmbeddingData>> => {
    return fetchApi(`/api/viz/embeddings/${modelId}?method=${method}`);
  },

  // Compute endpoints for real-time calculations
  computeAttention: async (
    tokens: string[], 
    query: number[][], 
    key: number[][], 
    value: number[][]
  ): Promise<ApiResponse<{ attention_weights: number[][]; output: number[][] }>> => {
    return fetchApi('/api/compute/attention', {
      method: 'POST',
      body: JSON.stringify({ tokens, query, key, value }),
    });
  },

  computeEmbeddings: async (
    tokens: string[],
    vocabSize: number,
    dModel: number
  ): Promise<ApiResponse<{ embeddings: number[][] }>> => {
    return fetchApi('/api/compute/embeddings', {
      method: 'POST',
      body: JSON.stringify({ tokens, vocab_size: vocabSize, d_model: dModel }),
    });
  },
};

// ============== Datasets API ==============

export interface Dataset {
  id: string;
  name: string;
  description: string;
  size: number;
  tokens: number;
  builtin: boolean;
}

export const datasetApi = {
  list: async (): Promise<ApiResponse<{ datasets: Dataset[] }>> => {
    return fetchApi('/api/datasets');
  },

  get: async (datasetId: string): Promise<ApiResponse<Dataset>> => {
    return fetchApi(`/api/datasets/${datasetId}`);
  },

  upload: async (name: string, text: string): Promise<ApiResponse<any>> => {
    return fetchApi('/api/datasets/upload', {
      method: 'POST',
      body: JSON.stringify({ name, text }),
    });
  },
};

// ============== System API ==============

export interface GPUStatus {
  available: boolean;
  device_name?: string;
  total_memory?: number;
  used_memory?: number;
  cuda_version?: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  timestamp: string;
  gpu?: GPUStatus;
}

export const systemApi = {
  health: async (): Promise<ApiResponse<HealthStatus>> => {
    return fetchApi('/health');
  },

  gpuStatus: async (): Promise<ApiResponse<GPUStatus>> => {
    return fetchApi('/api/gpu/status');
  },
};

// ============== Atomic/Educational API ==============

export interface StepResponse {
  loss: number;
  gradients: any;
  activations: any;
}

export const atomicApi = {
  step: async (modelId: string, inputIds: number[], targets: number[]): Promise<ApiResponse<StepResponse>> => {
    return fetchApi('/api/compute/step', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, input_ids: inputIds, targets }),
    });
  },

  forward: async (modelId: string, inputIds: number[]): Promise<ApiResponse<any>> => {
    return fetchApi(`/api/model/${modelId}/forward`, {
      method: 'POST',
      body: JSON.stringify({ input_ids: inputIds }),
    });
  },

  trainStep: async (modelId: string, inputIds: number[], targets: number[], learningRate: number): Promise<ApiResponse<any>> => {
    return fetchApi(`/api/model/${modelId}/train_step`, {
      method: 'POST',
      body: JSON.stringify({ input_ids: inputIds, targets, learning_rate: learningRate }),
    });
  },
};

export default {
  model: modelApi,
  training: trainingApi,
  inference: inferenceApi,
  visualization: visualizationApi,
  dataset: datasetApi,
  system: systemApi,
  atomic: atomicApi,
};
