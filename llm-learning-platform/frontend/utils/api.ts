import {
  GPTConfig,
  ModelInfo,
  TrainingConfig,
  TrainingStatus,
  TrainingMetrics,
  GenerationRequest,
  GenerationResponse,
  TokenizeRequest,
  TokenizeResponse,
  AttentionData,
  EmbeddingData
} from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

// ==================== HTTP API ====================

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Model Management
  async createModel(config: GPTConfig): Promise<ModelInfo> {
    return this.fetch('/api/model/create', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getModel(modelId: string): Promise<ModelInfo> {
    return this.fetch(`/api/model/${modelId}`);
  }

  async listModels(): Promise<ModelInfo[]> {
    return this.fetch('/api/models');
  }

  async resetModel(modelId: string): Promise<{ status: string }> {
    return this.fetch(`/api/model/${modelId}/reset`, { method: 'POST' });
  }

  async deleteModel(modelId: string): Promise<{ status: string }> {
    return this.fetch(`/api/model/${modelId}`, { method: 'DELETE' });
  }

  // Training
  async startTraining(config: TrainingConfig & { model_id: string }): Promise<{ session_id: string; status: string }> {
    return this.fetch('/api/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async stopTraining(sessionId: string): Promise<{ status: string }> {
    return this.fetch(`/api/training/${sessionId}/stop`, { method: 'POST' });
  }

  async getTrainingStatus(sessionId: string): Promise<TrainingStatus> {
    return this.fetch(`/api/training/${sessionId}/status`);
  }

  async getTrainingHistory(sessionId: string, lastN?: number): Promise<{ session_id: string; metrics: TrainingMetrics[] }> {
    const query = lastN ? `?last_n=${lastN}` : '';
    return this.fetch(`/api/training/${sessionId}/history${query}`);
  }

  // Inference
  async generate(request: GenerationRequest): Promise<GenerationResponse> {
    return this.fetch('/api/inference/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async tokenize(request: TokenizeRequest): Promise<TokenizeResponse> {
    return this.fetch('/api/inference/tokenize', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Visualization
  async getAttentionData(modelId: string, text: string, layer: number = 0, head: number = 0): Promise<AttentionData> {
    const params = new URLSearchParams({ text, layer: layer.toString(), head: head.toString() });
    return this.fetch(`/api/viz/attention/${modelId}?${params}`);
  }

  async getEmbeddings(modelId: string, method: string = 'pca'): Promise<EmbeddingData> {
    return this.fetch(`/api/viz/embeddings/${modelId}?method=${method}`);
  }

  // ====== Standalone Compute Endpoints ======

  async computeAttention(request: {
    text: string;
    d_model?: number;
    num_heads?: number;
    num_layers?: number;
    show_causal_mask?: boolean;
  }): Promise<any> {
    return this.fetch('/api/compute/attention', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async computeEmbeddings(request: {
    text?: string;
    vocab_size?: number;
    embedding_dim?: number;
    seed?: number;
  }): Promise<any> {
    return this.fetch('/api/compute/embeddings', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async computeSampling(request: {
    logits?: number[];
    text?: string;
    temperature?: number;
    top_k?: number;
    top_p?: number;
    vocab_size?: number;
  }): Promise<any> {
    return this.fetch('/api/compute/sampling', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async computeForwardStep(request: {
    text: string;
    d_model?: number;
    num_heads?: number;
    num_layers?: number;
    step?: number;
  }): Promise<any> {
    return this.fetch('/api/compute/forward-step', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async listDatasets(): Promise<any> {
    return this.fetch('/api/compute/datasets');
  }

  async getDataset(datasetId: string): Promise<any> {
    return this.fetch(`/api/compute/datasets/${datasetId}`);
  }

  async uploadDataset(name: string, text: string): Promise<any> {
    const formData = new URLSearchParams();
    formData.append('name', name);
    formData.append('text', text);
    return this.fetch('/api/compute/datasets/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData.toString(),
    });
  }

  async computeBpeTokenize(text: string, numMerges: number = 50): Promise<any> {
    const params = new URLSearchParams({ text, num_merges: numMerges.toString() });
    return this.fetch(`/api/compute/tokenize/bpe?${params}`, { method: 'POST' });
  }

  // Health Check
  async healthCheck(): Promise<{ status: string }> {
    return this.fetch('/health');
  }
}

export const api = new ApiClient(API_URL);

// ==================== WebSocket Service ====================

type MetricsCallback = (metrics: TrainingMetrics) => void;
type StatusCallback = (status: any) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private metricsCallbacks: MetricsCallback[] = [];
  private statusCallbacks: StatusCallback[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect(sessionId: string) {
    this.sessionId = sessionId;
    this.ws = new WebSocket(`${WS_URL}/api/ws/training/${sessionId}`);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'metrics') {
        this.metricsCallbacks.forEach(cb => cb(data.data));
      } else if (data.type === 'status') {
        this.statusCallbacks.forEach(cb => cb(data.data));
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      this.sessionId = null;
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts && this.sessionId) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect... (${this.reconnectAttempts})`);
        this.connect(this.sessionId!);
      }, 1000 * this.reconnectAttempts);
    }
  }

  onMetrics(callback: MetricsCallback) {
    this.metricsCallbacks.push(callback);
    return () => {
      this.metricsCallbacks = this.metricsCallbacks.filter(cb => cb !== callback);
    };
  }

  onStatus(callback: StatusCallback) {
    this.statusCallbacks.push(callback);
    return () => {
      this.statusCallbacks = this.statusCallbacks.filter(cb => cb !== callback);
    };
  }

  sendCommand(command: string, data?: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ command, ...data }));
    }
  }
}

export const wsService = new WebSocketService();
