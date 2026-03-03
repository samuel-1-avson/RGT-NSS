// ==================== Model Types ====================

export interface GPTConfig {
  vocab_size: number;
  max_seq_len: number;
  d_model: number;
  num_layers: number;
  num_heads: number;
  d_ff: number;
  dropout: number;
  attention_dropout: number;
  activation: 'gelu' | 'relu' | 'swiglu';
  norm_type: 'rmsnorm' | 'layernorm';
  tie_weights: boolean;
}

export interface ModelInfo {
  model_id: string;
  config: GPTConfig;
  num_parameters: number;
  created_at: string;
  status: string;
}

// ==================== Training Types ====================

export interface TrainingConfig {
  model_id: string;
  batch_size: number;
  learning_rate: number;
  min_learning_rate: number;
  warmup_steps: number;
  max_steps: number;
  grad_clip: number;
  weight_decay: number;
  seq_length: number;
  eval_interval: number;
  checkpoint_interval: number;
}

export interface TrainingMetrics {
  step: number;
  epoch: number;
  loss: number;
  perplexity: number;
  learning_rate: number;
  grad_norm: number;
  tokens_per_sec: number;
  time_elapsed: number;
  time_remaining?: number;
}

export interface TrainingStatus {
  session_id: string;
  is_training: boolean;
  current_step: number;
  current_epoch: number;
  best_loss: number;
  progress: number;
}

// ==================== Inference Types ====================

export interface GenerationRequest {
  model_id: string;
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty: number;
}

export interface GenerationResponse {
  model_id: string;
  prompt: string;
  generated_text: string;
  full_text: string;
  tokens_generated: number;
}

export interface TokenizeRequest {
  text: string;
  strategy: 'character' | 'word' | 'bpe';
}

export interface TokenizeResponse {
  strategy: string;
  text: string;
  tokens: string[];
  token_ids: number[];
  num_tokens: number;
  vocabulary: string[];
  vocab_size?: number;
}

// ==================== Visualization Types ====================

export interface AttentionData {
  model_id: string;
  text: string;
  tokens: string[];
  layer: number;
  head: number;
  attention_matrix: number[][];
  shape: [number, number];
}

export interface EmbeddingData {
  model_id: string;
  method: string;
  vocab_size: number;
  embedding_dim: number;
  projections: number[][];
  tokens: string[];
}

// ==================== UI Types ====================

export interface ModuleRoute {
  id: string;
  title: string;
  description: string;
  icon: string;
  path: string;
}

export interface NavItem {
  label: string;
  href: string;
  icon?: string;
}
