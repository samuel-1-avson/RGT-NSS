import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface GPTConfig {
  vocab_size: number;
  d_model: number;
  num_layers: number;
  num_heads: number;
  d_ff: number;
  max_seq_len: number;
  dropout: number;
  tie_weights?: boolean;
  activation?: 'gelu' | 'relu' | 'swiglu';
  norm_type?: 'rmsnorm' | 'layernorm';
}

export interface TrainingMetrics {
  step: number;
  loss: number;
  perplexity: number;
  learningRate: number;
  gradNorm: number;
  tokensPerSec: number;
}

export interface HistoryData {
  steps: number[];
  losses: number[];
  perplexities: number[];
  learningRates: number[];
  gradNorms: number[];
}

export interface ModelCheckpoint {
  id: string;
  name: string;
  createdAt: string;
  step: number;
  loss: number;
}

export interface ModelState {
  // Model configuration
  config: GPTConfig;
  activeModelId: string | null;
  
  // Training state
  isTraining: boolean;
  currentStep: number;
  loss: number;
  perplexity: number;
  learningRate: number;
  gradNorm: number;
  tokensPerSec: number;
  history: HistoryData;
  checkpoint: ModelCheckpoint | null;
  
  // Actions
  setConfig: (config: Partial<GPTConfig>) => void;
  setActiveModelId: (id: string | null) => void;
  startTraining: () => void;
  stopTraining: () => void;
  updateMetrics: (metrics: TrainingMetrics) => void;
  resetHistory: () => void;
  saveCheckpoint: (name: string) => void;
  loadCheckpoint: (checkpoint: ModelCheckpoint) => void;
  resetState: () => void;
}

export const defaultConfig: GPTConfig = {
  vocab_size: 50257,
  d_model: 768,
  num_layers: 12,
  num_heads: 12,
  d_ff: 3072,
  max_seq_len: 1024,
  dropout: 0.1,
  tie_weights: true,
  activation: 'gelu',
  norm_type: 'layernorm',
};

export interface ModelPreset {
  name: string;
  config: GPTConfig;
  estimatedParams: number;
}

// Model presets for different sizes
export const modelPresets: Record<string, ModelPreset> = {
  micro: {
    name: "Micro GPT",
    config: {
      vocab_size: 256,
      d_model: 64,
      num_layers: 2,
      num_heads: 2,
      d_ff: 256,
      max_seq_len: 128,
      dropout: 0.1,
      tie_weights: true,
      activation: 'gelu',
      norm_type: 'layernorm',
    },
    estimatedParams: 0.5,
  },
  tiny: {
    name: "Tiny GPT",
    config: {
      vocab_size: 256,
      d_model: 128,
      num_layers: 4,
      num_heads: 4,
      d_ff: 512,
      max_seq_len: 256,
      dropout: 0.1,
      tie_weights: true,
      activation: 'gelu',
      norm_type: 'layernorm',
    },
    estimatedParams: 2.1,
  },
  small: {
    name: "Small GPT",
    config: {
      vocab_size: 50257,
      d_model: 256,
      num_layers: 6,
      num_heads: 8,
      d_ff: 1024,
      max_seq_len: 512,
      dropout: 0.1,
      tie_weights: true,
      activation: 'gelu',
      norm_type: 'layernorm',
    },
    estimatedParams: 10,
  },
  medium: {
    name: "Medium GPT",
    config: {
      vocab_size: 50257,
      d_model: 512,
      num_layers: 8,
      num_heads: 8,
      d_ff: 2048,
      max_seq_len: 1024,
      dropout: 0.1,
      tie_weights: true,
      activation: 'gelu',
      norm_type: 'layernorm',
    },
    estimatedParams: 44,
  },
  gpt2_small: {
    name: "GPT-2 Small",
    config: {
      vocab_size: 50257,
      d_model: 768,
      num_layers: 12,
      num_heads: 12,
      d_ff: 3072,
      max_seq_len: 1024,
      dropout: 0.1,
      tie_weights: true,
      activation: 'gelu',
      norm_type: 'layernorm',
    },
    estimatedParams: 124,
  },
};

// Calculate model parameters
export function calculateModelParams(config: GPTConfig): number {
  const embeddingParams = config.vocab_size * config.d_model;
  const positionalParams = config.max_seq_len * config.d_model;
  const layerParams = config.num_layers * (
    // Self-attention: Q, K, V projections + output projection
    (4 * config.d_model * config.d_model) +
    // FFN: two linear layers
    (config.d_model * config.d_ff * 2)
  );
  const lnParams = config.num_layers * 2 * config.d_model; // Layer norms
  const lmHeadParams = config.tie_weights ? 0 : config.vocab_size * config.d_model;
  
  return embeddingParams + positionalParams + layerParams + lnParams + lmHeadParams;
}

// Format parameters count
export function formatParams(params: number): string {
  if (params >= 1_000_000_000) {
    return `${(params / 1_000_000_000).toFixed(1)}B`;
  }
  if (params >= 1_000_000) {
    return `${(params / 1_000_000).toFixed(1)}M`;
  }
  if (params >= 1_000) {
    return `${(params / 1_000).toFixed(1)}K`;
  }
  return `${params}`;
}

// Estimate memory usage
export function estimateMemory(
  config: GPTConfig, 
  precision: 'fp32' | 'fp16' | 'bf16' = 'fp32'
): number {
  const bytesPerParam = precision === 'fp32' ? 4 : 2;
  const params = calculateModelParams(config);
  const modelMemory = params * bytesPerParam;
  
  // Activation memory (rough estimate: ~2x model size for batch=1)
  const activationMemory = modelMemory * 2;
  
  // Optimizer memory (Adam: 2x model size for momentum buffers)
  const optimizerMemory = modelMemory * 2;
  
  // Gradients (same as model size)
  const gradientMemory = modelMemory;
  
  return modelMemory + activationMemory + optimizerMemory + gradientMemory;
}

const initialState = {
  config: defaultConfig,
  activeModelId: null,
  isTraining: false,
  currentStep: 0,
  loss: 0,
  perplexity: 0,
  learningRate: 0.001,
  gradNorm: 0,
  tokensPerSec: 0,
  history: {
    steps: [],
    losses: [],
    perplexities: [],
    learningRates: [],
    gradNorms: [],
  },
  checkpoint: null,
};

export const useModelStore = create<ModelState>()(
  persist(
    (set, get) => ({
      ...initialState,

      setConfig: (config) =>
        set((state) => ({
          config: { ...state.config, ...config },
        })),

      setActiveModelId: (id) =>
        set({ activeModelId: id }),

      startTraining: () =>
        set({ isTraining: true }),

      stopTraining: () =>
        set({ isTraining: false }),

      updateMetrics: (metrics) =>
        set((state) => ({
          currentStep: metrics.step,
          loss: metrics.loss,
          perplexity: metrics.perplexity,
          learningRate: metrics.learningRate,
          gradNorm: metrics.gradNorm,
          tokensPerSec: metrics.tokensPerSec,
          history: {
            steps: [...state.history.steps, metrics.step].slice(-500), // Keep last 500 points
            losses: [...state.history.losses, metrics.loss].slice(-500),
            perplexities: [...state.history.perplexities, metrics.perplexity].slice(-500),
            learningRates: [...state.history.learningRates, metrics.learningRate].slice(-500),
            gradNorms: [...state.history.gradNorms, metrics.gradNorm].slice(-500),
          },
        })),

      resetHistory: () =>
        set({
          currentStep: 0,
          loss: 0,
          perplexity: 0,
          history: {
            steps: [],
            losses: [],
            perplexities: [],
            learningRates: [],
            gradNorms: [],
          },
        }),

      saveCheckpoint: (name) => {
        const checkpoint: ModelCheckpoint = {
          id: `checkpoint-${Date.now()}`,
          name,
          createdAt: new Date().toISOString(),
          step: get().currentStep,
          loss: get().loss,
        };
        set({ checkpoint });
      },

      loadCheckpoint: (checkpoint) => {
        set({ checkpoint });
      },

      resetState: () => set(initialState),
    }),
    {
      name: 'llm-learning-storage',
      partialize: (state) => ({
        config: state.config,
        checkpoint: state.checkpoint,
      }),
    }
  )
);

export default useModelStore;
