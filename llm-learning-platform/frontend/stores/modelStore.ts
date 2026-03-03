import { create } from 'zustand';
import { persist } from 'zustand/middleware';

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
  norm_type: 'layernorm' | 'rmsnorm';
  tie_weights: boolean;
}

export interface TrainingMetrics {
  step: number;
  loss: number;
  perplexity: number;
  learning_rate: number;
  grad_norm: number;
  tokens_per_sec?: number;
  time_elapsed?: number;
  time_remaining?: number;
}

export interface TrainingHistory {
  steps: number[];
  losses: number[];
  perplexities: number[];
  learningRates: number[];
  gradNorms: number[];
}

export interface ModelCheckpoint {
  config: GPTConfig;
  step: number;
  timestamp: number;
  name?: string;
}

export interface ModelState {
  // Model configuration
  config: GPTConfig;
  setConfig: (config: Partial<GPTConfig>) => void;
  
  // Training state
  isTraining: boolean;
  currentStep: number;
  loss: number;
  perplexity: number;
  learningRate: number;
  gradNorm: number;
  tokensPerSec: number;
  
  // History for charts
  history: TrainingHistory;
  
  // Actions
  startTraining: () => void;
  stopTraining: () => void;
  updateMetrics: (metrics: Partial<TrainingMetrics>) => void;
  resetHistory: () => void;
  
  // Model checkpoint
  checkpoint: ModelCheckpoint | null;
  saveCheckpoint: (name?: string) => void;
  loadCheckpoint: (checkpoint: ModelCheckpoint) => void;
  
  // User progress
  completedModules: string[];
  completeModule: (moduleId: string) => void;
  
  // Active model/session
  activeModelId: string | null;
  setActiveModelId: (id: string | null) => void;
}

export const defaultConfig: GPTConfig = {
  vocab_size: 256,
  max_seq_len: 256,
  d_model: 128,
  num_layers: 4,
  num_heads: 4,
  d_ff: 512,
  dropout: 0.1,
  attention_dropout: 0.1,
  activation: 'gelu',
  norm_type: 'rmsnorm',
  tie_weights: true,
};

export const modelPresets = {
  micro: {
    name: 'Micro (1M params)',
    config: {
      ...defaultConfig,
      vocab_size: 256,
      d_model: 128,
      num_layers: 4,
      num_heads: 4,
      d_ff: 512,
    },
    estimatedParams: 1_052_672,
  },
  small: {
    name: 'Small (10M params)',
    config: {
      ...defaultConfig,
      vocab_size: 1000,
      d_model: 256,
      num_layers: 6,
      num_heads: 8,
      d_ff: 1024,
    },
    estimatedParams: 10_500_000,
  },
  medium: {
    name: 'Medium (100M params)',
    config: {
      ...defaultConfig,
      vocab_size: 5000,
      d_model: 512,
      num_layers: 12,
      num_heads: 8,
      d_ff: 2048,
    },
    estimatedParams: 100_000_000,
  },
  gpt2: {
    name: 'GPT-2 Small (124M params)',
    config: {
      ...defaultConfig,
      vocab_size: 50257,
      d_model: 768,
      num_layers: 12,
      num_heads: 12,
      d_ff: 3072,
    },
    estimatedParams: 124_000_000,
  },
};

export const useModelStore = create<ModelState>()(
  persist(
    (set, get) => ({
      // Initial state
      config: defaultConfig,
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
      completedModules: [],
      activeModelId: null,
      
      // Actions
      setConfig: (config) => set((state) => ({
        config: { ...state.config, ...config }
      })),
      
      startTraining: () => set({ isTraining: true }),
      stopTraining: () => set({ isTraining: false }),
      
      updateMetrics: (metrics) => set((state) => {
        const step = metrics.step ?? state.currentStep + 1;
        return {
          currentStep: step,
          loss: metrics.loss ?? state.loss,
          perplexity: metrics.perplexity ?? state.perplexity,
          learningRate: metrics.learning_rate ?? state.learningRate,
          gradNorm: metrics.grad_norm ?? state.gradNorm,
          tokensPerSec: metrics.tokens_per_sec ?? state.tokensPerSec,
          history: {
            steps: [...state.history.steps, step],
            losses: [...state.history.losses, metrics.loss ?? state.loss],
            perplexities: [...state.history.perplexities, metrics.perplexity ?? state.perplexity],
            learningRates: [...state.history.learningRates, metrics.learning_rate ?? state.learningRate],
            gradNorms: [...state.history.gradNorms, metrics.grad_norm ?? state.gradNorm],
          }
        };
      }),
      
      resetHistory: () => set({
        currentStep: 0,
        loss: 0,
        perplexity: 0,
        gradNorm: 0,
        tokensPerSec: 0,
        history: {
          steps: [],
          losses: [],
          perplexities: [],
          learningRates: [],
          gradNorms: [],
        }
      }),
      
      saveCheckpoint: (name) => {
        const state = get();
        set({
          checkpoint: {
            config: state.config,
            step: state.currentStep,
            timestamp: Date.now(),
            name: name || `Checkpoint-${state.currentStep}`,
          }
        });
      },
      
      loadCheckpoint: (checkpoint) => set({
        config: checkpoint.config,
        currentStep: checkpoint.step,
      }),
      
      completeModule: (moduleId) => set((state) => ({
        completedModules: state.completedModules.includes(moduleId) 
          ? state.completedModules 
          : [...state.completedModules, moduleId]
      })),
      
      setActiveModelId: (id) => set({ activeModelId: id }),
    }),
    {
      name: 'llm-learning-storage',
      partialize: (state) => ({
        config: state.config,
        checkpoint: state.checkpoint,
        completedModules: state.completedModules,
      }),
    }
  )
);

// Helper function to calculate model parameters
export function calculateModelParams(config: GPTConfig): number {
  const { vocab_size, d_model, num_layers, num_heads, d_ff } = config;
  
  // Embedding parameters
  const tokenEmbeddings = vocab_size * d_model;
  const positionalEmbeddings = config.max_seq_len * d_model;
  
  // Attention parameters per layer
  const qkvProjection = 3 * d_model * d_model;
  const outputProjection = d_model * d_model;
  const attentionPerLayer = qkvProjection + outputProjection;
  
  // MLP parameters per layer
  const mlp = d_model * d_ff + d_ff * d_model;
  
  // Layer normalization (2 per layer + 1 final)
  const layerNorm = (2 * num_layers + 1) * d_model;
  
  // Total
  const total = tokenEmbeddings + positionalEmbeddings + 
                num_layers * (attentionPerLayer + mlp) + layerNorm;
  
  // Add output head if not tying weights
  if (!config.tie_weights) {
    return total + d_model * vocab_size;
  }
  
  return total;
}

// Helper to format parameter count
export function formatParams(count: number): string {
  if (count >= 1_000_000_000) {
    return `${(count / 1_000_000_000).toFixed(1)}B`;
  } else if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M`;
  } else if (count >= 1_000) {
    return `${(count / 1_000).toFixed(1)}K`;
  }
  return count.toString();
}

// Helper to estimate memory
export function estimateMemory(config: GPTConfig, precision: 'fp32' | 'fp16' = 'fp32'): number {
  const params = calculateModelParams(config);
  const bytesPerParam = precision === 'fp32' ? 4 : 2;
  
  // Model weights
  const modelMemory = params * bytesPerParam;
  
  // Gradients (same size as weights)
  const gradientMemory = params * bytesPerParam;
  
  // Optimizer states (Adam has 2 states per param)
  const optimizerMemory = params * bytesPerParam * 2;
  
  // Activations (rough estimate)
  const activationMemory = config.d_model * config.max_seq_len * bytesPerParam * 4;
  
  return modelMemory + gradientMemory + optimizerMemory + activationMemory;
}
