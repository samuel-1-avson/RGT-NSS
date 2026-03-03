import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { 
  GPTConfig, 
  ModelInfo, 
  TrainingMetrics, 
  TrainingStatus 
} from '@/types';

// ==================== Default Configurations ====================

export const defaultGPTConfig: GPTConfig = {
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
  nano: {
    name: 'Nano (100K params)',
    config: {
      ...defaultGPTConfig,
      vocab_size: 128,
      d_model: 64,
      num_layers: 2,
      num_heads: 2,
      d_ff: 256,
    }
  },
  micro: {
    name: 'Micro (1M params)',
    config: {
      ...defaultGPTConfig,
      vocab_size: 256,
      d_model: 128,
      num_layers: 4,
      num_heads: 4,
      d_ff: 512,
    }
  },
  small: {
    name: 'Small (10M params)',
    config: {
      ...defaultGPTConfig,
      vocab_size: 512,
      d_model: 256,
      num_layers: 6,
      num_heads: 8,
      d_ff: 1024,
    }
  },
  medium: {
    name: 'Medium (100M params)',
    config: {
      ...defaultGPTConfig,
      vocab_size: 1000,
      d_model: 512,
      num_layers: 8,
      num_heads: 8,
      d_ff: 2048,
    }
  },
};

// ==================== Store Interface ====================

interface ModelState {
  // Current model
  currentModel: ModelInfo | null;
  modelConfig: GPTConfig;
  
  // Training state
  trainingSessionId: string | null;
  isTraining: boolean;
  trainingStatus: TrainingStatus | null;
  trainingMetrics: TrainingMetrics[];
  currentStep: number;
  
  // UI state
  activeModule: string;
  
  // Actions
  setCurrentModel: (model: ModelInfo | null) => void;
  setModelConfig: (config: Partial<GPTConfig>) => void;
  loadPreset: (preset: keyof typeof modelPresets) => void;
  
  setTrainingSessionId: (id: string | null) => void;
  setIsTraining: (isTraining: boolean) => void;
  setTrainingStatus: (status: TrainingStatus | null) => void;
  addTrainingMetric: (metric: TrainingMetrics) => void;
  clearTrainingMetrics: () => void;
  setCurrentStep: (step: number) => void;
  
  setActiveModule: (module: string) => void;
}

// ==================== Store Implementation ====================

export const useModelStore = create<ModelState>()(
  persist(
    (set, get) => ({
      // Initial state
      currentModel: null,
      modelConfig: defaultGPTConfig,
      trainingSessionId: null,
      isTraining: false,
      trainingStatus: null,
      trainingMetrics: [],
      currentStep: 0,
      activeModule: 'dashboard',
      
      // Model actions
      setCurrentModel: (model) => set({ currentModel: model }),
      
      setModelConfig: (config) => set((state) => ({
        modelConfig: { ...state.modelConfig, ...config }
      })),
      
      loadPreset: (preset) => {
        const presetConfig = modelPresets[preset];
        if (presetConfig) {
          set({ modelConfig: presetConfig.config as GPTConfig });
        }
      },
      
      // Training actions
      setTrainingSessionId: (id) => set({ trainingSessionId: id }),
      
      setIsTraining: (isTraining) => set({ isTraining }),
      
      setTrainingStatus: (status) => set({ trainingStatus: status }),
      
      addTrainingMetric: (metric) => set((state) => ({
        trainingMetrics: [...state.trainingMetrics.slice(-500), metric],
        currentStep: metric.step,
      })),
      
      clearTrainingMetrics: () => set({ 
        trainingMetrics: [],
        currentStep: 0 
      }),
      
      setCurrentStep: (step) => set({ currentStep: step }),
      
      // UI actions
      setActiveModule: (module) => set({ activeModule: module }),
    }),
    {
      name: 'llm-learning-storage',
      partialize: (state) => ({
        currentModel: state.currentModel,
        modelConfig: state.modelConfig,
        activeModule: state.activeModule,
      }),
    }
  )
);

// ==================== Derived Stores ====================

export const useTrainingProgress = () => {
  return useModelStore((state) => ({
    progress: state.trainingStatus?.progress || 0,
    currentStep: state.currentStep,
    maxSteps: 10000, // Default, can be fetched from config
  }));
};

export const useLatestMetrics = () => {
  return useModelStore((state) => {
    const metrics = state.trainingMetrics;
    return metrics.length > 0 ? metrics[metrics.length - 1] : null;
  });
};
