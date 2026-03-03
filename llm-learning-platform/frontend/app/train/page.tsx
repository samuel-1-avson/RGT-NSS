'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  Play, 
  Square, 
  Settings, 
  TrendingUp, 
  Clock,
  Activity,
  Brain,
  RotateCcw,
  Save
} from 'lucide-react';
import { useModelStore, modelPresets, defaultGPTConfig } from '@/stores/modelStore';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import toast from 'react-hot-toast';
import { api, wsService } from '@/utils/api';
import { GPTConfig, TrainingMetrics } from '@/types';

// Model Configurator Component
function ModelConfigurator({ 
  config, 
  onChange 
}: { 
  config: GPTConfig; 
  onChange: (config: Partial<GPTConfig>) => void;
}) {
  return (
    <div className="bg-white rounded-xl shadow-sm ring-1 ring-slate-200 p-6">
      <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
        <Settings className="h-5 w-5 text-primary-500" />
        Model Configuration
      </h3>
      
      {/* Presets */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Quick Presets
        </label>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(modelPresets).map(([key, preset]) => (
            <button
              key={key}
              onClick={() => onChange(preset.config as GPTConfig)}
              className="px-3 py-2 text-sm font-medium text-slate-700 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors"
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {/* Sliders */}
      <div className="space-y-4">
        <ConfigSlider
          label="Vocabulary Size"
          value={config.vocab_size}
          min={16}
          max={1000}
          step={16}
          onChange={(v) => onChange({ vocab_size: v })}
        />
        <ConfigSlider
          label="Model Dimension (d_model)"
          value={config.d_model}
          min={32}
          max={512}
          step={32}
          onChange={(v) => onChange({ d_model: v })}
        />
        <ConfigSlider
          label="Number of Layers"
          value={config.num_layers}
          min={1}
          max={12}
          step={1}
          onChange={(v) => onChange({ num_layers: v })}
        />
        <ConfigSlider
          label="Number of Heads"
          value={config.num_heads}
          min={1}
          max={16}
          step={1}
          onChange={(v) => onChange({ num_heads: v })}
        />
        <ConfigSlider
          label="Feedforward Dimension"
          value={config.d_ff}
          min={64}
          max={2048}
          step={64}
          onChange={(v) => onChange({ d_ff: v })}
        />
        <ConfigSlider
          label="Context Length"
          value={config.max_seq_len}
          min={32}
          max={512}
          step={32}
          onChange={(v) => onChange({ max_seq_len: v })}
        />
        <ConfigSlider
          label="Dropout"
          value={config.dropout}
          min={0}
          max={0.5}
          step={0.05}
          onChange={(v) => onChange({ dropout: v })}
        />
      </div>

      {/* Activation Function */}
      <div className="mt-4">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Activation Function
        </label>
        <select
          value={config.activation}
          onChange={(e) => onChange({ activation: e.target.value as any })}
          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
        >
          <option value="gelu">GELU</option>
          <option value="relu">ReLU</option>
          <option value="swiglu">SwiGLU</option>
        </select>
      </div>

      {/* Norm Type */}
      <div className="mt-4">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Normalization
        </label>
        <select
          value={config.norm_type}
          onChange={(e) => onChange({ norm_type: e.target.value as any })}
          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
        >
          <option value="rmsnorm">RMSNorm (Modern)</option>
          <option value="layernorm">LayerNorm (Classic)</option>
        </select>
      </div>
    </div>
  );
}

function ConfigSlider({ label, value, min, max, step, onChange }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        <span className="text-sm text-slate-500">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

// Training Dashboard Component
export default function TrainPage() {
  const { 
    currentModel, 
    modelConfig, 
    setCurrentModel, 
    setModelConfig,
    isTraining,
    setIsTraining,
    trainingSessionId,
    setTrainingSessionId,
    trainingMetrics,
    addTrainingMetric,
    clearTrainingMetrics,
    currentStep,
  } = useModelStore();

  const [isCreatingModel, setIsCreatingModel] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    batch_size: 32,
    learning_rate: 0.001,
    max_steps: 1000,
    warmup_steps: 100,
    grad_clip: 1.0,
  });

  // Create model
  const createModel = useCallback(async () => {
    setIsCreatingModel(true);
    try {
      const model = await api.createModel(modelConfig);
      setCurrentModel(model);
      toast.success(`Model created with ${model.num_parameters.toLocaleString()} parameters`);
    } catch (error) {
      toast.error('Failed to create model');
      console.error(error);
    } finally {
      setIsCreatingModel(false);
    }
  }, [modelConfig, setCurrentModel]);

  // Start training
  const startTraining = useCallback(async () => {
    if (!currentModel) {
      toast.error('Create a model first');
      return;
    }

    try {
      const session = await api.startTraining({
        ...trainingConfig,
        model_id: currentModel.model_id,
      });
      
      setTrainingSessionId(session.session_id);
      setIsTraining(true);
      clearTrainingMetrics();
      
      // Connect to WebSocket
      wsService.connect(session.session_id);
      
      toast.success('Training started');
    } catch (error) {
      toast.error('Failed to start training');
      console.error(error);
    }
  }, [currentModel, trainingConfig, setTrainingSessionId, setIsTraining, clearTrainingMetrics]);

  // Stop training
  const stopTraining = useCallback(async () => {
    if (!trainingSessionId) return;
    
    try {
      await api.stopTraining(trainingSessionId);
      setIsTraining(false);
      wsService.disconnect();
      toast.success('Training stopped');
    } catch (error) {
      toast.error('Failed to stop training');
    }
  }, [trainingSessionId, setIsTraining]);

  // Listen to WebSocket messages
  useEffect(() => {
    wsService.onMetrics((metrics: TrainingMetrics) => {
      addTrainingMetric(metrics);
    });

    return () => {
      wsService.disconnect();
    };
  }, [addTrainingMetric]);

  // Calculate estimated parameters
  const estimatedParams = React.useMemo(() => {
    const d_model = modelConfig.d_model;
    const vocab_size = modelConfig.vocab_size;
    const num_layers = modelConfig.num_layers;
    const d_ff = modelConfig.d_ff;
    
    // Embeddings
    const embedding = vocab_size * d_model;
    
    // Per layer: attention + MLP + norms
    const attention_per_layer = 4 * d_model * d_model; // Q, K, V, O projections
    const mlp_per_layer = d_model * d_ff * 3; // gate, up, down projections
    const layer_total = attention_per_layer + mlp_per_layer + 2 * d_model; // + norms
    
    // Total
    const total = embedding * 2 + layer_total * num_layers + vocab_size * d_model;
    
    return total;
  }, [modelConfig]);

  return (
    <main className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="h-8 w-8 text-primary-600" />
              <span className="text-xl font-semibold text-slate-900">
                Training Dashboard
              </span>
            </div>
            <div className="flex items-center gap-4">
              {currentModel && (
                <span className="text-sm text-slate-600">
                  Model: <span className="font-medium">{currentModel.model_id}</span>
                </span>
              )}
              <a
                href="/"
                className="text-sm font-medium text-slate-600 hover:text-slate-900"
              >
                Home
              </a>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid gap-8 lg:grid-cols-3">
          {/* Left Panel - Configuration */}
          <div className="space-y-6">
            {/* Model Config */}
            <ModelConfigurator config={modelConfig} onChange={setModelConfig} />

            {/* Training Config */}
            <div className="bg-white rounded-xl shadow-sm ring-1 ring-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">
                Training Configuration
              </h3>
              <div className="space-y-4">
                <ConfigSlider
                  label="Batch Size"
                  value={trainingConfig.batch_size}
                  min={1}
                  max={128}
                  step={1}
                  onChange={(v) => setTrainingConfig(c => ({ ...c, batch_size: v }))}
                />
                <ConfigSlider
                  label="Learning Rate"
                  value={trainingConfig.learning_rate}
                  min={0.0001}
                  max={0.01}
                  step={0.0001}
                  onChange={(v) => setTrainingConfig(c => ({ ...c, learning_rate: v }))}
                />
                <ConfigSlider
                  label="Max Steps"
                  value={trainingConfig.max_steps}
                  min={100}
                  max={50000}
                  step={100}
                  onChange={(v) => setTrainingConfig(c => ({ ...c, max_steps: v }))}
                />
                <ConfigSlider
                  label="Warmup Steps"
                  value={trainingConfig.warmup_steps}
                  min={0}
                  max={1000}
                  step={10}
                  onChange={(v) => setTrainingConfig(c => ({ ...c, warmup_steps: v }))}
                />
              </div>
            </div>

            {/* Model Stats */}
            <div className="bg-primary-50 rounded-xl p-6">
              <h4 className="text-sm font-medium text-primary-900 mb-2">
                Estimated Model Size
              </h4>
              <p className="text-2xl font-bold text-primary-700">
                {(estimatedParams / 1e6).toFixed(2)}M
              </p>
              <p className="text-sm text-primary-600">
                parameters
              </p>
            </div>
          </div>

          {/* Center/Right Panel - Training & Visualization */}
          <div className="lg:col-span-2 space-y-6">
            {/* Action Bar */}
            <div className="bg-white rounded-xl shadow-sm ring-1 ring-slate-200 p-4">
              <div className="flex items-center gap-4">
                {!currentModel ? (
                  <button
                    onClick={createModel}
                    disabled={isCreatingModel}
                    className="flex items-center gap-2 px-6 py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors"
                  >
                    {isCreatingModel ? (
                      <>
                        <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        Creating...
                      </>
                    ) : (
                      <>
                        <Brain className="h-5 w-5" />
                        Create Model
                      </>
                    )}
                  </button>
                ) : (
                  <>
                    {!isTraining ? (
                      <button
                        onClick={startTraining}
                        className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors"
                      >
                        <Play className="h-5 w-5" />
                        Start Training
                      </button>
                    ) : (
                      <button
                        onClick={stopTraining}
                        className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 transition-colors"
                      >
                        <Square className="h-5 w-5" />
                        Stop Training
                      </button>
                    )}
                    <button
                      onClick={createModel}
                      className="flex items-center gap-2 px-4 py-3 border border-slate-300 rounded-lg font-medium text-slate-700 hover:bg-slate-50 transition-colors"
                    >
                      <RotateCcw className="h-5 w-5" />
                      Reset
                    </button>
                  </>
                )}

                {currentModel && (
                  <div className="ml-auto flex items-center gap-4">
                    <div className="text-sm text-slate-600">
                      Status: {' '}
                      <span className={`font-medium ${
                        isTraining ? 'text-green-600' : 'text-slate-900'
                      }`}>
                        {isTraining ? 'Training' : 'Ready'}
                      </span>
                    </div>
                    {currentStep > 0 && (
                      <div className="text-sm text-slate-600">
                        Step: <span className="font-medium">{currentStep}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Charts */}
            {trainingMetrics.length > 0 && (
              <>
                {/* Loss Chart */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white rounded-xl shadow-sm ring-1 ring-slate-200 p-6"
                >
                  <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-primary-500" />
                    Loss Curve
                  </h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingMetrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis 
                          dataKey="step" 
                          stroke="#64748b"
                          fontSize={12}
                        />
                        <YAxis 
                          stroke="#64748b"
                          fontSize={12}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#fff',
                            border: '1px solid #e2e8f0',
                            borderRadius: '8px'
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="loss" 
                          stroke="#0ea5e9" 
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </motion.div>

                {/* Perplexity Chart */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white rounded-xl shadow-sm ring-1 ring-slate-200 p-6"
                >
                  <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                    <Activity className="h-5 w-5 text-primary-500" />
                    Perplexity
                  </h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingMetrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis 
                          dataKey="step" 
                          stroke="#64748b"
                          fontSize={12}
                        />
                        <YAxis 
                          stroke="#64748b"
                          fontSize={12}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#fff',
                            border: '1px solid #e2e8f0',
                            borderRadius: '8px'
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="perplexity" 
                          stroke="#8b5cf6" 
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </motion.div>

                {/* Learning Rate Chart */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white rounded-xl shadow-sm ring-1 ring-slate-200 p-6"
                >
                  <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                    <Clock className="h-5 w-5 text-primary-500" />
                    Learning Rate Schedule
                  </h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingMetrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis 
                          dataKey="step" 
                          stroke="#64748b"
                          fontSize={12}
                        />
                        <YAxis 
                          stroke="#64748b"
                          fontSize={12}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#fff',
                            border: '1px solid #e2e8f0',
                            borderRadius: '8px'
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="learning_rate" 
                          stroke="#f59e0b" 
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </motion.div>
              </>
            )}

            {/* Current Metrics */}
            {trainingMetrics.length > 0 && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: 'Loss', value: trainingMetrics[trainingMetrics.length - 1]?.loss.toFixed(4), color: 'text-primary-600' },
                  { label: 'Perplexity', value: trainingMetrics[trainingMetrics.length - 1]?.perplexity.toFixed(2), color: 'text-purple-600' },
                  { label: 'Learning Rate', value: trainingMetrics[trainingMetrics.length - 1]?.learning_rate.toExponential(2), color: 'text-amber-600' },
                  { label: 'Tokens/sec', value: trainingMetrics[trainingMetrics.length - 1]?.tokens_per_sec.toFixed(0), color: 'text-green-600' },
                ].map((metric) => (
                  <div key={metric.label} className="bg-white rounded-xl shadow-sm ring-1 ring-slate-200 p-4">
                    <p className="text-sm text-slate-500">{metric.label}</p>
                    <p className={`text-xl font-bold ${metric.color}`}>{metric.value}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Empty State */}
            {trainingMetrics.length === 0 && currentModel && !isTraining && (
              <div className="bg-slate-100 rounded-xl p-12 text-center">
                <Play className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-900 mb-2">
                  Ready to Train
                </h3>
                <p className="text-slate-600">
                  Click "Start Training" to begin training your model.
                  Real-time metrics will appear here.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
