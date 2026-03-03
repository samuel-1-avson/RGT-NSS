"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { 
  ArrowLeft, 
  Play, 
  Square, 
  RotateCcw, 
  Save,
  Activity,
  TrendingDown,
  Zap,
  Settings,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Layers,
  Loader2
} from "lucide-react";
import Link from "next/link";
import { useModelStore, modelPresets, formatParams, calculateModelParams, estimateMemory } from "@/stores/modelStore";
import { useTraining, useBackendHealth } from "@/hooks/useTrainingSocket";
import { trainingApi, systemApi } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// Loss Chart Component
function LossChart({ data }: { data: { step: number; loss: number; perplexity: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="step" 
          stroke="#64748b" 
          tick={{ fill: '#64748b', fontSize: 12 }}
        />
        <YAxis 
          stroke="#64748b" 
          tick={{ fill: '#64748b', fontSize: 12 }}
          domain={['auto', 'auto']}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1e293b', 
            border: '1px solid #334155',
            borderRadius: '8px',
            color: '#fff'
          }}
        />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="loss" 
          stroke="#8b5cf6" 
          strokeWidth={2}
          dot={false}
          name="Loss"
        />
        <Line 
          type="monotone" 
          dataKey="perplexity" 
          stroke="#10b981" 
          strokeWidth={2}
          dot={false}
          name="Perplexity"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// Learning Rate Chart
function LearningRateChart({ data }: { data: { step: number; learningRate: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={150}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="step" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
        <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1e293b', 
            border: '1px solid #334155',
            borderRadius: '8px',
            color: '#fff'
          }}
        />
        <Line 
          type="monotone" 
          dataKey="learningRate" 
          stroke="#f59e0b" 
          strokeWidth={2}
          dot={false}
          name="Learning Rate"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// Gradient Norm Chart
function GradientNormChart({ data }: { data: { step: number; gradNorm: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={150}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="step" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
        <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1e293b', 
            border: '1px solid #334155',
            borderRadius: '8px',
            color: '#fff'
          }}
        />
        <Line 
          type="monotone" 
          dataKey="gradNorm" 
          stroke="#ec4899" 
          strokeWidth={2}
          dot={false}
          name="Gradient Norm"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// Hyperparameter Control Component
function HyperparameterControl({ 
  label, 
  value, 
  onChange, 
  min, 
  max, 
  step, 
  unit = '' 
}: { 
  label: string; 
  value: number; 
  onChange: (val: number) => void; 
  min: number; 
  max: number; 
  step: number;
  unit?: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between">
        <label className="text-sm text-slate-400">{label}</label>
        <span className="text-sm text-white font-mono">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-violet-500"
      />
    </div>
  );
}

export default function TrainPage() {
  const [showConfig, setShowConfig] = useState(true);
  const [selectedPreset, setSelectedPreset] = useState<keyof typeof modelPresets>('micro');
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<any>(null);
  
  // Get state from Zustand store
  const { 
    config, 
    setConfig, 
    isTraining, 
    currentStep, 
    loss, 
    perplexity, 
    learningRate, 
    gradNorm,
    tokensPerSec,
    history,
    resetHistory,
    saveCheckpoint,
    setActiveModelId,
  } = useModelStore();

  const { 
    isConnected, 
    isLoading,
    error: socketError, 
    sessionId,
    startTraining,
    stopTraining,
  } = useTraining();

  const { isHealthy, gpuStatus } = useBackendHealth();

  // Check backend status on mount
  useEffect(() => {
    const checkStatus = async () => {
      const { data, error } = await systemApi.health();
      if (data) {
        setBackendStatus(data);
      }
    };
    checkStatus();
  }, []);

  // Prepare chart data
  const chartData = history.steps.map((step, i) => ({
    step,
    loss: history.losses[i] || 0,
    perplexity: history.perplexities[i] || 0,
    learningRate: history.learningRates[i] || 0,
    gradNorm: history.gradNorms[i] || 0,
  }));

  // Handle preset selection
  const handlePresetChange = (preset: keyof typeof modelPresets) => {
    setSelectedPreset(preset);
    setConfig(modelPresets[preset].config);
  };

  // Handle training start with real API
  const handleStartTraining = async () => {
    setTrainingError(null);
    
    try {
      // First create a model if we don't have an active one
      let modelId = useModelStore.getState().activeModelId;
      
      if (!modelId) {
        // Create a new model
        const { data: modelData, error: modelError } = await trainingApi.start({
          model_id: '', // Will be created by backend
          dataset: 'shakespeare',
          batch_size: 16,
          learning_rate: learningRate || 0.001,
          min_learning_rate: 0.0001,
          warmup_steps: 100,
          max_steps: 1000,
          grad_clip: 1.0,
          weight_decay: 0.1,
          optimizer: 'adamw',
        });

        if (modelError) {
          throw new Error(modelError);
        }

        if (modelData?.session_id) {
          // Training started successfully
          console.log('Training started:', modelData);
        }
      } else {
        // Use existing model
        await startTraining({
          model_id: modelId,
          dataset: 'shakespeare',
          batch_size: 16,
          learning_rate: learningRate || 0.001,
          warmup_steps: 100,
          max_steps: 1000,
          grad_clip: 1.0,
          optimizer: 'adamw',
        });
      }
    } catch (err: any) {
      console.error('Failed to start training:', err);
      setTrainingError(err.message || 'Failed to start training');
    }
  };

  // Handle training stop
  const handleStopTraining = async () => {
    try {
      await stopTraining();
    } catch (err: any) {
      console.error('Failed to stop training:', err);
    }
  };

  // Calculate model stats
  const paramCount = calculateModelParams(config);
  const memoryEstimate = estimateMemory(config, 'fp32');

  // Combine errors
  const error = trainingError || socketError;

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-slate-800/50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>Back to Home</span>
              </Link>
            </div>
            
            {/* Backend Status */}
            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                isHealthy 
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                  : 'bg-red-500/10 text-red-400 border border-red-500/20'
              }`}>
                {isHealthy ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                <span>{isHealthy ? 'Backend Connected' : 'Backend Offline'}</span>
              </div>
              
              {gpuStatus?.available && (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-sm bg-violet-500/10 text-violet-400 border border-violet-500/20">
                  <Zap className="w-4 h-4" />
                  <span>GPU: {gpuStatus.device_name?.split(' ').slice(0, 2).join(' ')}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Title */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-2">Training Dashboard</h1>
          <p className="text-slate-400">
            Train your language model with real-time metrics and visualization
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Column - Controls */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Model Configuration */}
            <div className="bg-slate-900/50 rounded-2xl border border-slate-700/50 overflow-hidden">
              <button
                onClick={() => setShowConfig(!showConfig)}
                className="w-full flex items-center justify-between p-6 hover:bg-slate-800/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Settings className="w-5 h-5 text-violet-400" />
                  <h2 className="text-lg font-semibold">Model Configuration</h2>
                </div>
                {showConfig ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {showConfig && (
                <div className="px-6 pb-6 space-y-6">
                  {/* Preset Selector */}
                  <div>
                    <label className="block text-sm text-slate-400 mb-2">Preset</label>
                    <select
                      value={selectedPreset}
                      onChange={(e) => handlePresetChange(e.target.value as keyof typeof modelPresets)}
                      className="w-full p-3 bg-slate-800 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                    >
                      {Object.entries(modelPresets).map(([key, preset]) => (
                        <option key={key} value={key}>
                          {preset.name} (~{formatParams(preset.estimatedParams)} params)
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Model Stats */}
                  <div className="p-4 bg-slate-800/50 rounded-xl space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Parameters</span>
                      <span className="text-white font-mono">{formatParams(paramCount)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Memory (FP32)</span>
                      <span className="text-white font-mono">{(memoryEstimate / 1024 / 1024).toFixed(1)} MB</span>
                    </div>
                  </div>

                  {/* Hyperparameters */}
                  <div className="space-y-4">
                    <HyperparameterControl
                      label="Learning Rate"
                      value={learningRate || 0.001}
                      onChange={(v) => {}}
                      min={0.0001}
                      max={0.01}
                      step={0.0001}
                    />
                    <HyperparameterControl
                      label="Batch Size"
                      value={16}
                      onChange={(v) => {}}
                      min={1}
                      max={128}
                      step={1}
                    />
                    <HyperparameterControl
                      label="Max Steps"
                      value={1000}
                      onChange={(v) => {}}
                      min={100}
                      max={10000}
                      step={100}
                    />
                  </div>

                  {/* Architecture Details */}
                  <div className="pt-4 border-t border-slate-700/50">
                    <h3 className="text-sm font-medium text-slate-300 mb-3">Architecture</h3>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-slate-400">Layers: <span className="text-white">{config.num_layers}</span></div>
                      <div className="text-slate-400">Heads: <span className="text-white">{config.num_heads}</span></div>
                      <div className="text-slate-400">Dim: <span className="text-white">{config.d_model}</span></div>
                      <div className="text-slate-400">Vocab: <span className="text-white">{config.vocab_size}</span></div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Training Controls */}
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-emerald-400" />
                Training Controls
              </h2>
              
              <div className="flex gap-3">
                {!isTraining ? (
                  <button
                    onClick={handleStartTraining}
                    disabled={!isHealthy || isLoading}
                    className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-emerald-600 text-white rounded-xl font-medium hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isLoading ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <Play className="w-5 h-5" />
                    )}
                    {isLoading ? 'Starting...' : 'Start Training'}
                  </button>
                ) : (
                  <button
                    onClick={handleStopTraining}
                    className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-red-600 text-white rounded-xl font-medium hover:bg-red-500 transition-colors"
                  >
                    <Square className="w-5 h-5" />
                    Stop Training
                  </button>
                )}
                
                <button
                  onClick={resetHistory}
                  className="px-4 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition-colors"
                  title="Reset History"
                >
                  <RotateCcw className="w-5 h-5" />
                </button>
                
                <button
                  onClick={() => saveCheckpoint(`checkpoint-${currentStep}`)}
                  className="px-4 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition-colors"
                  title="Save Checkpoint"
                >
                  <Save className="w-5 h-5" />
                </button>
              </div>

              {error && (
                <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
                  {error}
                </div>
              )}

              {sessionId && (
                <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                  <div className="text-xs text-slate-400">Session ID</div>
                  <div className="text-sm font-mono text-slate-300">{sessionId}</div>
                </div>
              )}
            </div>
          </motion.div>

          {/* Right Column - Metrics & Charts */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Live Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/50">
                <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                  <TrendingDown className="w-4 h-4" />
                  Loss
                </div>
                <div className="text-2xl font-bold text-white">{loss.toFixed(4)}</div>
              </div>
              
              <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/50">
                <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                  <BarChart3 className="w-4 h-4" />
                  Perplexity
                </div>
                <div className="text-2xl font-bold text-emerald-400">{perplexity.toFixed(2)}</div>
              </div>
              
              <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/50">
                <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                  <Layers className="w-4 h-4" />
                  Step
                </div>
                <div className="text-2xl font-bold text-white">{currentStep.toLocaleString()}</div>
              </div>
              
              <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/50">
                <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                  <Zap className="w-4 h-4" />
                  Tokens/sec
                </div>
                <div className="text-2xl font-bold text-amber-400">{tokensPerSec.toFixed(0)}</div>
              </div>
            </div>

            {/* Main Loss Chart */}
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-violet-400" />
                Loss & Perplexity
              </h3>
              {chartData.length > 0 ? (
                <LossChart data={chartData} />
              ) : (
                <div className="h-[250px] flex items-center justify-center text-slate-500">
                  <div className="text-center">
                    <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>Start training to see metrics</p>
                  </div>
                </div>
              )}
            </div>

            {/* Secondary Charts */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-sm font-semibold mb-4 text-slate-300">Learning Rate</h3>
                {chartData.length > 0 ? (
                  <LearningRateChart data={chartData} />
                ) : (
                  <div className="h-[150px] flex items-center justify-center text-slate-500 text-sm">
                    No data yet
                  </div>
                )}
              </div>
              
              <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-sm font-semibold mb-4 text-slate-300">Gradient Norm</h3>
                {chartData.length > 0 ? (
                  <GradientNormChart data={chartData} />
                ) : (
                  <div className="h-[150px] flex items-center justify-center text-slate-500 text-sm">
                    No data yet
                  </div>
                )}
              </div>
            </div>

            {/* Training Status */}
            {isTraining && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 rounded-2xl p-6 border border-emerald-500/20"
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
                  <h3 className="font-semibold text-emerald-400">Training in Progress</h3>
                </div>
                
                <div className="w-full bg-slate-700 rounded-full h-2 mb-2">
                  <div 
                    className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min((currentStep / 1000) * 100, 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm text-slate-400">
                  <span>Step {currentStep}</span>
                  <span>Target: 1,000</span>
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>
      </div>
    </main>
  );
}
