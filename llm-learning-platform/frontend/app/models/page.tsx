"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { 
  ArrowLeft, 
  Brain, 
  Settings,
  Calculator,
  Save,
  Play,
  Layers,
  Hash,
  Maximize,
  Zap,
  AlertCircle,
  CheckCircle,
  ChevronDown,
  ChevronUp
} from "lucide-react";
import Link from "next/link";
import { useModelStore, modelPresets, formatParams, calculateModelParams, estimateMemory, GPTConfig } from "@/stores/modelStore";

// Architecture visualization component
function ArchitectureVisualizer({ config }: { config: GPTConfig }) {
  const blockHeight = 60;
  const spacing = 10;
  
  return (
    <div className="relative">
      <svg 
        viewBox={`0 0 300 ${100 + config.num_layers * (blockHeight + spacing)}`}
        className="w-full h-auto"
      >
        {/* Input */}
        <rect x="100" y="10" width="100" height="30" rx="4" fill="#8b5cf6" opacity="0.3" />
        <text x="150" y="30" textAnchor="middle" fill="#a78bfa" fontSize="12">Input + Pos Embed</text>
        
        {/* Transformer Blocks */}
        {Array.from({ length: config.num_layers }).map((_, i) => (
          <g key={i} transform={`translate(0, ${60 + i * (blockHeight + spacing)})`}>
            {/* Block container */}
            <rect x="20" y="0" width="260" height={blockHeight} rx="8" fill="#1e293b" stroke="#334155" strokeWidth="1" />
            
            {/* Attention */}
            <rect x="30" y="10" width="100" height="40" rx="4" fill="#f59e0b" opacity="0.3" />
            <text x="80" y="35" textAnchor="middle" fill="#fbbf24" fontSize="10">Attention</text>
            
            {/* MLP */}
            <rect x="150" y="10" width="100" height="40" rx="4" fill="#10b981" opacity="0.3" />
            <text x="200" y="35" textAnchor="middle" fill="#34d399" fontSize="10">MLP</text>
            
            {/* Layer number */}
            <text x="270" y="35" textAnchor="middle" fill="#64748b" fontSize="10">{i + 1}</text>
          </g>
        ))}
        
        {/* Output */}
        <rect 
          x="100" 
          y={70 + config.num_layers * (blockHeight + spacing)} 
          width="100" 
          height="30" 
          rx="4" 
          fill="#ec4899" 
          opacity="0.3" 
        />
        <text 
          x="150" 
          y={90 + config.num_layers * (blockHeight + spacing)} 
          textAnchor="middle" 
          fill="#f472b6" 
          fontSize="12"
        >
          Output
        </text>
      </svg>
    </div>
  );
}

// Config slider component
function ConfigSlider({ 
  label, 
  value, 
  onChange, 
  min, 
  max, 
  step,
  description
}: { 
  label: string; 
  value: number; 
  onChange: (val: number) => void; 
  min: number; 
  max: number; 
  step: number;
  description?: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-start">
        <div>
          <label className="text-sm font-medium text-slate-300">{label}</label>
          {description && <p className="text-xs text-slate-500">{description}</p>}
        </div>
        <span className="text-sm font-mono text-violet-400 bg-violet-500/10 px-2 py-1 rounded">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full accent-violet-500 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
      />
      <div className="flex justify-between text-xs text-slate-500">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [modelName, setModelName] = useState('');
  
  const { config, setConfig, saveCheckpoint } = useModelStore();
  
  // Calculate stats
  const paramCount = calculateModelParams(config);
  const memoryBytes = estimateMemory(config, 'fp32');
  const memoryMB = (memoryBytes / 1024 / 1024).toFixed(1);
  
  // Estimate FLOPs per token (rough approximation)
  const flopsPerToken = (
    2 * paramCount + // Forward pass
    2 * config.num_layers * config.max_seq_len * config.d_model // Attention quadratic term
  );
  
  // Find closest preset
  const findClosestPreset = () => {
    let closest = 'custom';
    let minDiff = Infinity;
    
    Object.entries(modelPresets).forEach(([key, preset]) => {
      const diff = Math.abs(preset.estimatedParams - paramCount);
      if (diff < minDiff) {
        minDiff = diff;
        closest = key;
      }
    });
    
    return closest;
  };
  
  const currentPreset = findClosestPreset();
  
  // Handle config update
  const updateConfig = (updates: Partial<GPTConfig>) => {
    setConfig(updates);
  };
  
  // Save model configuration
  const handleSave = () => {
    saveCheckpoint(modelName || `Custom-${Date.now()}`);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-slate-800/50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back to Home</span>
            </Link>
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
          <h1 className="text-4xl font-bold mb-2">Model Configurator</h1>
          <p className="text-slate-400">
            Design and configure your own transformer architecture
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - Configuration */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Presets */}
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Layers className="w-5 h-5 text-violet-400" />
                Architecture Presets
              </h2>
              
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(modelPresets).map(([key, preset]) => (
                  <button
                    key={key}
                    onClick={() => setConfig(preset.config)}
                    className={`p-3 rounded-xl text-left transition-all ${
                      currentPreset === key
                        ? 'bg-violet-600 text-white'
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                    }`}
                  >
                    <div className="font-medium text-sm">{preset.name}</div>
                    <div className={`text-xs ${currentPreset === key ? 'text-violet-200' : 'text-slate-500'}`}>
                      {formatParams(preset.estimatedParams)} params
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Basic Configuration */}
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <Settings className="w-5 h-5 text-amber-400" />
                Basic Configuration
              </h2>
              
              <div className="space-y-6">
                <ConfigSlider
                  label="Number of Layers"
                  value={config.num_layers}
                  onChange={(v) => updateConfig({ num_layers: v })}
                  min={1}
                  max={24}
                  step={1}
                  description="Depth of the transformer"
                />
                
                <ConfigSlider
                  label="Embedding Dimension"
                  value={config.d_model}
                  onChange={(v) => updateConfig({ d_model: v })}
                  min={64}
                  max={1024}
                  step={64}
                  description="Size of token embeddings"
                />
                
                <ConfigSlider
                  label="Attention Heads"
                  value={config.num_heads}
                  onChange={(v) => updateConfig({ num_heads: v })}
                  min={1}
                  max={16}
                  step={1}
                  description="Parallel attention mechanisms"
                />
                
                <ConfigSlider
                  label="Feedforward Dimension"
                  value={config.d_ff}
                  onChange={(v) => updateConfig({ d_ff: v })}
                  min={256}
                  max={4096}
                  step={256}
                  description="MLP hidden layer size"
                />
              </div>
            </div>

            {/* Advanced Settings */}
            <div className="bg-slate-900/50 rounded-2xl border border-slate-700/50 overflow-hidden">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full flex items-center justify-between p-6 hover:bg-slate-800/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-emerald-400" />
                  <h2 className="text-lg font-semibold">Advanced Settings</h2>
                </div>
                {showAdvanced ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {showAdvanced && (
                <div className="px-6 pb-6 space-y-6">
                  <ConfigSlider
                    label="Vocabulary Size"
                    value={config.vocab_size}
                    onChange={(v) => updateConfig({ vocab_size: v })}
                    min={128}
                    max={50000}
                    step={128}
                    description="Number of unique tokens"
                  />
                  
                  <ConfigSlider
                    label="Context Window"
                    value={config.max_seq_len}
                    onChange={(v) => updateConfig({ max_seq_len: v })}
                    min={128}
                    max={2048}
                    step={128}
                    description="Maximum sequence length"
                  />
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-slate-400 mb-2">Activation</label>
                      <select
                        value={config.activation}
                        onChange={(e) => updateConfig({ activation: e.target.value as any })}
                        className="w-full p-3 bg-slate-800 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                      >
                        <option value="gelu">GELU</option>
                        <option value="relu">ReLU</option>
                        <option value="swiglu">SwiGLU</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-slate-400 mb-2">Normalization</label>
                      <select
                        value={config.norm_type}
                        onChange={(e) => updateConfig({ norm_type: e.target.value as any })}
                        className="w-full p-3 bg-slate-800 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                      >
                        <option value="rmsnorm">RMSNorm</option>
                        <option value="layernorm">LayerNorm</option>
                      </select>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-slate-400 mb-2">Dropout Rate</label>
                    <input
                      type="range"
                      min={0}
                      max={0.5}
                      step={0.05}
                      value={config.dropout}
                      onChange={(e) => updateConfig({ dropout: parseFloat(e.target.value) })}
                      className="w-full accent-violet-500"
                    />
                    <div className="text-center text-sm text-violet-400 mt-1">
                      {(config.dropout * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex gap-4">
              <input
                type="text"
                placeholder="Model name (optional)"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="flex-1 px-4 py-3 bg-slate-800 rounded-xl text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
              <button
                onClick={handleSave}
                className="flex items-center gap-2 px-6 py-3 bg-violet-600 text-white rounded-xl font-medium hover:bg-violet-500 transition-colors"
              >
                <Save className="w-5 h-5" />
                Save
              </button>
            </div>
          </motion.div>

          {/* Right Column - Visualization & Stats */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Architecture Preview */}
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-pink-400" />
                Architecture Preview
              </h2>
              <div className="bg-slate-800/50 rounded-xl p-4">
                <ArchitectureVisualizer config={config} />
              </div>
            </div>

            {/* Model Statistics */}
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5 text-emerald-400" />
                Model Statistics
              </h2>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-slate-800/50 rounded-xl">
                  <div className="text-sm text-slate-400 mb-1">Total Parameters</div>
                  <div className="text-2xl font-bold text-white">{formatParams(paramCount)}</div>
                  <div className="text-xs text-slate-500">~{(paramCount / 1e6).toFixed(2)}M</div>
                </div>
                
                <div className="p-4 bg-slate-800/50 rounded-xl">
                  <div className="text-sm text-slate-400 mb-1">Memory (FP32)</div>
                  <div className="text-2xl font-bold text-emerald-400">{memoryMB} MB</div>
                  <div className="text-xs text-slate-500">{(parseFloat(memoryMB) / 1024).toFixed(2)} GB</div>
                </div>
                
                <div className="p-4 bg-slate-800/50 rounded-xl">
                  <div className="text-sm text-slate-400 mb-1">FLOPs / Token</div>
                  <div className="text-2xl font-bold text-amber-400">{(flopsPerToken / 1e6).toFixed(1)}M</div>
                  <div className="text-xs text-slate-500">Forward pass only</div>
                </div>
                
                <div className="p-4 bg-slate-800/50 rounded-xl">
                  <div className="text-sm text-slate-400 mb-1">Est. Training Time</div>
                  <div className="text-2xl font-bold text-violet-400">~5 min</div>
                  <div className="text-xs text-slate-500">1K steps, batch=32</div>
                </div>
              </div>
            </div>

            {/* Configuration Summary */}
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Hash className="w-5 h-5 text-cyan-400" />
                Configuration Summary
              </h2>
              
              <div className="space-y-3">
                <div className="flex justify-between py-2 border-b border-slate-700/50">
                  <span className="text-slate-400">Architecture Type</span>
                  <span className="text-white font-medium">Decoder-only Transformer</span>
                </div>
                <div className="flex justify-between py-2 border-b border-slate-700/50">
                  <span className="text-slate-400">Attention Pattern</span>
                  <span className="text-white font-medium">Causal (Autoregressive)</span>
                </div>
                <div className="flex justify-between py-2 border-b border-slate-700/50">
                  <span className="text-slate-400">Position Encoding</span>
                  <span className="text-white font-medium">Learned Embeddings</span>
                </div>
                <div className="flex justify-between py-2 border-b border-slate-700/50">
                  <span className="text-slate-400">Head Dimension</span>
                  <span className="text-white font-medium">{config.d_model / config.num_heads}</span>
                </div>
                <div className="flex justify-between py-2">
                  <span className="text-slate-400">Parameters per Layer</span>
                  <span className="text-white font-medium">
                    {formatParams(Math.floor(paramCount / config.num_layers))}
                  </span>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-gradient-to-br from-violet-900/20 to-indigo-900/20 rounded-2xl p-6 border border-violet-500/20">
              <h3 className="font-semibold text-white mb-3">Ready to Train?</h3>
              <p className="text-slate-400 text-sm mb-4">
                Your model configuration is ready. Start training to see it learn!
              </p>
              <Link
                href="/train/"
                className="flex items-center justify-center gap-2 w-full px-6 py-3 bg-gradient-to-r from-violet-600 to-indigo-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-violet-500/25 transition-all"
              >
                <Play className="w-5 h-5" />
                Start Training
              </Link>
            </div>
          </motion.div>
        </div>
      </div>
    </main>
  );
}
