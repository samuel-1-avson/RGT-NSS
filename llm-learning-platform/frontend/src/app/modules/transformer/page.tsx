"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { modelsApi, visualizationsApi } from "@/lib/api";
import { cn, formatNumber } from "@/lib/utils";

const PRESETS = ["nano", "micro", "mini", "small", "medium", "large"];

const NORM_OPTIONS = [
  { value: "layernorm", label: "LayerNorm" },
  { value: "rmsnorm", label: "RMSNorm" },
  { value: "deepnorm", label: "DeepNorm" },
];

const ACTIVATION_OPTIONS = [
  { value: "gelu", label: "GELU" },
  { value: "relu", label: "ReLU" },
  { value: "swiglu", label: "SwiGLU" },
  { value: "silu", label: "SiLU" },
];

const POSITION_OPTIONS = [
  { value: "sinusoidal", label: "Sinusoidal" },
  { value: "learned", label: "Learned" },
  { value: "rope", label: "RoPE" },
  { value: "alibi", label: "ALiBi" },
];

export default function TransformerPage() {
  const [config, setConfig] = useState({
    d_model: 128,
    num_heads: 4,
    num_layers: 4,
    d_ff: 512,
    vocab_size: 256,
    norm_type: "rmsnorm",
    activation: "gelu",
    positional_encoding: "sinusoidal",
    attention_type: "full",
    dropout: 0.1,
  });

  const [archData, setArchData] = useState<any>(null);
  const [modelResult, setModelResult] = useState<any>(null);

  const createModel = useMutation({
    mutationFn: () =>
      modelsApi.create(config),
    onSuccess: (data) => setModelResult(data),
  });

  const viewArchitecture = useMutation({
    mutationFn: () => visualizationsApi.modelArchitecture("micro"),
    onSuccess: (data) => setArchData(data),
  });

  const applyPreset = (preset: string) => {
    const presets: Record<string, any> = {
      nano: { d_model: 64, num_heads: 2, num_layers: 2, d_ff: 256 },
      micro: { d_model: 128, num_heads: 4, num_layers: 4, d_ff: 512 },
      mini: { d_model: 256, num_heads: 8, num_layers: 6, d_ff: 1024 },
      small: { d_model: 512, num_heads: 8, num_layers: 8, d_ff: 2048 },
      medium: { d_model: 768, num_heads: 12, num_layers: 12, d_ff: 3072 },
      large: { d_model: 1024, num_heads: 16, num_layers: 16, d_ff: 4096 },
    };
    if (presets[preset]) {
      setConfig((prev) => ({ ...prev, ...presets[preset] }));
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Transformer Builder</h1>
        <p className="text-muted-foreground">
          Assemble a transformer architecture by choosing dimensions, normalization, activations, and positional encodings.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <div className="col-span-2 glass rounded-2xl p-6 space-y-6">
          {/* Presets */}
          <div>
            <label className="text-sm font-medium mb-2 block">Quick Presets</label>
            <div className="flex gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  onClick={() => applyPreset(p)}
                  className="px-4 py-2 rounded-xl border border-border hover:border-primary-300 text-sm capitalize transition-colors"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          {/* Dimensions */}
          <div className="grid grid-cols-2 gap-4">
            {[
              { key: "d_model", label: "d_model", min: 32, max: 2048, step: 32 },
              { key: "num_heads", label: "Attention Heads", min: 1, max: 32, step: 1 },
              { key: "num_layers", label: "Transformer Layers", min: 1, max: 24, step: 1 },
              { key: "d_ff", label: "Feed-Forward Dim", min: 64, max: 8192, step: 64 },
              { key: "vocab_size", label: "Vocabulary Size", min: 32, max: 100000, step: 32 },
              { key: "dropout", label: "Dropout", min: 0, max: 0.5, step: 0.05 },
            ].map((param) => (
              <div key={param.key}>
                <label className="text-sm font-medium mb-1 block">
                  {param.label}: <span className="text-primary-500">{(config as any)[param.key]}</span>
                </label>
                <input
                  type="range"
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  value={(config as any)[param.key]}
                  onChange={(e) =>
                    setConfig((prev) => ({ ...prev, [param.key]: parseFloat(e.target.value) }))
                  }
                  className="w-full"
                />
              </div>
            ))}
          </div>

          {/* Architecture Choices */}
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Normalization</label>
              {NORM_OPTIONS.map((n) => (
                <label key={n.value} className="flex items-center gap-2 text-sm py-1 cursor-pointer">
                  <input
                    type="radio"
                    value={n.value}
                    checked={config.norm_type === n.value}
                    onChange={(e) => setConfig((prev) => ({ ...prev, norm_type: e.target.value }))}
                    className="accent-primary-500"
                  />
                  {n.label}
                </label>
              ))}
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Activation</label>
              {ACTIVATION_OPTIONS.map((a) => (
                <label key={a.value} className="flex items-center gap-2 text-sm py-1 cursor-pointer">
                  <input
                    type="radio"
                    value={a.value}
                    checked={config.activation === a.value}
                    onChange={(e) => setConfig((prev) => ({ ...prev, activation: e.target.value }))}
                    className="accent-primary-500"
                  />
                  {a.label}
                </label>
              ))}
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Position Encoding</label>
              {POSITION_OPTIONS.map((p) => (
                <label key={p.value} className="flex items-center gap-2 text-sm py-1 cursor-pointer">
                  <input
                    type="radio"
                    value={p.value}
                    checked={config.positional_encoding === p.value}
                    onChange={(e) => setConfig((prev) => ({ ...prev, positional_encoding: e.target.value }))}
                    className="accent-primary-500"
                  />
                  {p.label}
                </label>
              ))}
            </div>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => createModel.mutate()}
              disabled={createModel.isPending}
              className="px-6 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors disabled:opacity-50"
            >
              {createModel.isPending ? "Creating..." : "Create Model"}
            </button>
            <button
              onClick={() => viewArchitecture.mutate()}
              className="px-6 py-2.5 glass rounded-xl font-medium hover:bg-white/90 dark:hover:bg-gray-800/90 transition-colors"
            >
              View Architecture
            </button>
          </div>
        </div>

        {/* Model Summary */}
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6">
            <h2 className="text-lg font-semibold mb-4">Model Summary</h2>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">d_model</span>
                <span className="font-mono">{config.d_model}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Heads</span>
                <span className="font-mono">{config.num_heads}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Layers</span>
                <span className="font-mono">{config.num_layers}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Head Dim</span>
                <span className="font-mono">{Math.floor(config.d_model / config.num_heads)}</span>
              </div>
              <hr className="border-border" />
              {modelResult && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Parameters</span>
                  <span className="font-mono font-bold gradient-text">
                    {formatNumber(modelResult.num_parameters)}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Architecture Diagram */}
          {archData && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Layer Stack</h2>
              <div className="space-y-1">
                {archData.layers?.map((layer: any, i: number) => (
                  <div
                    key={i}
                    className={cn(
                      "px-3 py-2 rounded-lg text-xs font-mono",
                      layer.type === "attention" && "bg-amber-100 dark:bg-amber-900/30",
                      layer.type === "mlp" && "bg-emerald-100 dark:bg-emerald-900/30",
                      layer.type === "norm" && "bg-blue-100 dark:bg-blue-900/30",
                      layer.type === "embedding" && "bg-purple-100 dark:bg-purple-900/30",
                      layer.type === "positional" && "bg-pink-100 dark:bg-pink-900/30",
                      layer.type === "output" && "bg-red-100 dark:bg-red-900/30"
                    )}
                  >
                    <div className="flex justify-between">
                      <span>{layer.name}</span>
                      <span className="text-muted-foreground">
                        {formatNumber(layer.params)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
