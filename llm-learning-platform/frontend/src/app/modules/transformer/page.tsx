"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Transformer Builder</h1>
        <p className="text-muted-foreground max-w-3xl">
          Assemble a transformer architecture by choosing dimensions, normalization, activations, and positional encodings.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="col-span-1 md:col-span-2 glass rounded-2xl p-6 space-y-8"
        >
          {/* Presets */}
          <div>
            <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-3 block">Quick Presets</label>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  onClick={() => applyPreset(p)}
                  className="px-4 py-1.5 rounded-lg border border-white/10 bg-black/20 hover:bg-white/10 text-xs text-muted-foreground hover:text-foreground capitalize transition-all"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          {/* Dimensions */}
          <div>
            <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-4 block">Dimensions</label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
              {[
                { key: "d_model", label: "d_model", min: 32, max: 2048, step: 32 },
                { key: "num_heads", label: "Attention Heads", min: 1, max: 32, step: 1 },
                { key: "num_layers", label: "Transformer Layers", min: 1, max: 24, step: 1 },
                { key: "d_ff", label: "Feed-Forward Dim", min: 64, max: 8192, step: 64 },
                { key: "vocab_size", label: "Vocabulary Size", min: 32, max: 100000, step: 32 },
                { key: "dropout", label: "Dropout", min: 0, max: 0.5, step: 0.05 },
              ].map((param) => (
                <div key={param.key}>
                  <div className="flex justify-between mb-1.5">
                    <span className="text-sm font-medium text-foreground/80">{param.label}</span>
                    <span className="text-sm font-mono text-primary">{(config as any)[param.key]}</span>
                  </div>
                  <input
                    type="range"
                    min={param.min}
                    max={param.max}
                    step={param.step}
                    value={(config as any)[param.key]}
                    onChange={(e) =>
                      setConfig((prev) => ({ ...prev, [param.key]: parseFloat(e.target.value) }))
                    }
                    className="w-full accent-primary h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-primary/50"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Architecture Choices */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
            <div>
              <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-3 block">Normalization</label>
              <div className="flex flex-col gap-2">
                {NORM_OPTIONS.map((n) => (
                  <button
                    key={n.value}
                    onClick={() => setConfig((prev) => ({ ...prev, norm_type: n.value }))}
                    className={cn(
                      "relative px-4 py-2 text-sm text-left rounded-lg border transition-all overflow-hidden",
                      config.norm_type === n.value ? "border-primary/50 text-primary" : "border-white/5 text-muted-foreground hover:bg-white/5"
                    )}
                  >
                    {config.norm_type === n.value && (
                      <motion.div layoutId="normType" className="absolute inset-0 bg-primary/10" transition={{ type: "spring", bounce: 0.2, duration: 0.6 }} />
                    )}
                    <span className="relative z-10 font-medium">{n.label}</span>
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-3 block">Activation</label>
              <div className="flex flex-col gap-2">
                {ACTIVATION_OPTIONS.map((a) => (
                  <button
                    key={a.value}
                    onClick={() => setConfig((prev) => ({ ...prev, activation: a.value }))}
                    className={cn(
                      "relative px-4 py-2 text-sm text-left rounded-lg border transition-all overflow-hidden",
                      config.activation === a.value ? "border-accent/50 text-accent" : "border-white/5 text-muted-foreground hover:bg-white/5"
                    )}
                  >
                    {config.activation === a.value && (
                      <motion.div layoutId="activationType" className="absolute inset-0 bg-accent/10" transition={{ type: "spring", bounce: 0.2, duration: 0.6 }} />
                    )}
                    <span className="relative z-10 font-medium">{a.label}</span>
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-3 block">Position Encoding</label>
              <div className="flex flex-col gap-2">
                {POSITION_OPTIONS.map((p) => (
                  <button
                    key={p.value}
                    onClick={() => setConfig((prev) => ({ ...prev, positional_encoding: p.value }))}
                    className={cn(
                      "relative px-4 py-2 text-sm text-left rounded-lg border transition-all overflow-hidden",
                      config.positional_encoding === p.value ? "border-emerald-500/50 text-emerald-400" : "border-white/5 text-muted-foreground hover:bg-white/5"
                    )}
                  >
                    {config.positional_encoding === p.value && (
                      <motion.div layoutId="posEncoding" className="absolute inset-0 bg-emerald-500/10" transition={{ type: "spring", bounce: 0.2, duration: 0.6 }} />
                    )}
                    <span className="relative z-10 font-medium">{p.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex gap-3 pt-4 border-t border-white/5">
            <button
              onClick={() => createModel.mutate()}
              disabled={createModel.isPending}
              className="px-6 py-2.5 bg-primary hover:bg-primary/90 text-primary-foreground rounded-xl font-medium transition-all shadow-lg shadow-primary/20 disabled:opacity-50 disabled:shadow-none text-sm"
            >
              {createModel.isPending ? "Creating..." : "Build Model"}
            </button>
            <button
              onClick={() => viewArchitecture.mutate()}
              disabled={viewArchitecture.isPending}
              className="px-6 py-2.5 glass rounded-xl font-medium transition-all hover:bg-white/10 text-sm border-white/10 disabled:opacity-50"
            >
              {viewArchitecture.isPending ? "Loading..." : "Visualize Architecture"}
            </button>
          </div>
        </motion.div>

        {/* Model Summary */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="space-y-6"
        >
          <div className="glass rounded-2xl p-6">
            <h2 className="text-lg font-semibold text-foreground/90 mb-4 pb-2 border-b border-white/10">Model Summary</h2>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Dimensions</span>
                <span className="font-mono text-foreground/80 bg-white/5 px-2 py-0.5 rounded">{config.d_model}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Heads</span>
                <span className="font-mono text-foreground/80 bg-white/5 px-2 py-0.5 rounded">{config.num_heads}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Layers</span>
                <span className="font-mono text-foreground/80 bg-white/5 px-2 py-0.5 rounded">{config.num_layers}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Head Dim</span>
                <span className="font-mono text-foreground/80 bg-white/5 px-2 py-0.5 rounded">{Math.floor(config.d_model / config.num_heads)}</span>
              </div>
              <div className="h-px bg-white/10 w-full my-4" />
              {modelResult && (
                <div className="flex justify-between items-center bg-primary/5 p-3 rounded-lg border border-primary/20">
                  <span className="text-primary font-medium">Parameters</span>
                  <span className="font-mono font-bold text-primary">
                    {formatNumber(modelResult.num_parameters)}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Architecture Diagram */}
          {archData && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="glass rounded-2xl p-6 border-accent/20"
            >
              <h2 className="text-lg font-semibold text-accent mb-4 pb-2 border-b border-white/10">Layer Stack</h2>
              <div className="space-y-1.5 max-h-[400px] overflow-y-auto custom-scrollbar pr-2">
                {archData.layers?.map((layer: any, i: number) => (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    key={i}
                    className={cn(
                      "px-3 py-2.5 rounded-lg text-xs font-mono border",
                      layer.type === "attention" && "bg-amber-500/10 border-amber-500/20 text-amber-500",
                      layer.type === "mlp" && "bg-emerald-500/10 border-emerald-500/20 text-emerald-500",
                      layer.type === "norm" && "bg-blue-500/10 border-blue-500/20 text-blue-500",
                      layer.type === "embedding" && "bg-purple-500/10 border-purple-500/20 text-purple-400",
                      layer.type === "positional" && "bg-pink-500/10 border-pink-500/20 text-pink-400",
                      layer.type === "output" && "bg-rose-500/10 border-rose-500/20 text-rose-500"
                    )}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-semibold">{layer.name}</span>
                      <span className="opacity-70 bg-black/20 px-1.5 py-0.5 rounded">
                        {formatNumber(layer.params)}
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
}
