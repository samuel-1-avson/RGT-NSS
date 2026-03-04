"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { inferenceOptApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function InferenceOptPage() {
  const [activeTab, setActiveTab] = useState<"kvcache" | "quantize" | "speculative">("kvcache");
  const [kvConfig, setKvConfig] = useState({ num_layers: 6, num_heads: 8, head_dim: 64, prompt_len: 20, gen_len: 50 });
  const [quantConfig, setQuantConfig] = useState({ rows: 512, cols: 512 });
  const [specConfig, setSpecConfig] = useState({ total_tokens: 100, gamma: 4, acceptance_rate: 0.7 });

  const techniques = useQuery<any>({ queryKey: ["inf-opt-techniques"], queryFn: () => inferenceOptApi.techniques() });

  const kvMutation = useMutation<any>({ mutationFn: () => inferenceOptApi.analyzeKVCache(kvConfig) });
  const quantMutation = useMutation<any>({ mutationFn: () => inferenceOptApi.compareQuantization(quantConfig) });
  const specMutation = useMutation<any>({ mutationFn: () => inferenceOptApi.runSpeculativeDecoding(specConfig) });

  const tabs = [
    { id: "kvcache" as const, label: "KV Cache", icon: "💾" },
    { id: "quantize" as const, label: "Quantization", icon: "📦" },
    { id: "speculative" as const, label: "Speculative Decoding", icon: "🚀" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Inference Optimization</h1>
        <p className="text-muted-foreground max-w-3xl">
          Explore KV caching, weight quantization, and speculative decoding to accelerate LLM inference latency and reduce memory costs.
        </p>
      </div>

      <div className="flex gap-2 border-b border-white/5 pb-0 overflow-x-auto custom-scrollbar">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "relative px-5 py-3 rounded-t-xl text-sm font-medium transition-all outline-none whitespace-nowrap",
              activeTab === tab.id
                ? "text-primary z-10"
                : "text-muted-foreground hover:text-foreground hover:bg-white/5"
            )}
          >
            {activeTab === tab.id && (
              <motion.div
                layoutId="activeTabInfOpt"
                className="absolute inset-0 bg-primary/10 border-b-2 border-primary"
                transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
              />
            )}
            <span className="relative z-10 flex items-center gap-2">
              <span className="opacity-80">{tab.icon}</span> {tab.label}
            </span>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeTab === "kvcache" && (
          <motion.div
            key="kvcache"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">KV Cache Analysis</h2>
                <p className="text-sm text-muted-foreground mt-1">Analyze how the key-value cache grows during autoregressive generation and its memory trade-offs.</p>
              </div>
              <div className="space-y-5">
                {[
                  { key: "num_layers", label: "Transformer Layers", min: 1, max: 32, step: 1 },
                  { key: "num_heads", label: "Attention Heads", min: 1, max: 32, step: 1 },
                  { key: "head_dim", label: "Head Dimension", min: 16, max: 128, step: 16 },
                  { key: "prompt_len", label: "Prompt Length", min: 1, max: 256, step: 5 },
                  { key: "gen_len", label: "Generation Length", min: 10, max: 200, step: 10 },
                ].map((p) => (
                  <div key={p.key}>
                    <div className="flex justify-between items-end mb-2">
                      <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">{p.label}</label>
                      <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">{(kvConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={(kvConfig as any)[p.key]}
                      onChange={(e) => setKvConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))}
                      className="w-full h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20" />
                  </div>
                ))}
              </div>
              <button onClick={() => kvMutation.mutate()} disabled={kvMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {kvMutation.isPending ? "Analyzing..." : "Analyze KV Cache Growth"}
              </button>
            </div>
            <div className="glass rounded-2xl p-6 flex flex-col">
              <h2 className="text-lg font-semibold text-foreground/90 mb-4 border-b border-white/10 pb-4">Cache Growth Telemetry</h2>
              {kvMutation.data ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6 flex-1 flex flex-col">
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: "Final Cache Size", value: `${(kvMutation.data as any).final_cache_mb?.toFixed(2)} MB`, color: "text-primary" },
                      { label: "Maximum Speedup", value: `${(kvMutation.data as any).max_speedup?.toFixed(1)}x`, color: "text-accent" },
                    ].map((m) => (
                      <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} key={m.label}
                        className="bg-black/40 border border-white/5 rounded-xl p-4 flex flex-col justify-center text-center">
                        <div className={`font-mono text-2xl font-bold ${m.color} mb-1`}>{m.value}</div>
                        <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground/80">{m.label}</div>
                      </motion.div>
                    ))}
                  </div>
                  <div className="bg-black/20 border border-white/5 rounded-xl p-5 flex-1 flex flex-col">
                    <div className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground mb-4">Cache Growth Over Tokens</div>
                    <div className="flex-1 flex items-end gap-[1px] min-h-[140px] bg-black/40 rounded p-1">
                      {((kvMutation.data as any).steps || []).filter((_: any, i: number) => i % 3 === 0).map((s: any, i: number) => {
                        const maxMb = (kvMutation.data as any).final_cache_mb || 1;
                        const h = (s.cache_mb / maxMb) * 100;
                        return (
                          <motion.div
                            initial={{ height: 0 }} animate={{ height: `${Math.max(h, 1)}%` }} transition={{ duration: 0.8, delay: i * 0.02 }}
                            key={i}
                            className="flex-1 bg-gradient-to-t from-primary/80 to-accent/90 rounded-t-sm opacity-80"
                            title={`Token ${s.token}: ${s.cache_mb.toFixed(2)} MB`}
                          />
                        );
                      })}
                    </div>
                  </div>
                </motion.div>
              ) : (
                <div className="flex-1 min-h-[250px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                  Run analysis to see KV cache growth curve
                </div>
              )}
            </div>
          </motion.div>
        )}

        {activeTab === "quantize" && (
          <motion.div
            key="quantize"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">Weight Quantization Comparison</h2>
                <p className="text-sm text-muted-foreground mt-1">Compare FP32, FP16, INT8, INT4 quantization quality, memory savings, and reconstruction error.</p>
              </div>
              <div className="flex flex-col md:flex-row gap-4 items-end">
                {[{ key: "rows", label: "Matrix Rows" }, { key: "cols", label: "Matrix Cols" }].map((p) => (
                  <div key={p.key} className="flex-1 w-full">
                    <div className="flex justify-between items-end mb-2">
                      <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">{p.label}</label>
                      <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">{(quantConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={64} max={2048} step={64} value={(quantConfig as any)[p.key]}
                      onChange={(e) => setQuantConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))}
                      className="w-full h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20" />
                  </div>
                ))}
                <button onClick={() => quantMutation.mutate()} disabled={quantMutation.isPending}
                  className="px-8 py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm whitespace-nowrap">
                  {quantMutation.isPending ? "Analyzing..." : "Compare Quantization"}
                </button>
              </div>
            </div>
            {quantMutation.data && (
              <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="glass rounded-2xl p-6">
                <div className="flex justify-between items-center mb-6 pb-4 border-b border-white/10">
                  <h2 className="text-lg font-semibold text-foreground/90">Quantization Comparison</h2>
                  <div className="text-[10px] font-bold text-primary uppercase tracking-widest bg-primary/10 px-3 py-1 rounded-full">Results Ready</div>
                </div>
                <div className="overflow-x-auto rounded-xl border border-white/5 bg-black/40">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10 bg-white/5">
                        <th className="text-left py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Format</th>
                        <th className="text-right py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Bits</th>
                        <th className="text-right py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Memory (MB)</th>
                        <th className="text-right py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Compression</th>
                        <th className="text-right py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">MSE</th>
                        <th className="text-right py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Max Error</th>
                      </tr>
                    </thead>
                    <tbody>
                      {((quantMutation.data as any).comparisons || []).map((c: any, i: number) => (
                        <motion.tr
                          initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                          key={i}
                          className="border-b last:border-0 border-white/5 hover:bg-white/5 transition-colors group"
                        >
                          <td className="py-4 px-5 font-semibold text-foreground/80 group-hover:text-primary transition-colors">{c.format}</td>
                          <td className="py-4 px-5 text-right font-mono text-blue-400">{c.bits}</td>
                          <td className="py-4 px-5 text-right font-mono">{c.memory_mb?.toFixed(3)}</td>
                          <td className="py-4 px-5 text-right font-mono text-accent">{c.compression_ratio?.toFixed(1)}x</td>
                          <td className="py-4 px-5 text-right font-mono text-muted-foreground">{c.mse?.toFixed(6)}</td>
                          <td className="py-4 px-5 text-right font-mono text-muted-foreground">{c.max_error?.toFixed(6)}</td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {activeTab === "speculative" && (
          <motion.div
            key="speculative"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">Speculative Decoding</h2>
                <p className="text-sm text-muted-foreground mt-1">Use a small draft model to propose tokens, verified by the large target model in parallel for faster inference.</p>
              </div>
              <div className="space-y-5">
                {[
                  { key: "total_tokens", label: "Total Tokens", min: 10, max: 500, step: 10 },
                  { key: "gamma", label: "Draft Tokens (γ)", min: 1, max: 8, step: 1 },
                  { key: "acceptance_rate", label: "Acceptance Rate", min: 0.1, max: 1, step: 0.05 },
                ].map((p) => (
                  <div key={p.key}>
                    <div className="flex justify-between items-end mb-2">
                      <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">{p.label}</label>
                      <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">{(specConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={(specConfig as any)[p.key]}
                      onChange={(e) => setSpecConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))}
                      className="w-full h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20" />
                  </div>
                ))}
              </div>
              <button onClick={() => specMutation.mutate()} disabled={specMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {specMutation.isPending ? "Simulating..." : "Run Speculative Decoding Simulation"}
              </button>
            </div>
            <div className="glass rounded-2xl p-6 flex flex-col">
              <h2 className="text-lg font-semibold text-foreground/90 mb-4 border-b border-white/10 pb-4">Decoding Analysis</h2>
              {specMutation.data ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3 flex-1">
                  {Object.entries(specMutation.data as Record<string, any>).map(([k, v], i) => (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.05 }}
                      key={k}
                      className="bg-black/40 border border-white/5 rounded-xl p-4 flex justify-between items-center group hover:bg-white/5 transition-colors"
                    >
                      <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground/80">{k.replace(/_/g, " ")}</div>
                      <div className="font-mono text-sm font-bold text-accent">
                        {typeof v === "number" ? v.toFixed(4) : JSON.stringify(v)}
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              ) : (
                <div className="flex-1 min-h-[250px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                  Run decoding analysis to see results
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Techniques Overview */}
      {techniques.data && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="glass rounded-2xl p-6 space-y-4"
        >
          <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Optimization Techniques</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {((techniques.data as any).techniques || []).map((t: any) => (
              <div key={t.name} className="bg-black/20 border border-white/5 rounded-xl p-4 space-y-2 hover:bg-white/5 transition-colors group cursor-default">
                <h3 className="font-semibold text-primary text-xs uppercase tracking-widest group-hover:text-primary/80 transition-colors">{t.name}</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">{t.description}</p>
                <div className="flex gap-4 pt-2 border-t border-white/5 text-[10px]">
                  <span><span className="text-emerald-500 font-bold uppercase">Speedup:</span> <span className="text-foreground/80">{t.speedup}</span></span>
                  <span><span className="text-amber-500 font-bold uppercase">Cost:</span> <span className="text-foreground/80">{t.memory_cost}</span></span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
