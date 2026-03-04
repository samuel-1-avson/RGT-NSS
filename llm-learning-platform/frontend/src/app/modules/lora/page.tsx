"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { loraApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function LoRAPage() {
  const [activeTab, setActiveTab] = useState<"lora" | "qlora" | "compare">("lora");
  const [loraConfig, setLoraConfig] = useState({ d_model: 256, num_layers: 4, rank: 8, alpha: 16, dropout: 0.05, target_modules: ["q_proj", "v_proj"] });
  const [trainConfig, setTrainConfig] = useState({ d_model: 256, num_layers: 4, rank: 8, num_steps: 30 });
  const [quantConfig, setQuantConfig] = useState({ rows: 256, cols: 256 });

  const ranks = useQuery<any>({ queryKey: ["lora-ranks"], queryFn: () => loraApi.ranks() });
  const createMutation = useMutation<any>({ mutationFn: () => loraApi.create(loraConfig) });
  const forwardMutation = useMutation<any>({ mutationFn: () => loraApi.forward({ d_model: loraConfig.d_model, rank: loraConfig.rank, seq_len: 8 }) });
  const trainMutation = useMutation<any>({ mutationFn: () => loraApi.train(trainConfig) });
  const quantMutation = useMutation<any>({ mutationFn: () => loraApi.quantize(quantConfig) });
  const compareMutation = useMutation<any>({ mutationFn: () => loraApi.comparePeft(512, 6) });

  const toMethodsArray = (value: unknown): any[] => {
    if (Array.isArray(value)) {
      return value;
    }
    if (value && typeof value === "object") {
      return Object.entries(value as Record<string, any>).map(([method, meta]) => ({ method, ...(meta || {}) }));
    }
    return [];
  };

  const tabs = [
    { id: "lora" as const, label: "LoRA Explorer", icon: "🔧" },
    { id: "qlora" as const, label: "QLoRA & Quantization", icon: "📦" },
    { id: "compare" as const, label: "PEFT Comparison", icon: "📊" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">LoRA Studio</h1>
        <p className="text-muted-foreground max-w-3xl">
          Explore Low-Rank Adaptation — build LoRA layers, visualize weight decomposition,
          experiment with QLoRA quantization, and compare parameter-efficient methods.
        </p>
      </div>

      <div className="flex gap-2 border-b border-white/5 pb-0">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "relative px-5 py-3 rounded-t-xl text-sm font-medium transition-all outline-none",
              activeTab === tab.id
                ? "text-primary z-10"
                : "text-muted-foreground hover:text-foreground hover:bg-white/5"
            )}
          >
            {activeTab === tab.id && (
              <motion.div
                layoutId="activeTabLoRA"
                className="absolute inset-0 bg-primary/10 border-b-2 border-primary"
                transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
              />
            )}
            <span className="relative z-10 flex items-center gap-2">
              <span className="opacity-80">{tab.icon}</span>
              {tab.label}
            </span>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeTab === "lora" && (
          <motion.div
            key="lora"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">LoRA Configuration</h2>
              <div className="p-4 rounded-xl border-l-2 border-primary bg-primary/5">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  LoRA decomposes weight updates: <span className="font-mono text-primary font-bold">ΔW = A × B</span> where <span className="font-mono text-foreground/80">A ∈ ℝ^(d×r)</span> and <span className="font-mono text-foreground/80">B ∈ ℝ^(r×d)</span>.
                </p>
              </div>
              <div className="space-y-4">
                {[
                  { key: "d_model", label: "Model Dimension", min: 32, max: 1024, step: 32 },
                  { key: "num_layers", label: "Num Layers", min: 1, max: 12, step: 1 },
                  { key: "rank", label: "LoRA Rank (r)", min: 1, max: 64, step: 1 },
                  { key: "alpha", label: "LoRA Alpha", min: 1, max: 128, step: 1 },
                  { key: "dropout", label: "Dropout", min: 0, max: 0.5, step: 0.05 },
                ].map((p) => (
                  <div key={p.key}>
                    <div className="flex justify-between mb-1.5">
                      <span className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest">{p.label}</span>
                      <span className="text-sm font-mono text-primary">{(loraConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={(loraConfig as any)[p.key]}
                      onChange={(e) => setLoraConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))}
                      className="w-full accent-primary h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-primary/50" />
                  </div>
                ))}
              </div>
              <div className="flex gap-3 pt-2">
                <button onClick={() => createMutation.mutate()} disabled={createMutation.isPending}
                  className="flex-1 py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                  {createMutation.isPending ? "Creating..." : "Create LoRA Model"}
                </button>
                <button onClick={() => forwardMutation.mutate()} disabled={forwardMutation.isPending}
                  className="flex-1 py-2.5 rounded-xl bg-accent hover:bg-accent/90 text-accent-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-accent/20 text-sm">
                  {forwardMutation.isPending ? "Running..." : "Forward Pass"}
                </button>
              </div>
            </div>
            <div className="space-y-6">
              {createMutation.data && (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="glass rounded-2xl p-6 space-y-4 border-primary/20">
                  <h2 className="text-lg font-semibold text-primary pb-2 border-b border-white/10">Model Summary</h2>
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(createMutation.data as Record<string, any>).map(([k, v]) => (
                      <div key={k} className="bg-black/20 border border-white/5 rounded-xl p-3">
                        <div className="text-[11px] uppercase tracking-wider text-muted-foreground mb-1">{k.replace(/_/g, " ")}</div>
                        <div className="font-mono text-foreground/90 text-sm">
                          {typeof v === "number" ? v.toLocaleString() : typeof v === "object" ? JSON.stringify(v) : String(v)}
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
              {forwardMutation.data && (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="glass rounded-2xl p-6 space-y-4 border-accent/20">
                  <h2 className="text-lg font-semibold text-accent pb-2 border-b border-white/10">Forward Pass</h2>
                  <div className="grid grid-cols-2 gap-3">
                    {["A_shape", "B_shape", "delta_w_norm", "scaling"].map((k) => (
                      <div key={k} className="bg-black/20 border border-white/5 rounded-xl p-3">
                        <div className="text-[11px] uppercase tracking-wider text-muted-foreground mb-1">{k.replace(/_/g, " ")}</div>
                        <div className="font-mono text-sm text-foreground/80">{JSON.stringify((forwardMutation.data as any)[k])}</div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
              {ranks.data && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass rounded-2xl p-6 space-y-4">
                  <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Rank Guide</h2>
                  <div className="space-y-2 max-h-[250px] overflow-y-auto custom-scrollbar pr-2">
                    {((ranks.data as any).ranks || []).map((r: any) => (
                      <div key={r.rank} className="flex items-center gap-3 bg-white/5 hover:bg-white/10 transition-colors border border-white/5 rounded-xl p-3">
                        <div className="w-10 h-10 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center font-bold text-primary text-sm shadow-sm">r={r.rank}</div>
                        <div className="flex-1">
                          <div className="text-[13px] font-semibold text-foreground/90">{r.use_case}</div>
                          <div className="text-[11px] font-mono text-muted-foreground mt-0.5">Quality: {r.quality}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
            <div className="glass rounded-2xl p-6 space-y-6 col-span-1 md:col-span-2">
              <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">LoRA Training Process</h2>
              <div className="flex flex-col md:flex-row md:items-end gap-6">
                {[{ key: "rank", label: "Rank", min: 1, max: 64, step: 1 }, { key: "num_steps", label: "Steps", min: 5, max: 100, step: 5 }].map((p) => (
                  <div key={p.key} className="flex-1">
                    <div className="flex justify-between mb-1.5">
                      <span className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest">{p.label}</span>
                      <span className="text-sm font-mono text-emerald-400">{(trainConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={(trainConfig as any)[p.key]}
                      onChange={(e) => setTrainConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))}
                      className="w-full accent-emerald-500 h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-emerald-500/50" />
                  </div>
                ))}
                <button onClick={() => trainMutation.mutate()} disabled={trainMutation.isPending}
                  className="px-8 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-medium disabled:opacity-50 transition-all shadow-lg shadow-emerald-500/20 text-sm">
                  {trainMutation.isPending ? "Training..." : "Run Training"}
                </button>
              </div>
              {trainMutation.data && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-black/20 border border-white/5 rounded-2xl p-6">
                  <div className="flex items-end gap-1 h-32 w-full">
                    {((trainMutation.data as any).training_metrics || []).map((m: any, i: number) => {
                      const losses = ((trainMutation.data as any).training_metrics || []).map((x: any) => x.loss);
                      const max = Math.max(...losses); const min = Math.min(...losses);
                      const h = ((m.loss - min) / (max - min || 1)) * 100;
                      return <div key={i} className="flex-1 bg-emerald-500/80 hover:bg-emerald-400 rounded-t transition-all group relative" style={{ height: `${Math.max(100 - h, 2)}%` }}>
                        <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-black text-white text-[10px] font-mono py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
                          loss: {m.loss.toFixed(4)}
                        </div>
                      </div>;
                    })}
                  </div>
                  <div className="text-[11px] font-bold tracking-widest uppercase text-muted-foreground mt-4 text-center">Training Loss (lower = better)</div>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}

        {activeTab === "qlora" && (
          <motion.div
            key="qlora"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">NF4 Quantization Analysis</h2>
              <div className="p-4 rounded-xl border-l-2 border-primary bg-primary/5">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  QLoRA uses 4-bit NormalFloat quantization with double quantization to drastically reduce memory usage while maintaining precision.
                </p>
              </div>
              <div className="space-y-4">
                {[{ key: "rows", label: "Matrix Rows", min: 32, max: 1024, step: 32 }, { key: "cols", label: "Matrix Cols", min: 32, max: 1024, step: 32 }].map((p) => (
                  <div key={p.key}>
                    <div className="flex justify-between mb-1.5">
                      <span className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest">{p.label}</span>
                      <span className="text-sm font-mono text-primary">{(quantConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={(quantConfig as any)[p.key]}
                      onChange={(e) => setQuantConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))}
                      className="w-full accent-primary h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-primary/50" />
                  </div>
                ))}
              </div>
              <button onClick={() => quantMutation.mutate()} disabled={quantMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {quantMutation.isPending ? "Analyzing..." : "Analyze Quantization"}
              </button>
            </div>
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Quantization Results</h2>
              {quantMutation.data ? (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="space-y-3">
                  {Object.entries(quantMutation.data as Record<string, any>).map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center bg-black/20 border border-white/5 rounded-xl p-3.5">
                      <div className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">{k.replace(/_/g, " ")}</div>
                      <div className="font-mono text-sm text-primary">{typeof v === "number" ? v.toFixed(4) : JSON.stringify(v)}</div>
                    </div>
                  ))}
                </motion.div>
              ) : (<div className="flex items-center justify-center h-48 border border-dashed border-white/10 rounded-xl text-center text-muted-foreground text-sm">Run quantization analysis to see memory savings results</div>)}
            </div>
          </motion.div>
        )}

        {activeTab === "compare" && (
          <motion.div
            key="compare"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="glass rounded-2xl p-6 space-y-4 flex flex-col md:flex-row md:items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-foreground/90 mb-1">Parameter-Efficient Fine-Tuning Comparison</h2>
                <p className="text-sm text-muted-foreground">Compare Full Fine-Tuning, LoRA, QLoRA, and Prefix Tuning metrics.</p>
              </div>
              <button onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}
                className="px-6 py-2.5 whitespace-nowrap rounded-xl bg-accent hover:bg-accent/90 text-accent-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-accent/20 text-sm">
                {compareMutation.isPending ? "Evaluating Methods..." : "Run PEFT Comparison"}
              </button>
            </div>
            {compareMutation.data && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-2xl overflow-hidden p-[1px]">
                <div className="bg-background/80 rounded-[15px] p-6 h-full">
                  <h2 className="text-lg font-semibold mb-4 text-foreground/90">Comparison Results Matrix</h2>
                  <div className="overflow-x-auto custom-scrollbar pb-2">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10 uppercase tracking-widest text-[10px] text-muted-foreground text-left">
                          <th className="py-3 px-4 font-bold">Method</th>
                          <th className="text-right py-3 px-4 font-bold shrink-0 min-w-[120px]">Trainable</th>
                          <th className="text-right py-3 px-4 font-bold shrink-0 min-w-[120px]">Total Params</th>
                          <th className="text-right py-3 px-4 font-bold">% Trn</th>
                          <th className="text-right py-3 px-4 font-bold">Memory (MB)</th>
                          <th className="text-right py-3 px-4 font-bold">Quality</th>
                        </tr>
                      </thead>
                      <tbody>
                        {toMethodsArray((compareMutation.data as any).methods).map((m: any, i: number) => (
                          <motion.tr
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.1 }}
                            key={i}
                            className="border-b border-white/5 hover:bg-white/5 transition-colors group"
                          >
                            <td className="py-3 px-4 font-semibold text-[13px]">{m.method}</td>
                            <td className="py-3 px-4 text-right font-mono text-muted-foreground">{m.trainable_params?.toLocaleString()}</td>
                            <td className="py-3 px-4 text-right font-mono text-muted-foreground">{(m.total_params || (compareMutation.data as any)?.total_base_params)?.toLocaleString()}</td>
                            <td className="py-3 px-4 text-right font-mono text-primary font-bold">{(m.pct_trainable ?? m.percentage)?.toFixed?.(2)}%</td>
                            <td className="py-3 px-4 text-right font-mono text-accent">{m.memory_mb?.toFixed(1)}</td>
                            <td className="py-3 px-4 text-right font-mono text-emerald-400">
                              <div className="flex items-center justify-end gap-2">
                                <div className="h-1.5 w-16 bg-white/10 rounded-full overflow-hidden flex-shrink-0">
                                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${m.estimated_quality}%` }} />
                                </div>
                                {m.estimated_quality?.toFixed(1)}%
                              </div>
                            </td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
