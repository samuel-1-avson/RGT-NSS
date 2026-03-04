"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { longContextApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function LongContextPage() {
  const [activeTab, setActiveTab] = useState<"rope" | "alibi" | "compare">("rope");
  const [ropeConfig, setRopeConfig] = useState({ dim: 128, scaling_factor: 4, max_position: 2048, method: "none" });
  const [alibiConfig, setAlibiConfig] = useState({ num_heads: 8, seq_len: 16 });
  const [extConfig, setExtConfig] = useState({ num_heads: 8, train_length: 2048 });

  const ropeMutation = useMutation<any>({ mutationFn: () => longContextApi.ropeFrequencies(ropeConfig) });
  const ropeCompareMutation = useMutation<any>({ mutationFn: () => longContextApi.ropeCompare(ropeConfig) });
  const alibiMutation = useMutation<any>({ mutationFn: () => longContextApi.alibiBiasMatrix(alibiConfig) });
  const extMutation = useMutation<any>({ mutationFn: () => longContextApi.alibiExtrapolation(extConfig) });
  const compareAllMutation = useMutation<any>({ mutationFn: () => longContextApi.compareAll(128, 8, 2048) });

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
    { id: "rope" as const, label: "RoPE & Scaling", icon: "🔄" },
    { id: "alibi" as const, label: "ALiBi", icon: "📏" },
    { id: "compare" as const, label: "Compare Methods", icon: "📊" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Long Context Techniques</h1>
        <p className="text-muted-foreground max-w-3xl">
          Explore advanced positional encoding algorithms for extending transformer context windows.
          Experiment with RoPE scaling, YaRN, and ALiBi zero-shot extrapolation.
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
                layoutId="activeTabLongCtx"
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
        {activeTab === "rope" && (
          <motion.div
            key="rope"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">RoPE Topology</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  Rotary Position Embeddings explicitly map token positions into representations by rotating the query/key projections.
                </p>
              </div>
              <div className="space-y-5">
                <div>
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block mb-2">Scaling Method</label>
                  <select value={ropeConfig.method} onChange={(e) => setRopeConfig((prev) => ({ ...prev, method: e.target.value }))}
                    className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-2.5 text-sm text-foreground/90 focus:border-primary/50 outline-none transition-all">
                    <option value="none">None (Standard Base RoPE)</option>
                    <option value="linear">Linear Interpolation</option>
                    <option value="ntk">NTK-Aware Scaling</option>
                    <option value="yarn">YaRN (Yet another RoPE extensioN)</option>
                  </select>
                </div>
                {[
                  { key: "dim", label: "Head Dimension (dim)", min: 32, max: 512, step: 32, color: "accent-primary" },
                  { key: "max_position", label: "Max Position Limit", min: 128, max: 16384, step: 128, color: "accent-accent" },
                  { key: "scaling_factor", label: "Context Scaling Factor", min: 1, max: 32, step: 1, color: "accent-emerald-500" },
                ].map((p) => (
                  <div key={p.key}>
                    <div className="flex justify-between items-end mb-2">
                      <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">{p.label}</label>
                      <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">{(ropeConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={(ropeConfig as any)[p.key]}
                      onChange={(e) => setRopeConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))}
                      className={`w-full ${p.color} h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20`} />
                  </div>
                ))}
              </div>
              <div className="flex gap-3 pt-2">
                <button onClick={() => ropeMutation.mutate()} disabled={ropeMutation.isPending}
                  className="flex-1 py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                  {ropeMutation.isPending ? "Projecting..." : "Compute Wave Frequencies"}
                </button>
                <button onClick={() => ropeCompareMutation.mutate()} disabled={ropeCompareMutation.isPending}
                  className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-foreground/90 disabled:opacity-50 transition-all text-sm font-medium">
                  {ropeCompareMutation.isPending ? "Benchmarking..." : "Compare Scaling Laws"}
                </button>
              </div>
            </div>

            <div className="glass rounded-2xl p-6 flex flex-col">
              <h2 className="text-lg font-semibold text-foreground/90 mb-4 border-b border-white/10 pb-4">RoPE Spectral Analysis</h2>
              {ropeMutation.data ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6 flex-1 flex flex-col">
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    {Object.entries(ropeMutation.data as Record<string, any>).filter(([k]) => k !== "frequencies").map(([k, v], i) => (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.05 }}
                        key={k}
                        className="bg-black/40 border border-white/5 rounded-xl p-3 flex flex-col justify-center"
                      >
                        <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground/80 mb-1.5">{k.replace(/_/g, " ")}</div>
                        <div className="font-mono text-sm font-bold text-accent">{typeof v === "number" ? v.toFixed(4) : JSON.stringify(v)}</div>
                      </motion.div>
                    ))}
                  </div>

                  {(ropeMutation.data as any).frequencies && (
                    <div className="bg-black/20 border border-white/5 rounded-xl p-5 flex-1 flex flex-col mt-2">
                      <div className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground mb-4">Spectral Wave Frequencies Map</div>
                      <div className="flex-1 flex items-end gap-[1px] min-h-[140px] bg-black/40 rounded p-1">
                        {((ropeMutation.data as any).frequencies || []).slice(0, 48).map((f: number, i: number) => {
                          const maxF = Math.max(...((ropeMutation.data as any).frequencies || []).slice(0, 48));
                          const h = maxF > 0 ? (f / maxF) * 100 : 0;
                          return (
                            <motion.div
                              initial={{ height: 0 }} animate={{ height: `${Math.max(h, 1)}%` }} transition={{ duration: 0.8, delay: i * 0.01 }}
                              key={i}
                              className="flex-1 bg-gradient-to-t from-primary/80 to-accent-400/90 rounded-t-sm opacity-80"
                            />
                          );
                        })}
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="flex-1 min-h-[250px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                  Compute RoPE frequencies to project the spectral map
                </div>
              )}

              {ropeCompareMutation.data && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-3 mt-6 pt-6 border-t border-white/10">
                  <div className="text-[11px] font-bold uppercase tracking-widest text-primary mb-3">Extrapolation Law Comparison</div>
                  <div className="space-y-2">
                    {toMethodsArray((ropeCompareMutation.data as any).methods).map((m: any, i: number) => (
                      <motion.div
                        initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
                        key={m.method}
                        className="bg-black/40 border border-white/5 rounded-lg p-3 flex justify-between items-center group hover:bg-white/5 transition-colors"
                      >
                        <span className="text-xs font-semibold uppercase tracking-wider text-foreground/80 group-hover:text-primary transition-colors">{m.method}</span>
                        <div className="flex gap-4 font-mono text-xs">
                          <span className="text-muted-foreground">Min: <span className="text-blue-400">{m.min_freq?.toFixed(4)}</span></span>
                          <span className="text-muted-foreground">Max: <span className="text-pink-400">{m.max_freq?.toFixed(4)}</span></span>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}

        {activeTab === "alibi" && (
          <motion.div
            key="alibi"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">ALiBi Configuration</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  Attention with Linear Biases injects static linear penalties proportional to token distances,
                  unlocking massive zero-shot context length extension inherently.
                </p>
              </div>
              <div className="space-y-5">
                {[
                  { key: "num_heads", label: "Attention Heads", min: 1, max: 32, step: 1, color: "accent-primary" },
                  { key: "seq_len", label: "Sub-Sequence Trace Length", min: 4, max: 64, step: 4, color: "accent-accent" },
                ].map((p) => (
                  <div key={p.key}>
                    <div className="flex justify-between items-end mb-2">
                      <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">{p.label}</label>
                      <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">{(alibiConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={(alibiConfig as any)[p.key]}
                      onChange={(e) => setAlibiConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))}
                      className={`w-full ${p.color} h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20`} />
                  </div>
                ))}
              </div>
              <div className="flex gap-3 pt-2">
                <button onClick={() => alibiMutation.mutate()} disabled={alibiMutation.isPending}
                  className="flex-1 py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                  {alibiMutation.isPending ? "Compiling..." : "Generate ALiBi Mask"}
                </button>
                <button onClick={() => extMutation.mutate()} disabled={extMutation.isPending}
                  className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-foreground/90 disabled:opacity-50 transition-all text-sm font-medium">
                  {extMutation.isPending ? "Validating..." : "Extrapolate Context"}
                </button>
              </div>
            </div>

            <div className="glass rounded-2xl p-6 flex flex-col">
              <h2 className="text-lg font-semibold text-foreground/90 border-b border-white/10 pb-4 mb-4">Linear Bias Matrix Formulation</h2>
              {alibiMutation.data ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6 flex-1 flex flex-col">
                  <div>
                    <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-3">Global Distance Slopes (m)</h3>
                    <div className="grid grid-cols-2 gap-2">
                      {((alibiMutation.data as any).slopes || []).slice(0, 16).map((s: number, i: number) => (
                        <div key={i} className="bg-black/40 border border-white/5 rounded-lg p-2 flex justify-between items-center text-xs">
                          <span className="text-muted-foreground/60 font-semibold uppercase tracking-wider">H_{i}</span>
                          <span className="font-mono text-emerald-400">{s?.toFixed(6)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {(alibiMutation.data as any).bias_matrix_sample && (
                    <div className="bg-black/20 border border-white/5 rounded-xl p-4 flex-1">
                      <div className="text-[11px] font-bold uppercase tracking-widest text-primary mb-3">Penalty Matrix Trace (Head 0)</div>
                      <div className="font-mono text-[10px] overflow-x-auto custom-scrollbar pb-2">
                        {((alibiMutation.data as any).bias_matrix_sample || []).slice(0, 8).map((row: number[], i: number) => (
                          <div key={i} className="flex gap-1.5 mb-1.5">
                            {row.slice(0, 12).map((v: number, j: number) => (
                              <motion.div
                                initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (i * 12 + j) * 0.01 }}
                                key={j} className={`w-10 text-center py-1 rounded bg-black/60 border border-white/5 ${v < -3 ? "text-pink-400" : v < -1 ? "text-amber-400" : "text-emerald-400"}`}
                              >
                                {v.toFixed(1)}
                              </motion.div>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="flex-1 min-h-[250px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                  Compute bias matrix to view cross-attention penalties
                </div>
              )}

              {extMutation.data && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-3 mt-6 pt-6 border-t border-white/10">
                  <div className="text-[11px] font-bold uppercase tracking-widest text-accent mb-3">Context Extension Analytics</div>
                  <div className="space-y-2">
                    {((extMutation.data as any).extension_analysis || []).map((e: any, i: number) => (
                      <motion.div
                        initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
                        key={e.test_length}
                        className="bg-black/40 border border-white/5 rounded-lg p-3 flex justify-between items-center text-xs"
                      >
                        <span className="font-bold text-foreground/80 uppercase tracking-widest text-[10px]">Pos: {e.test_length}</span>
                        <div className="flex items-center gap-6 font-mono">
                          <span className="flex items-center gap-2"><span className="text-muted-foreground/50">Scale:</span> <span className="text-accent bg-accent/10 px-1.5 rounded">{e.extension_ratio?.toFixed(1)}x</span></span>
                          <span className="flex items-center gap-2"><span className="text-muted-foreground/50">Max Bias:</span> <span className="text-pink-400">{e.max_bias?.toFixed(1)}</span></span>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
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
            <div className="glass rounded-2xl p-6 flex flex-col items-center max-w-2xl mx-auto text-center">
              <div className="w-12 h-12 rounded-2xl bg-primary/20 flex items-center justify-center text-primary mb-4 shadow-lg shadow-primary/20">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20" /><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" /></svg>
              </div>
              <h2 className="text-xl font-bold text-foreground/90 mb-2">Analyze Universal Positional Architectures</h2>
              <p className="text-sm text-muted-foreground mb-6 max-w-md">
                Cross-evaluate absolute, relative, Sinusoidal, Learned, standard RoPE, advanced YaRN scaling, and native ALiBi embeddings.
              </p>
              <button onClick={() => compareAllMutation.mutate()} disabled={compareAllMutation.isPending}
                className="w-full md:w-auto px-8 py-3 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {compareAllMutation.isPending ? "Executing Universal Benchmark..." : "Execute Comprehensive Architecture Review"}
              </button>
            </div>

            {compareAllMutation.data && (
              <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="glass rounded-2xl p-6">
                <div className="flex justify-between items-center mb-6 pb-4 border-b border-white/10">
                  <h2 className="text-lg font-semibold text-foreground/90">Macro Architectural Benchmarks</h2>
                  <div className="text-[10px] font-bold text-primary uppercase tracking-widest bg-primary/10 px-3 py-1 rounded-full">Results Generated</div>
                </div>
                <div className="overflow-x-auto rounded-xl border border-white/5 bg-black/40">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10 bg-white/5">
                        <th className="text-left py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Positional Method</th>
                        <th className="text-right py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Base Context (N)</th>
                        <th className="text-center py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Learnable Parameters</th>
                        <th className="text-center py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Relative Distance Form</th>
                        <th className="text-right py-4 px-5 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Extrapolation Capacity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {toMethodsArray((compareAllMutation.data as any).methods).map((m: any, i: number) => (
                        <motion.tr
                          initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                          key={i}
                          className="border-b last:border-0 border-white/5 hover:bg-white/5 transition-colors group"
                        >
                          <td className="py-4 px-5 font-semibold text-foreground/80 group-hover:text-primary transition-colors">{m.method}</td>
                          <td className="py-4 px-5 text-right font-mono text-blue-400">{m.max_context || m.max_extrapolation || "N/A"}</td>
                          <td className="py-4 px-5 text-center">
                            <span className={`inline-flex items-center justify-center w-6 h-6 rounded-md text-xs font-bold ${m.learnable ? "bg-amber-500/20 text-amber-500" : "bg-white/5 text-muted-foreground/40"}`}>
                              {m.learnable ? "✓" : "✗"}
                            </span>
                          </td>
                          <td className="py-4 px-5 text-center">
                            <span className={`inline-flex items-center justify-center w-6 h-6 rounded-md text-xs font-bold ${(m.relative ?? m.relative_position) ? "bg-emerald-500/20 text-emerald-500" : "bg-white/5 text-muted-foreground/40"}`}>
                              {(m.relative ?? m.relative_position) ? "✓" : "✗"}
                            </span>
                          </td>
                          <td className="py-4 px-5 text-right text-xs text-muted-foreground/80 max-w-[200px] truncate" title={m.extrapolation || m.description}>
                            {m.extrapolation || m.description || "-"}
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
