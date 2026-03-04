"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { distributedApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function DistributedPage() {
  const [activeTab, setActiveTab] = useState<"compare" | "zero" | "pipeline">("compare");
  const [config, setConfig] = useState({ num_gpus: 4, model_params_m: 125, batch_size: 32, gpu_memory_gb: 24 });
  const [zeroStage, setZeroStage] = useState(1);
  const [pipeConfig, setPipeConfig] = useState({ num_gpus: 4, model_params_m: 125, num_layers: 24, num_micro_batches: 8, gpu_memory_gb: 24 });

  const strategies = useQuery<any>({ queryKey: ["dist-strategies"], queryFn: () => distributedApi.strategies() });

  const compareMutation = useMutation<any>({ mutationFn: () => distributedApi.compareAll(config) });
  const dpMutation = useMutation<any>({ mutationFn: () => distributedApi.dataParallel(config) });
  const mpMutation = useMutation<any>({ mutationFn: () => distributedApi.modelParallel(config) });
  const zeroMutation = useMutation<any>({
    mutationFn: () => distributedApi.zero({ ...config, stage: zeroStage }),
  });
  const pipeMutation = useMutation<any>({ mutationFn: () => distributedApi.pipelineParallel(pipeConfig) });

  const tabs = [
    { id: "compare" as const, label: "Strategy Comparison", icon: "📊" },
    { id: "zero" as const, label: "ZeRO Optimizer", icon: "🔄" },
    { id: "pipeline" as const, label: "Pipeline Parallel", icon: "🏗️" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Multi-GPU Cluster Environment</h1>
        <p className="text-muted-foreground max-w-3xl">
          Simulate distributed training architectures. Analyze memory footprints and communication overheads
          across Data Parallel, Tensor Parallel, Pipeline Parallel strategies, and the ZeRO Optimizer suite.
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
                layoutId="activeTabDistrib"
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

      {/* Shared Config */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass rounded-2xl p-6"
      >
        <div className="pb-4 border-b border-white/10 mb-5">
          <h2 className="text-lg font-semibold text-foreground/90">Global Cluster Configuration</h2>
          <p className="text-xs text-muted-foreground mt-1">Hardware specs shared across all simulation models.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
          {[
            { key: "num_gpus", label: "Compute Nodes (GPUs)", min: 1, max: 64, step: 1, color: "accent-primary" },
            { key: "model_params_m", label: "Model Architecture (M Params)", min: 1, max: 7000, step: 50, color: "accent-accent" },
            { key: "batch_size", label: "Global Batch Size", min: 1, max: 512, step: 8, color: "accent-emerald-500" },
            { key: "gpu_memory_gb", label: "VRAM per Node (GB)", min: 4, max: 80, step: 4, color: "accent-purple-500" },
          ].map((p) => (
            <div key={p.key}>
              <div className="flex justify-between items-end mb-2">
                <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest block">{p.label}</label>
                <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">{(config as any)[p.key]}</span>
              </div>
              <input type="range" min={p.min} max={p.max} step={p.step} value={(config as any)[p.key]}
                onChange={(e) => setConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))}
                className={`w-full ${p.color} h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20`} />
            </div>
          ))}
        </div>
      </motion.div>

      <AnimatePresence mode="wait">
        {activeTab === "compare" && (
          <motion.div
            key="compare"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="flex gap-3 bg-black/20 p-2 rounded-2xl border border-white/5 w-fit">
              <button onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}
                className="px-6 py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {compareMutation.isPending ? "Simulating Benchmark..." : "Compare Overheads"}
              </button>
              <button onClick={() => dpMutation.mutate()} disabled={dpMutation.isPending}
                className="px-6 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-foreground/90 disabled:opacity-50 transition-all text-sm font-medium">
                Simulate DP
              </button>
              <button onClick={() => mpMutation.mutate()} disabled={mpMutation.isPending}
                className="px-6 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-foreground/90 disabled:opacity-50 transition-all text-sm font-medium">
                Simulate TP
              </button>
            </div>

            {compareMutation.data && (
              <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="glass rounded-2xl p-6">
                <div className="pb-4 border-b border-white/10 mb-4">
                  <h2 className="text-lg font-semibold text-foreground/90">Bottleneck Profiling & Strategy Comparison</h2>
                </div>
                <div className="overflow-x-auto rounded-xl border border-white/5 bg-black/40">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10 bg-white/5">
                        <th className="text-left py-3 px-4 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Distribution Layout</th>
                        <th className="text-right py-3 px-4 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">VRAM Overhead (GB)</th>
                        <th className="text-right py-3 px-4 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Comm Overhead</th>
                        <th className="text-right py-3 px-4 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Relative Speedup</th>
                        <th className="text-center py-3 px-4 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">OOM State</th>
                      </tr>
                    </thead>
                    <tbody>
                      {((compareMutation.data as any).strategies || []).map((s: any, i: number) => (
                        <tr key={i} className="border-b last:border-0 border-white/5 hover:bg-white/5 transition-colors group">
                          <td className="py-3 px-4 font-medium text-primary-300 group-hover:text-primary-400">{s.strategy}</td>
                          <td className="py-3 px-4 text-right font-mono text-muted-foreground relative">
                            {s.memory_per_gpu_gb?.toFixed(1)}
                            {s.memory_per_gpu_gb > config.gpu_memory_gb && (
                              <span className="absolute right-0 top-1/2 -translate-y-1/2 -translate-x-full pr-12 text-[10px] text-red-500">OOM</span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-right text-xs text-muted-foreground/80">{s.communication}</td>
                          <td className="py-3 px-4 text-right font-mono">
                            <span className="text-emerald-400 font-bold bg-emerald-500/10 px-2 py-0.5 rounded">{s.speedup?.toFixed(1)}x</span>
                          </td>
                          <td className="py-3 px-4 text-center">
                            <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${s.feasible ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}`}>
                              {s.feasible ? "✓" : "✗"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {dpMutation.data && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-2xl p-6">
                <h2 className="text-lg font-semibold mb-3 border-b border-white/10 pb-3 text-foreground/90">Data Parallel Profiling</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(dpMutation.data as Record<string, any>).map(([k, v]) => (
                    <div key={k} className="bg-black/40 border border-white/5 rounded-xl p-4 flex flex-col items-center justify-center text-center">
                      <div className="text-[10px] uppercase tracking-widest text-muted-foreground/80 mb-2">{k.replace(/_/g, " ")}</div>
                      <div className="font-mono text-xl font-bold text-accent">{typeof v === "number" ? v.toFixed(2) : String(v)}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {activeTab === "zero" && (
          <motion.div
            key="zero"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">ZeRO Optimization Constraints</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  Zero Redundancy Optimizer progressively partitions optimizer states, gradients, and parameters to fit giant models.
                </p>
              </div>
              <div className="bg-black/20 p-5 rounded-2xl border border-white/5 space-y-4">
                <div className="flex justify-between items-end mb-2">
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">ZeRO Target Stage</label>
                  <span className="text-xl font-mono font-bold text-primary">Stage {zeroStage}</span>
                </div>
                <input type="range" min={0} max={3} step={1} value={zeroStage}
                  onChange={(e) => setZeroStage(parseInt(e.target.value))} className="w-full accent-primary h-2 bg-black/60 outline-none rounded-full appearance-none focus:ring-1 focus:ring-white/20" />
                <div className="flex justify-between text-[10px] uppercase font-bold tracking-wider text-muted-foreground/60 pt-2 pb-1 relative px-1">
                  <span className={zeroStage === 0 ? "text-primary" : ""}>Baseline</span>
                  <span className={`absolute left-1/3 -translate-x-1/2 ${zeroStage === 1 ? "text-primary" : ""}`}>Optimizer</span>
                  <span className={`absolute left-2/3 -translate-x-1/2 ${zeroStage === 2 ? "text-primary" : ""}`}>+Gradients</span>
                  <span className={zeroStage === 3 ? "text-primary" : ""}>+Params</span>
                </div>
              </div>
              <button onClick={() => zeroMutation.mutate()} disabled={zeroMutation.isPending}
                className="w-full py-3 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {zeroMutation.isPending ? "Calculating Partitions..." : "Analyze State Partitioning"}
              </button>
            </div>

            {zeroMutation.data && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-2xl p-6">
                <h2 className="text-lg font-semibold mb-4 border-b border-white/10 pb-3 text-foreground/90">Stage {zeroStage} Memory Footprint</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {Object.entries(zeroMutation.data as Record<string, any>).filter(([k]) => typeof (zeroMutation.data as any)[k] !== "object").map(([k, v], i) => (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.05 }}
                      key={k}
                      className="bg-black/40 border border-white/5 rounded-xl p-4 flex flex-col"
                    >
                      <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground/80 mb-3">{k.replace(/_/g, " ")}</div>
                      <div className={`font-mono text-2xl font-bold mt-auto ${k.includes('memory') ? 'text-emerald-400' : 'text-blue-400'}`}>
                        {typeof v === "number" ? v.toFixed(2) : String(v)}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {activeTab === "pipeline" && (
          <motion.div
            key="pipeline"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">Pipeline Schedule Layout</h2>
                <p className="text-sm text-muted-foreground mt-1">Split transformer layers across GPUs with advanced micro-batch scheduling to minimize pipeline latency bubbles.</p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 bg-black/20 p-5 border border-white/5 rounded-2xl">
                {[
                  { key: "num_layers", label: "Transformer Layers", min: 4, max: 96, step: 4, color: "accent-blue-500" },
                  { key: "num_micro_batches", label: "Micro-batch Count", min: 1, max: 64, step: 1, color: "accent-purple-500" },
                ].map((p) => (
                  <div key={p.key}>
                    <div className="flex justify-between items-end mb-2">
                      <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">
                        {p.label}
                      </label>
                      <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">
                        {(pipeConfig as any)[p.key]}
                      </span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={(pipeConfig as any)[p.key]}
                      onChange={(e) => setPipeConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))}
                      className={`w-full ${p.color} h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20`} />
                  </div>
                ))}
              </div>
              <button onClick={() => pipeMutation.mutate()} disabled={pipeMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {pipeMutation.isPending ? "Generating 1F1B Schedule..." : "Generate Pipeline Execution Flow"}
              </button>
            </div>

            {pipeMutation.data && (
              <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="glass rounded-2xl p-6">
                <h2 className="text-lg font-semibold mb-4 border-b border-white/10 pb-3 text-foreground/90">Pipeline Efficiency Diagnostics</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {Object.entries(pipeMutation.data as Record<string, any>).filter(([, v]) => typeof v !== "object").map(([k, v]) => (
                    <div key={k} className="bg-black/40 border border-white/5 rounded-xl p-4 flex flex-col justify-center items-center">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground/80 mb-2">{k.replace(/_/g, " ")}</div>
                      <div className="font-mono text-xl font-bold text-emerald-400">{typeof v === "number" ? v.toFixed(3) : String(v)}</div>
                    </div>
                  ))}
                </div>

                {(pipeMutation.data as any).schedule && (
                  <div className="mt-4 pt-4 border-t border-white/10">
                    <h3 className="text-[11px] font-bold uppercase tracking-widest text-primary mb-4 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-primary inline-block"></span> Execution Trace (First 8 GPUs shown)
                    </h3>
                    <div className="bg-black/60 border border-white/5 rounded-xl p-4 overflow-x-auto custom-scrollbar">
                      <div className="space-y-1.5 min-w-max">
                        {((pipeMutation.data as any).schedule || []).slice(0, 8).map((row: any, i: number) => (
                          <div key={i} className="flex gap-2 items-center">
                            <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground/60 w-12 pt-0.5">GPU {row.gpu}</span>
                            <div className="flex gap-1">
                              {(row.steps || []).map((s: string, j: number) => (
                                <motion.div
                                  initial={{ opacity: 0, x: -5 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: j * 0.02 + i * 0.05 }}
                                  key={j}
                                  className={`px-1.5 py-1 rounded text-[10px] font-mono whitespace-nowrap border ${s === "idle" ? "bg-black/40 border-white/5 text-muted-foreground/30" :
                                    s.startsWith('F') ? "bg-blue-500/10 border-blue-500/20 text-blue-400" :
                                      "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                                    }`}
                                >
                                  {s}
                                </motion.div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
        className="glass rounded-2xl p-6"
      >
        <h2 className="text-lg font-semibold mb-4 border-b border-white/10 pb-3 text-foreground/90">Training Topologies Guide</h2>
        {strategies.data && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {((strategies.data as any).strategies || []).map((s: any) => (
              <div key={s.id} className="bg-black/20 border border-white/5 rounded-xl p-5 hover:bg-white/5 transition-colors group">
                <h3 className="font-semibold text-primary text-[11px] uppercase tracking-widest mb-3 flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-primary/40 rounded-full group-hover:bg-primary transition-colors"></div>
                  {s.name}
                </h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {s.description.length > 150 ? s.description.substring(0, 150) + "..." : s.description}
                </p>
              </div>
            ))}
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}
