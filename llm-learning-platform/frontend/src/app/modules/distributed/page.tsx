"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { distributedApi } from "@/lib/api";

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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Distributed Training</h1>
        <p className="text-muted-foreground">
          Explore data parallelism, tensor parallelism, pipeline parallelism, and ZeRO optimizer stages.
        </p>
      </div>

      <div className="flex gap-2 border-b border-border pb-2">
        {tabs.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-colors ${activeTab === tab.id
              ? "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 border-b-2 border-primary-500"
              : "text-muted-foreground hover:text-foreground"}`}>
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Shared Config */}
      <div className="glass rounded-2xl p-6 space-y-4">
        <h2 className="text-lg font-semibold">Cluster Configuration</h2>
        <div className="grid grid-cols-4 gap-4">
          {[
            { key: "num_gpus", label: "GPUs", min: 1, max: 64, step: 1 },
            { key: "model_params_m", label: "Model Params (M)", min: 1, max: 7000, step: 50 },
            { key: "batch_size", label: "Batch Size", min: 1, max: 512, step: 8 },
            { key: "gpu_memory_gb", label: "GPU Memory (GB)", min: 4, max: 80, step: 4 },
          ].map((p) => (
            <div key={p.key}>
              <label className="text-sm font-medium mb-1 block">
                {p.label}: <span className="text-primary-500">{(config as any)[p.key]}</span>
              </label>
              <input type="range" min={p.min} max={p.max} step={p.step} value={(config as any)[p.key]}
                onChange={(e) => setConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))} className="w-full" />
            </div>
          ))}
        </div>
      </div>

      {activeTab === "compare" && (
        <div className="space-y-6">
          <div className="flex gap-3">
            <button onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}
              className="px-6 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {compareMutation.isPending ? "Comparing..." : "Compare All Strategies"}
            </button>
            <button onClick={() => dpMutation.mutate()} disabled={dpMutation.isPending}
              className="px-4 py-2 rounded-xl border border-border hover:bg-muted transition-colors text-sm">
              Data Parallel
            </button>
            <button onClick={() => mpMutation.mutate()} disabled={mpMutation.isPending}
              className="px-4 py-2 rounded-xl border border-border hover:bg-muted transition-colors text-sm">
              Model Parallel
            </button>
          </div>

          {compareMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Strategy Comparison</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-border">
                    <th className="text-left py-2 px-3">Strategy</th>
                    <th className="text-right py-2 px-3">Memory/GPU (GB)</th>
                    <th className="text-right py-2 px-3">Communication</th>
                    <th className="text-right py-2 px-3">Speedup</th>
                    <th className="text-right py-2 px-3">Feasible</th>
                  </tr></thead>
                  <tbody>
                    {((compareMutation.data as any).strategies || []).map((s: any, i: number) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-muted/50">
                        <td className="py-2 px-3 font-medium">{s.strategy}</td>
                        <td className="py-2 px-3 text-right font-mono">{s.memory_per_gpu_gb?.toFixed(1)}</td>
                        <td className="py-2 px-3 text-right text-xs">{s.communication}</td>
                        <td className="py-2 px-3 text-right font-mono text-primary-500">{s.speedup?.toFixed(1)}x</td>
                        <td className="py-2 px-3 text-right">
                          <span className={s.feasible ? "text-green-500" : "text-red-500"}>{s.feasible ? "✓" : "✗"}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {dpMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-3">Data Parallel Analysis</h2>
              <div className="grid grid-cols-3 gap-4">
                {Object.entries(dpMutation.data as Record<string, any>).map(([k, v]) => (
                  <div key={k} className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground">{k.replace(/_/g, " ")}</div>
                    <div className="font-mono font-bold text-primary-500">{typeof v === "number" ? v.toFixed(2) : String(v)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "zero" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">ZeRO Optimizer</h2>
            <p className="text-sm text-muted-foreground">
              Zero Redundancy Optimizer progressively partitions optimizer states, gradients, and parameters.
            </p>
            <div>
              <label className="text-sm font-medium mb-1 block">ZeRO Stage: <span className="text-primary-500">{zeroStage}</span></label>
              <input type="range" min={0} max={3} step={1} value={zeroStage}
                onChange={(e) => setZeroStage(parseInt(e.target.value))} className="w-full" />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>Stage 0 (None)</span><span>Stage 1 (OS)</span><span>Stage 2 (+Grad)</span><span>Stage 3 (+Param)</span>
              </div>
            </div>
            <button onClick={() => zeroMutation.mutate()} disabled={zeroMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {zeroMutation.isPending ? "Analyzing..." : "Analyze ZeRO Stage"}
            </button>
          </div>
          {zeroMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-3">ZeRO Stage {zeroStage} Results</h2>
              <div className="grid grid-cols-3 gap-4">
                {Object.entries(zeroMutation.data as Record<string, any>).filter(([k]) => typeof (zeroMutation.data as any)[k] !== "object").map(([k, v]) => (
                  <div key={k} className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground">{k.replace(/_/g, " ")}</div>
                    <div className="font-mono font-bold text-primary-500">{typeof v === "number" ? v.toFixed(2) : String(v)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "pipeline" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Pipeline Parallelism</h2>
            <p className="text-sm text-muted-foreground">Split layers across GPUs with micro-batch scheduling to reduce pipeline bubbles.</p>
            <div className="grid grid-cols-2 gap-4">
              {[
                { key: "num_layers", label: "Num Layers", min: 4, max: 96, step: 4 },
                { key: "num_micro_batches", label: "Micro-batches", min: 1, max: 64, step: 1 },
              ].map((p) => (
                <div key={p.key}>
                  <label className="text-sm font-medium mb-1 block">
                    {p.label}: <span className="text-primary-500">{(pipeConfig as any)[p.key]}</span>
                  </label>
                  <input type="range" min={p.min} max={p.max} step={p.step} value={(pipeConfig as any)[p.key]}
                    onChange={(e) => setPipeConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))} className="w-full" />
                </div>
              ))}
            </div>
            <button onClick={() => pipeMutation.mutate()} disabled={pipeMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {pipeMutation.isPending ? "Analyzing..." : "Analyze Pipeline"}
            </button>
          </div>
          {pipeMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-3">Pipeline Analysis</h2>
              <div className="grid grid-cols-3 gap-4">
                {Object.entries(pipeMutation.data as Record<string, any>).filter(([, v]) => typeof v !== "object").map(([k, v]) => (
                  <div key={k} className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground">{k.replace(/_/g, " ")}</div>
                    <div className="font-mono font-bold text-primary-500">{typeof v === "number" ? v.toFixed(3) : String(v)}</div>
                  </div>
                ))}
              </div>
              {(pipeMutation.data as any).schedule && (
                <div className="mt-4">
                  <div className="text-sm font-semibold mb-2">Pipeline Schedule</div>
                  <div className="bg-background rounded-xl p-3 overflow-x-auto">
                    <div className="space-y-1">
                      {((pipeMutation.data as any).schedule || []).slice(0, 8).map((row: any, i: number) => (
                        <div key={i} className="flex gap-1">
                          <span className="text-xs font-mono w-16 text-muted-foreground">GPU {row.gpu}:</span>
                          <div className="flex gap-0.5">
                            {(row.steps || []).map((s: string, j: number) => (
                              <div key={j} className={`px-1 py-0.5 rounded text-[10px] font-mono ${
                                s === "idle" ? "bg-muted text-muted-foreground" : "bg-primary-200 dark:bg-primary-800 text-primary-700 dark:text-primary-300"
                              }`}>{s}</div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {strategies.data && (
        <div className="glass rounded-2xl p-6 space-y-4">
          <h2 className="text-lg font-semibold">Parallelism Strategies</h2>
          <div className="grid grid-cols-3 gap-4">
            {((strategies.data as any).strategies || []).map((s: any) => (
              <div key={s.id} className="bg-background rounded-xl p-4">
                <h3 className="font-semibold text-primary-500 text-sm">{s.name}</h3>
                <p className="text-xs text-muted-foreground mt-1">{s.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
