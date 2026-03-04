"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { loraApi } from "@/lib/api";

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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">LoRA Studio</h1>
        <p className="text-muted-foreground">
          Explore Low-Rank Adaptation — build LoRA layers, visualize weight decomposition,
          experiment with QLoRA quantization, and compare parameter-efficient methods.
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

      {activeTab === "lora" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">LoRA Configuration</h2>
            <p className="text-sm text-muted-foreground">LoRA decomposes weight updates: ΔW = A × B where A ∈ ℝ^(d×r) and B ∈ ℝ^(r×d).</p>
            {[
              { key: "d_model", label: "Model Dimension", min: 32, max: 1024, step: 32 },
              { key: "num_layers", label: "Num Layers", min: 1, max: 12, step: 1 },
              { key: "rank", label: "LoRA Rank (r)", min: 1, max: 64, step: 1 },
              { key: "alpha", label: "LoRA Alpha", min: 1, max: 128, step: 1 },
              { key: "dropout", label: "Dropout", min: 0, max: 0.5, step: 0.05 },
            ].map((p) => (
              <div key={p.key}>
                <label className="text-sm font-medium mb-1 block">
                  {p.label}: <span className="text-primary-500">{(loraConfig as any)[p.key]}</span>
                </label>
                <input type="range" min={p.min} max={p.max} step={p.step} value={(loraConfig as any)[p.key]}
                  onChange={(e) => setLoraConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))} className="w-full" />
              </div>
            ))}
            <div className="flex gap-2">
              <button onClick={() => createMutation.mutate()} disabled={createMutation.isPending}
                className="flex-1 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {createMutation.isPending ? "Creating..." : "Create LoRA Model"}
              </button>
              <button onClick={() => forwardMutation.mutate()} disabled={forwardMutation.isPending}
                className="flex-1 py-2 rounded-xl bg-accent-600 text-white font-medium hover:bg-accent-700 disabled:opacity-50 transition-colors">
                {forwardMutation.isPending ? "Running..." : "Forward Pass"}
              </button>
            </div>
          </div>
          <div className="space-y-6">
            {createMutation.data && (
              <div className="glass rounded-2xl p-6 space-y-4">
                <h2 className="text-lg font-semibold">Model Summary</h2>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(createMutation.data as Record<string, any>).map(([k, v]) => (
                    <div key={k} className="bg-background rounded-xl p-3">
                      <div className="text-xs text-muted-foreground">{k.replace(/_/g, " ")}</div>
                      <div className="font-mono font-bold text-primary-500">
                        {typeof v === "number" ? v.toLocaleString() : typeof v === "object" ? JSON.stringify(v) : String(v)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {forwardMutation.data && (
              <div className="glass rounded-2xl p-6 space-y-4">
                <h2 className="text-lg font-semibold">Forward Pass</h2>
                <div className="grid grid-cols-2 gap-3">
                  {["A_shape", "B_shape", "delta_w_norm", "scaling"].map((k) => (
                    <div key={k} className="bg-background rounded-xl p-3">
                      <div className="text-xs text-muted-foreground">{k.replace(/_/g, " ")}</div>
                      <div className="font-mono text-sm">{JSON.stringify((forwardMutation.data as any)[k])}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {ranks.data && (
              <div className="glass rounded-2xl p-6 space-y-3">
                <h2 className="text-lg font-semibold">Rank Guide</h2>
                <div className="space-y-2">
                  {((ranks.data as any).ranks || []).map((r: any) => (
                    <div key={r.rank} className="flex items-center gap-3 bg-background rounded-lg p-2">
                      <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center font-bold text-primary-500 text-xs">r={r.rank}</div>
                      <div className="flex-1">
                        <div className="text-sm font-medium">{r.use_case}</div>
                        <div className="text-xs text-muted-foreground">Quality: {r.quality}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          <div className="glass rounded-2xl p-6 space-y-4 col-span-2">
            <h2 className="text-lg font-semibold">LoRA Training</h2>
            <div className="flex items-end gap-4">
              {[{ key: "rank", label: "Rank", min: 1, max: 64, step: 1 }, { key: "num_steps", label: "Steps", min: 5, max: 100, step: 5 }].map((p) => (
                <div key={p.key} className="flex-1">
                  <label className="text-sm font-medium mb-1 block">{p.label}: {(trainConfig as any)[p.key]}</label>
                  <input type="range" min={p.min} max={p.max} step={p.step} value={(trainConfig as any)[p.key]}
                    onChange={(e) => setTrainConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))} className="w-full" />
                </div>
              ))}
              <button onClick={() => trainMutation.mutate()} disabled={trainMutation.isPending}
                className="px-6 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {trainMutation.isPending ? "Training..." : "Run Training"}
              </button>
            </div>
            {trainMutation.data && (
              <div className="bg-background rounded-xl p-4">
                <div className="flex items-end gap-0.5 h-32">
                  {((trainMutation.data as any).training_metrics || []).map((m: any, i: number) => {
                    const losses = ((trainMutation.data as any).training_metrics || []).map((x: any) => x.loss);
                    const max = Math.max(...losses); const min = Math.min(...losses);
                    const h = ((m.loss - min) / (max - min || 1)) * 100;
                    return <div key={i} className="flex-1 bg-primary-500/60 rounded-t transition-all" style={{ height: `${Math.max(100 - h, 2)}%` }}
                      title={`Step ${m.step}: loss=${m.loss.toFixed(4)}`} />;
                  })}
                </div>
                <div className="text-xs text-muted-foreground mt-2 text-center">Training Loss (lower = better)</div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "qlora" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">NF4 Quantization Analysis</h2>
            <p className="text-sm text-muted-foreground">QLoRA uses 4-bit NormalFloat quantization with double quantization to reduce memory.</p>
            {[{ key: "rows", label: "Matrix Rows", min: 32, max: 1024, step: 32 }, { key: "cols", label: "Matrix Cols", min: 32, max: 1024, step: 32 }].map((p) => (
              <div key={p.key}>
                <label className="text-sm font-medium mb-1 block">{p.label}: <span className="text-primary-500">{(quantConfig as any)[p.key]}</span></label>
                <input type="range" min={p.min} max={p.max} step={p.step} value={(quantConfig as any)[p.key]}
                  onChange={(e) => setQuantConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))} className="w-full" />
              </div>
            ))}
            <button onClick={() => quantMutation.mutate()} disabled={quantMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {quantMutation.isPending ? "Analyzing..." : "Analyze Quantization"}
            </button>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Quantization Results</h2>
            {quantMutation.data ? (
              <div className="space-y-3">
                {Object.entries(quantMutation.data as Record<string, any>).map(([k, v]) => (
                  <div key={k} className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground">{k.replace(/_/g, " ")}</div>
                    <div className="font-mono text-sm text-primary-500">{typeof v === "number" ? v.toFixed(4) : JSON.stringify(v)}</div>
                  </div>
                ))}
              </div>
            ) : (<div className="text-center text-muted-foreground py-12">Run quantization analysis to see results</div>)}
          </div>
        </div>
      )}

      {activeTab === "compare" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Parameter-Efficient Fine-Tuning Comparison</h2>
            <p className="text-sm text-muted-foreground">Compare Full Fine-Tuning, LoRA, QLoRA, and Prefix Tuning.</p>
            <button onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}
              className="px-6 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {compareMutation.isPending ? "Comparing..." : "Run PEFT Comparison"}
            </button>
          </div>
          {compareMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Comparison Results</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-border">
                    <th className="text-left py-2 px-3">Method</th><th className="text-right py-2 px-3">Trainable</th>
                    <th className="text-right py-2 px-3">Total</th><th className="text-right py-2 px-3">%</th>
                    <th className="text-right py-2 px-3">Memory (MB)</th><th className="text-right py-2 px-3">Quality</th>
                  </tr></thead>
                  <tbody>
                    {toMethodsArray((compareMutation.data as any).methods).map((m: any, i: number) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-muted/50">
                        <td className="py-2 px-3 font-medium">{m.method}</td>
                        <td className="py-2 px-3 text-right font-mono">{m.trainable_params?.toLocaleString()}</td>
                        <td className="py-2 px-3 text-right font-mono">{(m.total_params || (compareMutation.data as any)?.total_base_params)?.toLocaleString()}</td>
                        <td className="py-2 px-3 text-right font-mono text-primary-500">{(m.pct_trainable ?? m.percentage)?.toFixed?.(2)}%</td>
                        <td className="py-2 px-3 text-right font-mono">{m.memory_mb?.toFixed(1)}</td>
                        <td className="py-2 px-3 text-right font-mono">{m.estimated_quality?.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
