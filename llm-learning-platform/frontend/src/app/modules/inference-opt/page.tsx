"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { inferenceOptApi } from "@/lib/api";

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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Inference Optimization</h1>
        <p className="text-muted-foreground">
          Explore KV caching, weight quantization, and speculative decoding to make LLM inference faster and cheaper.
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

      {activeTab === "kvcache" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">KV Cache Analysis</h2>
            <p className="text-sm text-muted-foreground">Analyze how the key-value cache grows during autoregressive generation.</p>
            {[
              { key: "num_layers", label: "Layers", min: 1, max: 32, step: 1 },
              { key: "num_heads", label: "Heads", min: 1, max: 32, step: 1 },
              { key: "head_dim", label: "Head Dim", min: 16, max: 128, step: 16 },
              { key: "prompt_len", label: "Prompt Length", min: 1, max: 256, step: 5 },
              { key: "gen_len", label: "Generation Length", min: 10, max: 200, step: 10 },
            ].map((p) => (
              <div key={p.key}>
                <label className="text-sm font-medium mb-1 block">
                  {p.label}: <span className="text-primary-500">{(kvConfig as any)[p.key]}</span>
                </label>
                <input type="range" min={p.min} max={p.max} step={p.step} value={(kvConfig as any)[p.key]}
                  onChange={(e) => setKvConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))} className="w-full" />
              </div>
            ))}
            <button onClick={() => kvMutation.mutate()} disabled={kvMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {kvMutation.isPending ? "Analyzing..." : "Analyze KV Cache"}
            </button>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Cache Growth</h2>
            {kvMutation.data ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground">Final Cache Size</div>
                    <div className="font-mono text-lg font-bold text-primary-500">{(kvMutation.data as any).final_cache_mb?.toFixed(2)} MB</div>
                  </div>
                  <div className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground">Max Speedup</div>
                    <div className="font-mono text-lg font-bold text-primary-500">{(kvMutation.data as any).max_speedup?.toFixed(1)}x</div>
                  </div>
                </div>
                <div className="bg-background rounded-xl p-3">
                  <div className="text-xs text-muted-foreground mb-2">Cache Size Over Time</div>
                  <div className="flex items-end gap-0.5 h-32">
                    {((kvMutation.data as any).steps || []).filter((_: any, i: number) => i % 3 === 0).map((s: any, i: number) => {
                      const maxMb = (kvMutation.data as any).final_cache_mb || 1;
                      const h = (s.cache_mb / maxMb) * 100;
                      return <div key={i} className="flex-1 bg-primary-500/60 rounded-t" style={{ height: `${Math.max(h, 1)}%` }}
                        title={`Token ${s.token}: ${s.cache_mb.toFixed(2)} MB`} />;
                    })}
                  </div>
                </div>
              </div>
            ) : (<div className="text-center text-muted-foreground py-16">Run analysis to see KV cache growth</div>)}
          </div>
        </div>
      )}

      {activeTab === "quantize" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Weight Quantization Comparison</h2>
            <p className="text-sm text-muted-foreground">Compare FP32, FP16, INT8, INT4 quantization quality and memory.</p>
            <div className="flex gap-4 items-end">
              {[{ key: "rows", label: "Rows" }, { key: "cols", label: "Cols" }].map((p) => (
                <div key={p.key} className="flex-1">
                  <label className="text-sm font-medium mb-1 block">{p.label}: {(quantConfig as any)[p.key]}</label>
                  <input type="range" min={64} max={2048} step={64} value={(quantConfig as any)[p.key]}
                    onChange={(e) => setQuantConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))} className="w-full" />
                </div>
              ))}
              <button onClick={() => quantMutation.mutate()} disabled={quantMutation.isPending}
                className="px-6 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {quantMutation.isPending ? "Analyzing..." : "Compare Quantization"}
              </button>
            </div>
          </div>
          {quantMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Quantization Comparison</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-border">
                    <th className="text-left py-2 px-3">Format</th><th className="text-right py-2 px-3">Bits</th>
                    <th className="text-right py-2 px-3">Memory (MB)</th><th className="text-right py-2 px-3">Compression</th>
                    <th className="text-right py-2 px-3">MSE</th><th className="text-right py-2 px-3">Max Error</th>
                  </tr></thead>
                  <tbody>
                    {((quantMutation.data as any).comparisons || []).map((c: any, i: number) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-muted/50">
                        <td className="py-2 px-3 font-medium">{c.format}</td>
                        <td className="py-2 px-3 text-right font-mono">{c.bits}</td>
                        <td className="py-2 px-3 text-right font-mono">{c.memory_mb?.toFixed(3)}</td>
                        <td className="py-2 px-3 text-right font-mono text-primary-500">{c.compression_ratio?.toFixed(1)}x</td>
                        <td className="py-2 px-3 text-right font-mono">{c.mse?.toFixed(6)}</td>
                        <td className="py-2 px-3 text-right font-mono">{c.max_error?.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "speculative" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Speculative Decoding</h2>
            <p className="text-sm text-muted-foreground">Use a small draft model to propose tokens, verified by the large target model in parallel.</p>
            {[
              { key: "total_tokens", label: "Total Tokens", min: 10, max: 500, step: 10 },
              { key: "gamma", label: "Draft Tokens (γ)", min: 1, max: 8, step: 1 },
              { key: "acceptance_rate", label: "Acceptance Rate", min: 0.1, max: 1, step: 0.05 },
            ].map((p) => (
              <div key={p.key}>
                <label className="text-sm font-medium mb-1 block">
                  {p.label}: <span className="text-primary-500">{(specConfig as any)[p.key]}</span>
                </label>
                <input type="range" min={p.min} max={p.max} step={p.step} value={(specConfig as any)[p.key]}
                  onChange={(e) => setSpecConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))} className="w-full" />
              </div>
            ))}
            <button onClick={() => specMutation.mutate()} disabled={specMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {specMutation.isPending ? "Running..." : "Run Speculative Decoding"}
            </button>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Results</h2>
            {specMutation.data ? (
              <div className="space-y-3">
                {Object.entries(specMutation.data as Record<string, any>).map(([k, v]) => (
                  <div key={k} className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground">{k.replace(/_/g, " ")}</div>
                    <div className="font-mono font-bold text-primary-500">
                      {typeof v === "number" ? v.toFixed(4) : JSON.stringify(v)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (<div className="text-center text-muted-foreground py-16">Run decoding analysis to see results</div>)}
          </div>
        </div>
      )}

      {/* Techniques Overview */}
      {techniques.data && (
        <div className="glass rounded-2xl p-6 space-y-4">
          <h2 className="text-lg font-semibold">Optimization Techniques</h2>
          <div className="grid grid-cols-3 gap-4">
            {((techniques.data as any).techniques || []).map((t: any) => (
              <div key={t.name} className="bg-background rounded-xl p-4 space-y-2">
                <h3 className="font-semibold text-primary-500">{t.name}</h3>
                <p className="text-xs text-muted-foreground">{t.description}</p>
                <div className="text-xs"><span className="text-green-500">Speedup:</span> {t.speedup}</div>
                <div className="text-xs"><span className="text-yellow-500">Cost:</span> {t.memory_cost}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
