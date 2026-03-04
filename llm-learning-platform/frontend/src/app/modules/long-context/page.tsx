"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { longContextApi } from "@/lib/api";

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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Long Context Techniques</h1>
        <p className="text-muted-foreground">
          Explore positional encoding methods for extending context windows — RoPE, ALiBi, and scaling techniques.
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

      {activeTab === "rope" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">RoPE Configuration</h2>
            <p className="text-sm text-muted-foreground">
              Rotary Position Embeddings encode positions by rotating query and key vectors at different frequencies.
            </p>
            <div>
              <label className="text-xs font-medium mb-1 block">Scaling Method</label>
              <select value={ropeConfig.method} onChange={(e) => setRopeConfig((prev) => ({ ...prev, method: e.target.value }))}
                className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm">
                <option value="none">None (Standard RoPE)</option>
                <option value="linear">Linear Interpolation</option>
                <option value="ntk">NTK-Aware Scaling</option>
                <option value="yarn">YaRN</option>
              </select>
            </div>
            {[
              { key: "dim", label: "Dimension", min: 32, max: 512, step: 32 },
              { key: "max_position", label: "Max Position", min: 128, max: 16384, step: 128 },
              { key: "scaling_factor", label: "Scaling Factor", min: 1, max: 32, step: 1 },
            ].map((p) => (
              <div key={p.key}>
                <label className="text-sm font-medium mb-1 block">
                  {p.label}: <span className="text-primary-500">{(ropeConfig as any)[p.key]}</span>
                </label>
                <input type="range" min={p.min} max={p.max} step={p.step} value={(ropeConfig as any)[p.key]}
                  onChange={(e) => setRopeConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))} className="w-full" />
              </div>
            ))}
            <div className="flex gap-2">
              <button onClick={() => ropeMutation.mutate()} disabled={ropeMutation.isPending}
                className="flex-1 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {ropeMutation.isPending ? "Computing..." : "Compute Frequencies"}
              </button>
              <button onClick={() => ropeCompareMutation.mutate()} disabled={ropeCompareMutation.isPending}
                className="flex-1 py-2 rounded-xl bg-accent-600 text-white font-medium hover:bg-accent-700 disabled:opacity-50 transition-colors">
                {ropeCompareMutation.isPending ? "Comparing..." : "Compare Scaling"}
              </button>
            </div>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">RoPE Analysis</h2>
            {ropeMutation.data ? (
              <div className="space-y-3">
                {Object.entries(ropeMutation.data as Record<string, any>).filter(([k]) => k !== "frequencies").map(([k, v]) => (
                  <div key={k} className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground">{k.replace(/_/g, " ")}</div>
                    <div className="font-mono text-sm text-primary-500">{typeof v === "number" ? v.toFixed(4) : JSON.stringify(v)}</div>
                  </div>
                ))}
                {(ropeMutation.data as any).frequencies && (
                  <div className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground mb-2">Frequency Distribution</div>
                    <div className="flex items-end gap-0.5 h-24">
                      {((ropeMutation.data as any).frequencies || []).slice(0, 40).map((f: number, i: number) => {
                        const maxF = Math.max(...((ropeMutation.data as any).frequencies || []).slice(0, 40));
                        const h = maxF > 0 ? (f / maxF) * 100 : 0;
                        return <div key={i} className="flex-1 bg-primary-500/60 rounded-t" style={{ height: `${Math.max(h, 1)}%` }} />;
                      })}
                    </div>
                  </div>
                )}
              </div>
            ) : (<div className="text-center text-muted-foreground py-16">Compute RoPE frequencies to see analysis</div>)}

            {ropeCompareMutation.data && (
              <div className="space-y-2 mt-4">
                <div className="text-sm font-semibold">Scaling Method Comparison</div>
                {toMethodsArray((ropeCompareMutation.data as any).methods).map((m: any) => (
                  <div key={m.method} className="bg-background rounded-xl p-3 flex justify-between items-center">
                    <span className="text-sm font-medium">{m.method}</span>
                    <span className="text-xs text-muted-foreground">min: {m.min_freq?.toFixed(4)}, max: {m.max_freq?.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "alibi" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">ALiBi Configuration</h2>
            <p className="text-sm text-muted-foreground">
              Attention with Linear Biases adds a linear bias to attention scores based on distance,
              enabling zero-shot context extension without fine-tuning.
            </p>
            {[
              { key: "num_heads", label: "Num Heads", min: 1, max: 32, step: 1 },
              { key: "seq_len", label: "Sequence Length", min: 4, max: 64, step: 4 },
            ].map((p) => (
              <div key={p.key}>
                <label className="text-sm font-medium mb-1 block">
                  {p.label}: <span className="text-primary-500">{(alibiConfig as any)[p.key]}</span>
                </label>
                <input type="range" min={p.min} max={p.max} step={p.step} value={(alibiConfig as any)[p.key]}
                  onChange={(e) => setAlibiConfig((prev) => ({ ...prev, [p.key]: parseInt(e.target.value) }))} className="w-full" />
              </div>
            ))}
            <div className="flex gap-2">
              <button onClick={() => alibiMutation.mutate()} disabled={alibiMutation.isPending}
                className="flex-1 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {alibiMutation.isPending ? "Computing..." : "Compute Bias Matrix"}
              </button>
              <button onClick={() => extMutation.mutate()} disabled={extMutation.isPending}
                className="flex-1 py-2 rounded-xl bg-accent-600 text-white font-medium hover:bg-accent-700 disabled:opacity-50 transition-colors">
                {extMutation.isPending ? "Analyzing..." : "Analyze Extension"}
              </button>
            </div>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">ALiBi Bias Matrix</h2>
            {alibiMutation.data ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  {((alibiMutation.data as any).slopes || []).map((s: number, i: number) => (
                    <div key={i} className="bg-background rounded-lg p-2 text-xs">
                      <span className="text-muted-foreground">Head {i}:</span>
                      <span className="font-mono ml-1">{s?.toFixed(6)}</span>
                    </div>
                  ))}
                </div>
                {(alibiMutation.data as any).bias_matrix_sample && (
                  <div className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground mb-2">Bias Matrix (Head 0, first rows)</div>
                    <div className="font-mono text-[10px] overflow-x-auto">
                      {((alibiMutation.data as any).bias_matrix_sample || []).slice(0, 8).map((row: number[], i: number) => (
                        <div key={i} className="flex gap-1">
                          {row.slice(0, 12).map((v: number, j: number) => (
                            <span key={j} className={`w-10 text-right ${v < -1 ? "text-red-400" : v < 0 ? "text-yellow-400" : "text-green-400"}`}>
                              {v.toFixed(1)}
                            </span>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (<div className="text-center text-muted-foreground py-16">Compute bias matrix to see ALiBi pattern</div>)}

            {extMutation.data && (
              <div className="space-y-2 mt-4">
                <div className="text-sm font-semibold">Context Extension Analysis</div>
                {((extMutation.data as any).extension_analysis || []).map((e: any) => (
                  <div key={e.test_length} className="bg-background rounded-lg p-2 flex justify-between items-center text-xs">
                    <span>Length {e.test_length}</span>
                    <span className="font-mono">{e.extension_ratio?.toFixed(1)}x</span>
                    <span className="font-mono">max bias: {e.max_bias?.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "compare" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Compare All Position Methods</h2>
            <p className="text-sm text-muted-foreground">
              Compare Sinusoidal, Learned, RoPE, RoPE + YaRN, and ALiBi positional encoding methods.
            </p>
            <button onClick={() => compareAllMutation.mutate()} disabled={compareAllMutation.isPending}
              className="px-6 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {compareAllMutation.isPending ? "Comparing..." : "Compare All Methods"}
            </button>
          </div>
          {compareAllMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Method Comparison</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-border">
                    <th className="text-left py-2 px-3">Method</th>
                    <th className="text-right py-2 px-3">Max Context</th>
                    <th className="text-right py-2 px-3">Learnable</th>
                    <th className="text-right py-2 px-3">Extrapolation</th>
                    <th className="text-right py-2 px-3">Relative</th>
                  </tr></thead>
                  <tbody>
                    {toMethodsArray((compareAllMutation.data as any).methods).map((m: any, i: number) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-muted/50">
                        <td className="py-2 px-3 font-medium">{m.method}</td>
                        <td className="py-2 px-3 text-right font-mono">{m.max_context || m.max_extrapolation || "-"}</td>
                        <td className="py-2 px-3 text-right">{m.learnable ? "✓" : "✗"}</td>
                        <td className="py-2 px-3 text-right">{m.extrapolation || m.description || "-"}</td>
                        <td className="py-2 px-3 text-right">{(m.relative ?? m.relative_position) ? "✓" : "✗"}</td>
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
