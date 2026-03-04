"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { evaluationApi } from "@/lib/api";

export default function EvaluationPage() {
  const [activeTab, setActiveTab] = useState<"metrics" | "benchmark" | "compare">("metrics");
  const [bleuRef, setBleuRef] = useState("The cat sat on the mat");
  const [bleuHyp, setBleuHyp] = useState("The cat is on the mat");
  const [rougeRef, setRougeRef] = useState("The quick brown fox jumps over the lazy dog");
  const [rougeHyp, setRougeHyp] = useState("The fast brown fox leaps over the lazy dog");
  const [pplLosses, setPplLosses] = useState("2.5, 2.3, 2.1, 1.9, 1.8");
  const [benchModel, setBenchModel] = useState("MicroGPT");
  const [compareModels, setCompareModels] = useState("MicroGPT-nano, MicroGPT-micro, MicroGPT-small");

  const metricsInfo = useQuery<any>({ queryKey: ["eval-metrics"], queryFn: () => evaluationApi.metrics() });

  const bleuMutation = useMutation<any>({ mutationFn: () => evaluationApi.bleu({ reference: bleuRef, hypothesis: bleuHyp }) });
  const rougeMutation = useMutation<any>({ mutationFn: () => evaluationApi.rouge({ reference: rougeRef, hypothesis: rougeHyp }) });
  const pplMutation = useMutation<any>({
    mutationFn: () => evaluationApi.perplexity(pplLosses.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n))),
  });
  const benchMutation = useMutation<any>({ mutationFn: () => evaluationApi.benchmark(benchModel) });
  const compareMutation = useMutation<any>({
    mutationFn: () => evaluationApi.compare(compareModels.split(",").map((s) => s.trim()).filter(Boolean)),
  });

  const tabs = [
    { id: "metrics" as const, label: "Metric Calculator", icon: "📐" },
    { id: "benchmark" as const, label: "Benchmark Suite", icon: "🏋️" },
    { id: "compare" as const, label: "Model Comparison", icon: "⚔️" },
  ];

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Evaluation Suite</h1>
        <p className="text-muted-foreground">
          Compute BLEU, ROUGE, and perplexity scores. Run benchmark suites and compare models head-to-head.
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

      {activeTab === "metrics" && (
        <div className="grid grid-cols-2 gap-6">
          {/* BLEU */}
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">BLEU Score</h2>
            <p className="text-sm text-muted-foreground">N-gram precision with brevity penalty for machine translation quality.</p>
            <div><label className="text-xs font-medium mb-1 block">Reference</label>
              <textarea value={bleuRef} onChange={(e) => setBleuRef(e.target.value)} rows={2}
                className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm font-mono resize-none" /></div>
            <div><label className="text-xs font-medium mb-1 block">Hypothesis</label>
              <textarea value={bleuHyp} onChange={(e) => setBleuHyp(e.target.value)} rows={2}
                className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm font-mono resize-none" /></div>
            <button onClick={() => bleuMutation.mutate()} disabled={bleuMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {bleuMutation.isPending ? "Computing..." : "Compute BLEU"}
            </button>
            {bleuMutation.data && (
              <div className="bg-background rounded-xl p-4 space-y-2">
                {Object.entries(bleuMutation.data as Record<string, any>).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-sm text-muted-foreground">{k.replace(/_/g, " ")}</span>
                    <span className="font-mono font-bold text-primary-500">{typeof v === "number" ? v.toFixed(4) : String(v)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ROUGE */}
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">ROUGE Score</h2>
            <p className="text-sm text-muted-foreground">Recall-oriented overlap for summarization quality.</p>
            <div><label className="text-xs font-medium mb-1 block">Reference</label>
              <textarea value={rougeRef} onChange={(e) => setRougeRef(e.target.value)} rows={2}
                className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm font-mono resize-none" /></div>
            <div><label className="text-xs font-medium mb-1 block">Hypothesis</label>
              <textarea value={rougeHyp} onChange={(e) => setRougeHyp(e.target.value)} rows={2}
                className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm font-mono resize-none" /></div>
            <button onClick={() => rougeMutation.mutate()} disabled={rougeMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {rougeMutation.isPending ? "Computing..." : "Compute ROUGE"}
            </button>
            {rougeMutation.data && (
              <div className="bg-background rounded-xl p-4 space-y-3">
                {Object.entries(rougeMutation.data as Record<string, any>).map(([metric, vals]) => (
                  <div key={metric}>
                    <div className="text-xs font-semibold mb-1">{metric}</div>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      {Object.entries(vals as Record<string, number>).map(([k, v]) => (
                        <div key={k} className="bg-muted rounded-lg p-2 text-center">
                          <div className="text-muted-foreground">{k}</div>
                          <div className="font-mono font-bold text-primary-500">{v.toFixed(4)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Perplexity */}
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Perplexity</h2>
            <p className="text-sm text-muted-foreground">Exponentiated average cross-entropy loss. Lower = better.</p>
            <div><label className="text-xs font-medium mb-1 block">Loss Values (comma separated)</label>
              <input value={pplLosses} onChange={(e) => setPplLosses(e.target.value)}
                className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm font-mono" /></div>
            <button onClick={() => pplMutation.mutate()} disabled={pplMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {pplMutation.isPending ? "Computing..." : "Compute Perplexity"}
            </button>
            {pplMutation.data && (
              <div className="bg-background rounded-xl p-4 space-y-2">
                {Object.entries(pplMutation.data as Record<string, any>).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-sm text-muted-foreground">{k.replace(/_/g, " ")}</span>
                    <span className="font-mono font-bold text-primary-500">{v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Metrics Info */}
          {metricsInfo.data && (
            <div className="glass rounded-2xl p-6 space-y-3">
              <h2 className="text-lg font-semibold">Available Metrics</h2>
              {((metricsInfo.data as any).metrics || []).map((m: any) => (
                <div key={m.name} className="bg-background rounded-lg p-3">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-sm">{m.name}</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-primary-100 dark:bg-primary-900/30 text-primary-600">{m.category}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{m.description}</p>
                  <p className="text-xs text-muted-foreground">Range: {m.range}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === "benchmark" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Run Benchmark Suite</h2>
            <div className="flex gap-4 items-end">
              <div className="flex-1">
                <label className="text-sm font-medium mb-1 block">Model Name</label>
                <input value={benchModel} onChange={(e) => setBenchModel(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm" />
              </div>
              <button onClick={() => benchMutation.mutate()} disabled={benchMutation.isPending}
                className="px-6 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {benchMutation.isPending ? "Running..." : "Run Benchmarks"}
              </button>
            </div>
          </div>
          {benchMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Benchmark Results: {(benchMutation.data as any).model}</h2>
              <div className="grid grid-cols-2 gap-4">
                {((benchMutation.data as any).benchmarks || []).map((b: any) => (
                  <div key={b.benchmark} className="bg-background rounded-xl p-4 space-y-2">
                    <h3 className="font-semibold">{b.benchmark}</h3>
                    <p className="text-xs text-muted-foreground">{b.description}</p>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Score</span>
                      <span className="font-mono text-lg font-bold text-primary-500">{b.score?.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                      <div className="bg-primary-500 h-full rounded-full" style={{ width: `${Math.min(b.score || 0, 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "compare" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Compare Models</h2>
            <div className="flex gap-4 items-end">
              <div className="flex-1">
                <label className="text-sm font-medium mb-1 block">Model Names (comma separated)</label>
                <input value={compareModels} onChange={(e) => setCompareModels(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm" />
              </div>
              <button onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}
                className="px-6 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {compareMutation.isPending ? "Comparing..." : "Compare Models"}
              </button>
            </div>
          </div>
          {compareMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Comparison Results</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-border">
                    <th className="text-left py-2 px-3">Model</th>
                    {((compareMutation.data as any).benchmarks || []).map((b: string) => (
                      <th key={b} className="text-right py-2 px-3">{b}</th>
                    ))}
                    <th className="text-right py-2 px-3">Average</th>
                  </tr></thead>
                  <tbody>
                    {((compareMutation.data as any).results || []).map((r: any, i: number) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-muted/50">
                        <td className="py-2 px-3 font-medium">{r.model}</td>
                        {(r.scores || []).map((s: number, j: number) => (
                          <td key={j} className="py-2 px-3 text-right font-mono">{s?.toFixed(2)}</td>
                        ))}
                        <td className="py-2 px-3 text-right font-mono font-bold text-primary-500">{r.average?.toFixed(2)}</td>
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
