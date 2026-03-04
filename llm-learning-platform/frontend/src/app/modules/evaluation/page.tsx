"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { evaluationApi } from "@/lib/api";
import { cn } from "@/lib/utils";

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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Evaluation Suite</h1>
        <p className="text-muted-foreground max-w-3xl">
          Compute BLEU, ROUGE, and perplexity scores. Run benchmark suites and compare models head-to-head.
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
                layoutId="activeTabEval"
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
        {activeTab === "metrics" && (
          <motion.div
            key="metrics"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            {/* BLEU */}
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">BLEU Score</h2>
              <p className="text-xs font-medium text-muted-foreground tracking-wide leading-relaxed">N-GRAM PRECISION W/ BREVITY PENALTY (MACHINE TRANSLATION)</p>
              <div className="space-y-4">
                <div>
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 block">Reference Text</label>
                  <textarea value={bleuRef} onChange={(e) => setBleuRef(e.target.value)} rows={2}
                    className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm font-mono resize-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all placeholder:text-white/20 text-foreground/80" />
                </div>
                <div>
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 block">Hypothesis Output</label>
                  <textarea value={bleuHyp} onChange={(e) => setBleuHyp(e.target.value)} rows={2}
                    className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm font-mono resize-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all placeholder:text-white/20 text-foreground/80" />
                </div>
              </div>
              <button onClick={() => bleuMutation.mutate()} disabled={bleuMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {bleuMutation.isPending ? "Computing..." : "Compute BLEU Score"}
              </button>
              {bleuMutation.data && (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="bg-black/20 border border-white/5 rounded-xl p-4 space-y-3">
                  {Object.entries(bleuMutation.data as Record<string, any>).map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center group">
                      <span className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground group-hover:text-primary transition-colors">{k.replace(/_/g, " ")}</span>
                      <span className="font-mono text-sm text-foreground/90">{typeof v === "number" ? v.toFixed(4) : String(v)}</span>
                    </div>
                  ))}
                </motion.div>
              )}
            </div>

            {/* ROUGE */}
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">ROUGE Score</h2>
              <p className="text-xs font-medium text-muted-foreground tracking-wide leading-relaxed">RECALL-ORIENTED OVERLAP (SUMMARIZATION QUALITY)</p>
              <div className="space-y-4">
                <div>
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 block">Reference Text</label>
                  <textarea value={rougeRef} onChange={(e) => setRougeRef(e.target.value)} rows={2}
                    className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm font-mono resize-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 outline-none transition-all placeholder:text-white/20 text-foreground/80" />
                </div>
                <div>
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 block">Hypothesis Output</label>
                  <textarea value={rougeHyp} onChange={(e) => setRougeHyp(e.target.value)} rows={2}
                    className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm font-mono resize-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 outline-none transition-all placeholder:text-white/20 text-foreground/80" />
                </div>
              </div>
              <button onClick={() => rougeMutation.mutate()} disabled={rougeMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-accent hover:bg-accent/90 text-accent-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-accent/20 text-sm">
                {rougeMutation.isPending ? "Computing..." : "Compute ROUGE Score"}
              </button>
              {rougeMutation.data && (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="bg-black/20 border border-white/5 rounded-xl p-4 space-y-4">
                  {Object.entries(rougeMutation.data as Record<string, any>).map(([metric, vals]) => (
                    <div key={metric}>
                      <div className="text-[11px] font-bold uppercase tracking-widest text-accent mb-2">{metric}</div>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        {Object.entries(vals as Record<string, number>).map(([k, v]) => (
                          <div key={k} className="bg-white/5 border border-white/5 rounded-lg p-2.5 flex flex-col items-center justify-center">
                            <div className="text-[10px] uppercase text-muted-foreground mb-1">{k}</div>
                            <div className="font-mono text-foreground/90">{v.toFixed(4)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}
            </div>

            {/* Perplexity */}
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Perplexity</h2>
              <p className="text-xs font-medium text-muted-foreground tracking-wide leading-relaxed">EXPONENTIATED AVERAGE CROSS-ENTROPY LOSS (LOWER = BETTER)</p>
              <div>
                <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 block">Loss Values (comma separated)</label>
                <input value={pplLosses} onChange={(e) => setPplLosses(e.target.value)}
                  className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2.5 text-sm font-mono focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 outline-none transition-all placeholder:text-white/20 text-foreground/80" />
              </div>
              <button onClick={() => pplMutation.mutate()} disabled={pplMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-medium disabled:opacity-50 transition-all shadow-lg shadow-emerald-500/20 text-sm">
                {pplMutation.isPending ? "Computing..." : "Compute Perplexity"}
              </button>
              {pplMutation.data && (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="bg-black/20 border border-white/5 rounded-xl p-4 space-y-3">
                  {Object.entries(pplMutation.data as Record<string, any>).map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center group">
                      <span className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground group-hover:text-emerald-400 transition-colors">{k.replace(/_/g, " ")}</span>
                      <span className="font-mono text-sm text-foreground/90">{v}</span>
                    </div>
                  ))}
                </motion.div>
              )}
            </div>

            {/* Metrics Info */}
            {metricsInfo.data && (
              <div className="glass rounded-2xl p-6 space-y-4">
                <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Available Metrics Reference</h2>
                <div className="space-y-3 max-h-[300px] overflow-y-auto custom-scrollbar pr-2">
                  {((metricsInfo.data as any).metrics || []).map((m: any) => (
                    <div key={m.name} className="bg-white/5 border border-white/5 hover:bg-white/10 transition-colors rounded-xl p-3.5">
                      <div className="flex justify-between items-center mb-1.5">
                        <span className="font-semibold text-sm text-foreground/90">{m.name}</span>
                        <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-md bg-white/10 text-muted-foreground">{m.category}</span>
                      </div>
                      <p className="text-xs text-muted-foreground leading-relaxed">{m.description}</p>
                      <p className="text-[11px] font-mono text-muted-foreground/70 mt-2">Range: <span className="text-foreground/80">{m.range}</span></p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {activeTab === "benchmark" && (
          <motion.div
            key="benchmark"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Run Benchmark Suite</h2>
              <div className="flex flex-col md:flex-row md:items-end gap-4">
                <div className="flex-1">
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 block">Model Architecture Name</label>
                  <input value={benchModel} onChange={(e) => setBenchModel(e.target.value)}
                    className="w-full rounded-xl border border-white/10 bg-black/20 px-4 py-2.5 text-sm outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 transition-all font-mono" />
                </div>
                <button onClick={() => benchMutation.mutate()} disabled={benchMutation.isPending}
                  className="px-8 py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm whitespace-nowrap">
                  {benchMutation.isPending ? "Running Benchmark Suite..." : "Run Benchmarks"}
                </button>
              </div>
            </div>
            {benchMutation.data && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-2xl p-6">
                <h2 className="text-lg font-semibold mb-6 flex items-center text-foreground/90 pb-2 border-b border-white/10">
                  Benchmark Results <span className="text-muted-foreground font-normal ml-2">| {(benchMutation.data as any).model}</span>
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {((benchMutation.data as any).benchmarks || []).map((b: any, i: number) => (
                    <motion.div
                      key={b.benchmark}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.1 }}
                      className="bg-black/20 border border-white/5 rounded-xl p-5 space-y-3 relative overflow-hidden group"
                    >
                      <div className="absolute top-0 left-0 w-1 h-full bg-primary/50 group-hover:bg-primary transition-colors" />
                      <div>
                        <h3 className="font-semibold text-[15px]">{b.benchmark}</h3>
                        <p className="text-xs text-muted-foreground mt-1">{b.description}</p>
                      </div>
                      <div className="flex justify-between items-end pt-2">
                        <span className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Accuracy Score</span>
                        <div className="flex items-baseline gap-1">
                          <span className="font-mono text-xl font-bold text-primary">{b.score?.toFixed(2)}</span>
                          <span className="text-xs text-muted-foreground">/100</span>
                        </div>
                      </div>
                      <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
                        <motion.div
                          className="bg-gradient-to-r from-primary to-accent h-full rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(b.score || 0, 100)}%` }}
                          transition={{ duration: 1, delay: 0.2 }}
                        />
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
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
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Compare Benchmark Performance</h2>
              <div className="flex flex-col md:flex-row md:items-end gap-4">
                <div className="flex-1">
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 block">Model Names (Comma Separated)</label>
                  <input value={compareModels} onChange={(e) => setCompareModels(e.target.value)}
                    className="w-full rounded-xl border border-white/10 bg-black/20 px-4 py-2.5 text-sm outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 transition-all font-mono placeholder:text-white/20" />
                </div>
                <button onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}
                  className="px-8 py-2.5 whitespace-nowrap rounded-xl bg-accent hover:bg-accent/90 text-accent-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-accent/20 text-sm">
                  {compareMutation.isPending ? "Evaluating Models..." : "Compare Models"}
                </button>
              </div>
            </div>
            {compareMutation.data && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-2xl overflow-hidden p-[1px] border-accent/20">
                <div className="bg-background/80 rounded-[15px] p-6 h-full">
                  <h2 className="text-lg font-semibold mb-6 flex items-center text-foreground/90 border-b border-white/10 pb-2">
                    Comparative Matrix Head-to-Head
                  </h2>
                  <div className="overflow-x-auto custom-scrollbar pb-2">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10 uppercase tracking-widest text-[10px] text-muted-foreground text-left">
                          <th className="py-3 px-4 font-bold">Model Version</th>
                          {((compareMutation.data as any).benchmarks || []).map((b: string) => (
                            <th key={b} className="text-right py-3 px-4 font-bold truncate max-w-[120px]" title={b}>
                              {b.replace("Eval", "").replace("Benchmark", "").trim() || b}
                            </th>
                          ))}
                          <th className="text-right py-3 px-4 font-bold text-accent">Average</th>
                        </tr>
                      </thead>
                      <tbody>
                        {((compareMutation.data as any).results || []).map((r: any, i: number) => (
                          <motion.tr
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.1 }}
                            key={i}
                            className="border-b border-white/5 hover:bg-white/5 transition-colors group"
                          >
                            <td className="py-3 px-4 font-semibold text-[13px]">{r.model}</td>
                            {(r.scores || []).map((s: number, j: number) => (
                              <td key={j} className="py-3 px-4 text-right font-mono text-muted-foreground group-hover:text-foreground/80 transition-colors">
                                {s?.toFixed(2)}
                              </td>
                            ))}
                            <td className="py-3 px-4 text-right font-mono font-bold text-accent">
                              {r.average?.toFixed(2)}
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
