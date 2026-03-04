"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { safetyApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function SafetyPage() {
  const [activeTab, setActiveTab] = useState<"evaluate" | "redteam" | "constitutional">("evaluate");
  const [evalText, setEvalText] = useState("This is a test response for safety evaluation.");
  const [constResponse, setConstResponse] = useState("Here is a model response that could be improved for helpfulness and safety.");

  const categories = useQuery<any>({ queryKey: ["safety-categories"], queryFn: () => safetyApi.categories() });
  const principles = useQuery<any>({ queryKey: ["safety-principles"], queryFn: () => safetyApi.principles() });

  const evalMutation = useMutation<any>({ mutationFn: () => safetyApi.evaluate(evalText) });
  const redteamMutation = useMutation<any>({ mutationFn: () => safetyApi.redteam() });
  const constMutation = useMutation<any>({ mutationFn: () => safetyApi.constitutional({ response: constResponse }) });

  const tabs = [
    { id: "evaluate" as const, label: "Safety Evaluator", icon: "🛡️" },
    { id: "redteam" as const, label: "Red Team Suite", icon: "🔴" },
    { id: "constitutional" as const, label: "Constitutional AI", icon: "📜" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">AI Safety Center</h1>
        <p className="text-muted-foreground max-w-3xl">
          Evaluate model safety, run red-team adversarial attacks, and apply Constitutional AI principles for responsible model development.
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
                layoutId="activeTabSafety"
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
        {activeTab === "evaluate" && (
          <motion.div
            key="evaluate"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="glass rounded-2xl p-6 space-y-6">
                <div className="pb-4 border-b border-white/10">
                  <h2 className="text-lg font-semibold text-foreground/90">Safety Evaluator</h2>
                  <p className="text-sm text-muted-foreground mt-1">Check text for safety concerns across multiple risk categories.</p>
                </div>
                <textarea value={evalText} onChange={(e) => setEvalText(e.target.value)} rows={5}
                  className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm font-mono resize-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all text-foreground/90" />
                <button onClick={() => evalMutation.mutate()} disabled={evalMutation.isPending}
                  className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                  {evalMutation.isPending ? "Evaluating..." : "Evaluate Safety"}
                </button>
              </div>
              <div className="glass rounded-2xl p-6 flex flex-col">
                <h2 className="text-lg font-semibold text-foreground/90 mb-4 border-b border-white/10 pb-4">Safety Results</h2>
                {evalMutation.data ? (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5 flex-1">
                    <div className="flex items-center gap-4 mb-2">
                      <div className={`text-2xl font-bold ${(evalMutation.data as any).safe ? "text-emerald-400" : "text-pink-400"}`}>
                        {(evalMutation.data as any).safe ? "✓ Safe" : "✗ Unsafe"}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Score: <span className="font-mono font-bold text-foreground/90">{(evalMutation.data as any).overall_score?.toFixed(2)}</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      {((evalMutation.data as any).categories || []).map((c: any, i: number) => (
                        <motion.div
                          initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                          key={c.category}
                          className="bg-black/40 border border-white/5 rounded-lg p-3 flex items-center gap-3 group hover:bg-white/5 transition-colors"
                        >
                          <span className={`text-sm ${c.flagged ? "text-pink-400" : "text-emerald-400"}`}>{c.flagged ? "⚠️" : "✓"}</span>
                          <span className="text-sm flex-1 text-foreground/80">{c.category}</span>
                          <div className="w-20 bg-black/60 rounded-full h-1.5 overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }} animate={{ width: `${Math.min(c.score * 100, 100)}%` }}
                              transition={{ duration: 0.8, delay: i * 0.05 }}
                              className={`h-full rounded-full ${c.score > 0.5 ? "bg-pink-500" : c.score > 0.2 ? "bg-amber-500" : "bg-emerald-500"}`}
                            />
                          </div>
                          <span className="font-mono text-xs w-10 text-right text-muted-foreground">{c.score?.toFixed(2)}</span>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <div className="flex-1 min-h-[250px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                    Enter text and evaluate its safety
                  </div>
                )}
              </div>
            </div>

            {categories.data && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
                className="glass rounded-2xl p-6 space-y-4">
                <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Safety Risk Categories</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                  {((categories.data as any).categories || []).map((c: any) => (
                    <div key={c.name} className="bg-black/20 border border-white/5 rounded-xl p-4 hover:bg-white/5 transition-colors cursor-default group">
                      <h3 className="font-semibold text-xs uppercase tracking-widest text-foreground/80 group-hover:text-primary transition-colors">{c.name}</h3>
                      <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{c.description}</p>
                      <div className="flex gap-1 flex-wrap mt-2 pt-2 border-t border-white/5">
                        {(c.keywords || []).slice(0, 3).map((k: string) => (
                          <span key={k} className="px-1.5 py-0.5 rounded bg-pink-500/10 border border-pink-500/20 text-pink-400 text-[10px] font-semibold">{k}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {activeTab === "redteam" && (
          <motion.div
            key="redteam"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="glass rounded-2xl p-6 flex flex-col items-center max-w-2xl mx-auto text-center">
              <div className="w-12 h-12 rounded-2xl bg-pink-500/20 flex items-center justify-center text-pink-400 mb-4 shadow-lg shadow-pink-500/20">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="m4.93 4.93 14.14 14.14" /></svg>
              </div>
              <h2 className="text-xl font-bold text-foreground/90 mb-2">Adversarial Red Team Suite</h2>
              <p className="text-sm text-muted-foreground mb-6 max-w-md">
                Run automated adversarial attacks to probe model safety boundaries. Covers common jailbreak, prompt injection, and manipulation attempts.
              </p>
              <button onClick={() => redteamMutation.mutate()} disabled={redteamMutation.isPending}
                className="w-full md:w-auto px-8 py-3 rounded-xl bg-pink-600 hover:bg-pink-700 text-white font-medium disabled:opacity-50 transition-all shadow-lg shadow-pink-600/20 text-sm">
                {redteamMutation.isPending ? "Running Red Team..." : "Launch Red Team Suite"}
              </button>
            </div>
            {redteamMutation.data && (
              <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="glass rounded-2xl p-6">
                <div className="flex justify-between items-center mb-6 pb-4 border-b border-white/10">
                  <h2 className="text-lg font-semibold text-foreground/90">Red Team Results</h2>
                  <div className="flex gap-4 text-sm">
                    <div className="text-center">
                      <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Tests</div>
                      <div className="font-mono text-lg text-foreground/90">{(redteamMutation.data as any).total_tests}</div>
                    </div>
                    <div className="text-center">
                      <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Scenarios</div>
                      <div className="font-mono text-lg text-foreground/90">{(redteamMutation.data as any).scenarios_tested}</div>
                    </div>
                  </div>
                </div>
                <div className="space-y-3">
                  {((redteamMutation.data as any).results || []).map((r: any, i: number) => (
                    <motion.div
                      initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
                      key={i}
                      className="bg-black/40 border border-white/5 rounded-xl p-4 space-y-2 hover:bg-white/5 transition-colors"
                    >
                      <div className="flex justify-between items-center">
                        <h3 className="font-semibold text-sm text-foreground/80">{r.scenario}</h3>
                        <span className={cn(
                          "text-[10px] px-2.5 py-1 rounded-full font-bold uppercase tracking-wider",
                          r.passed ? "bg-emerald-500/10 text-emerald-400" : "bg-pink-500/10 text-pink-400"
                        )}>
                          {r.passed ? "Passed" : "Failed"}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground">{r.description}</p>
                      <div className="flex justify-between text-xs font-mono pt-1 border-t border-white/5">
                        <span className="text-muted-foreground">Safety: <span className="text-foreground/80">{r.safety_score?.toFixed(2)}</span></span>
                        <span className="text-muted-foreground">Attack Success: <span className={r.attack_success_rate > 0.5 ? "text-pink-400" : "text-emerald-400"}>{(r.attack_success_rate * 100)?.toFixed(0)}%</span></span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {activeTab === "constitutional" && (
          <motion.div
            key="constitutional"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="glass rounded-2xl p-6 space-y-6">
                <div className="pb-4 border-b border-white/10">
                  <h2 className="text-lg font-semibold text-foreground/90">Constitutional AI</h2>
                  <p className="text-sm text-muted-foreground mt-1">
                    Apply AI-generated critiques based on constitutional principles, then revise responses for better alignment.
                  </p>
                </div>
                <div>
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block mb-2">Model Response</label>
                  <textarea value={constResponse} onChange={(e) => setConstResponse(e.target.value)} rows={4}
                    className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm resize-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all text-foreground/90" />
                </div>
                <button onClick={() => constMutation.mutate()} disabled={constMutation.isPending}
                  className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                  {constMutation.isPending ? "Applying..." : "Apply Constitutional Review"}
                </button>
              </div>
              <div className="glass rounded-2xl p-6 flex flex-col">
                <h2 className="text-lg font-semibold text-foreground/90 mb-4 border-b border-white/10 pb-4">Constitutional Review</h2>
                {constMutation.data ? (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3 max-h-96 overflow-y-auto custom-scrollbar flex-1">
                    {((constMutation.data as any).reviews || []).map((r: any, i: number) => (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
                        key={i}
                        className="bg-black/40 border border-white/5 rounded-xl p-4 space-y-3"
                      >
                        <div className="text-[10px] font-bold text-primary uppercase tracking-widest">{r.principle}</div>
                        <div>
                          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">Critique</span>
                          <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{r.critique}</p>
                        </div>
                        <div className="pt-2 border-t border-white/5">
                          <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-wider">Revision</span>
                          <p className="text-xs text-emerald-400/80 mt-1 leading-relaxed">{r.revision}</p>
                        </div>
                      </motion.div>
                    ))}
                  </motion.div>
                ) : (
                  <div className="flex-1 min-h-[250px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                    Submit a response for constitutional review
                  </div>
                )}
              </div>
            </div>

            {principles.data && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
                className="glass rounded-2xl p-6 space-y-4">
                <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Constitutional Principles</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {((principles.data as any).principles || []).map((p: any, i: number) => (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.05 }}
                      key={i}
                      className="bg-black/20 border border-white/5 rounded-xl p-4 hover:bg-white/5 transition-colors cursor-default group"
                    >
                      <h3 className="font-semibold text-xs uppercase tracking-widest text-primary group-hover:text-primary/80 transition-colors">{p.name}</h3>
                      <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{p.principle}</p>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
