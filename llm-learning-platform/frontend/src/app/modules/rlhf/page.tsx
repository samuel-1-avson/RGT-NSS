"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { rlhfApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function RLHFPage() {
  const [activeTab, setActiveTab] = useState<"reward" | "ppo" | "dpo">("reward");
  const [prompt, setPrompt] = useState("Write a helpful response about climate change");
  const [chosen, setChosen] = useState("Climate change is a well-documented phenomenon supported by scientific consensus...");
  const [rejected, setRejected] = useState("I don't care about that topic.");
  const [scoreText, setScoreText] = useState("The quick brown fox jumps over the lazy dog");
  const [ppoConfig, setPPOConfig] = useState({ num_steps: 30, num_responses: 4, epsilon: 0.2, kl_coef: 0.1 });
  const [dpoConfig, setDPOConfig] = useState({ num_steps: 30, beta: 0.1 });

  const methods = useQuery<any>({ queryKey: ["rlhf-methods"], queryFn: () => rlhfApi.methods() });

  const scoreMutation = useMutation<any>({ mutationFn: () => rlhfApi.scoreText(scoreText) });
  const compareMutation = useMutation<any>({ mutationFn: () => rlhfApi.compareResponses({ prompt, chosen, rejected }) });
  const trainMutation = useMutation<any>({
    mutationFn: () => rlhfApi.trainReward({ pairs: [{ prompt, chosen, rejected }], learning_rate: 0.001, epochs: 10 }),
  });
  const ppoMutation = useMutation<any>({ mutationFn: () => rlhfApi.trainPPO(ppoConfig) });
  const dpoMutation = useMutation<any>({ mutationFn: () => rlhfApi.trainDPO(dpoConfig) });

  const toMethodsArray = (value: unknown): any[] => {
    if (Array.isArray(value)) {
      return value;
    }
    if (value && typeof value === "object") {
      return Object.entries(value as Record<string, any>).map(([name, meta]) => ({ name, ...(meta || {}) }));
    }
    return [];
  };

  const tabs = [
    { id: "reward" as const, label: "Reward Modeling", icon: "🏆" },
    { id: "ppo" as const, label: "PPO Training", icon: "🎯" },
    { id: "dpo" as const, label: "DPO Training", icon: "⚡" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">RLHF Laboratory</h1>
        <p className="text-muted-foreground max-w-3xl">
          Explore Reinforcement Learning from Human Feedback. Train reward models, run PPO optimization,
          and experiment with Direct Preference Optimization algorithms.
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
                layoutId="activeTabRLHF"
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
        {activeTab === "reward" && (
          <motion.div
            key="reward"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Score Text</h2>
              <p className="text-sm text-muted-foreground">The reward model assigns a scalar score predicting human preference.</p>
              <textarea value={scoreText} onChange={(e) => setScoreText(e.target.value)} rows={3}
                className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm font-mono resize-none" />
              <button onClick={() => scoreMutation.mutate()} disabled={scoreMutation.isPending}
                className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {scoreMutation.isPending ? "Scoring..." : "Compute Reward Score"}
              </button>
              {scoreMutation.data && (
                <div className="bg-background rounded-xl p-4 space-y-2">
                  <div className="flex justify-between"><span className="text-sm text-muted-foreground">Score</span>
                    <span className="font-mono font-bold text-lg text-primary-500">{(scoreMutation.data as any).score}</span></div>
                  <div className="flex justify-between"><span className="text-sm text-muted-foreground">Tokens</span>
                    <span className="font-mono">{(scoreMutation.data as any).token_count}</span></div>
                </div>
              )}
            </div>
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Compare Responses</h2>
              <div><label className="text-xs font-medium mb-1 block">Prompt</label>
                <input value={prompt} onChange={(e) => setPrompt(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm" /></div>
              <div><label className="text-xs font-medium mb-1 block text-green-500">Chosen Response</label>
                <textarea value={chosen} onChange={(e) => setChosen(e.target.value)} rows={2}
                  className="w-full rounded-xl border border-green-300 dark:border-green-800 bg-background px-3 py-2 text-sm resize-none" /></div>
              <div><label className="text-xs font-medium mb-1 block text-red-500">Rejected Response</label>
                <textarea value={rejected} onChange={(e) => setRejected(e.target.value)} rows={2}
                  className="w-full rounded-xl border border-red-300 dark:border-red-800 bg-background px-3 py-2 text-sm resize-none" /></div>
              <button onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}
                className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {compareMutation.isPending ? "Comparing..." : "Compare Responses"}
              </button>
              {compareMutation.data && (
                <div className="bg-background rounded-xl p-4 space-y-2">
                  <div className="flex justify-between"><span className="text-sm text-green-500">Chosen Score</span>
                    <span className="font-mono font-bold">{(compareMutation.data as any).chosen_score}</span></div>
                  <div className="flex justify-between"><span className="text-sm text-red-500">Rejected Score</span>
                    <span className="font-mono font-bold">{(compareMutation.data as any).rejected_score}</span></div>
                  <div className="flex justify-between"><span className="text-sm text-muted-foreground">Correct</span>
                    <span className={`font-bold ${(compareMutation.data as any).preference_correct ? "text-green-500" : "text-red-500"}`}>
                      {(compareMutation.data as any).preference_correct ? "✓ Yes" : "✗ No"}</span></div>
                </div>
              )}
            </div>
            <div className="glass rounded-2xl p-6 space-y-4 col-span-2">
              <h2 className="text-lg font-semibold">Train Reward Model</h2>
              <p className="text-sm text-muted-foreground">Train using Bradley-Terry preference loss on your pairs above.</p>
              <button onClick={() => trainMutation.mutate()} disabled={trainMutation.isPending}
                className="px-6 py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {trainMutation.isPending ? "Training..." : "Train on Preferences"}
              </button>
              {trainMutation.data && (
                <div className="bg-background rounded-xl p-4">
                  <h3 className="text-sm font-semibold mb-3">Training History</h3>
                  <div className="grid grid-cols-5 gap-2 text-xs font-mono">
                    <div className="font-semibold">Epoch</div><div className="font-semibold">Loss</div>
                    <div className="font-semibold">Accuracy</div><div className="font-semibold">Chosen Δ</div>
                    <div className="font-semibold">Rejected Δ</div>
                    {((trainMutation.data as any).training_history || []).map((row: any, i: number) => (
                      <div key={i} className="contents">
                        <div>{row.epoch}</div><div>{row.loss?.toFixed(4)}</div>
                        <div>{(row.accuracy * 100).toFixed(1)}%</div>
                        <div className="text-green-500">+{row.chosen_reward_delta?.toFixed(4)}</div>
                        <div className="text-red-500">{row.rejected_reward_delta?.toFixed(4)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            {methods.data && (
              <div className="glass rounded-2xl p-6 space-y-4 col-span-2">
                <h2 className="text-lg font-semibold">Alignment Methods</h2>
                <div className="grid grid-cols-3 gap-4">
                  {toMethodsArray((methods.data as any).methods).map((m: any) => (
                    <div key={m.name} className="bg-background rounded-xl p-4 space-y-2">
                      <h3 className="font-semibold text-primary-500">{m.name}</h3>
                      <p className="text-xs text-muted-foreground">{m.description}</p>
                      <div className="space-y-1">
                        <div className="text-xs font-medium">Pros:</div>
                        {(Array.isArray(m.pros) ? m.pros : []).map((p: string, i: number) => (<div key={i} className="text-xs text-green-500">+ {p}</div>))}
                        <div className="text-xs font-medium mt-1">Cons:</div>
                        {(Array.isArray(m.cons) ? m.cons : []).map((c: string, i: number) => (<div key={i} className="text-xs text-red-400">- {c}</div>))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {activeTab === "ppo" && (
          <motion.div
            key="ppo"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">PPO Optimization</h2>
                <p className="text-sm text-muted-foreground mt-1">RL-based fine-tuning using a clipped surrogate objective with KL divergence penalties.</p>
              </div>
              <div className="space-y-5">
                {[
                  { key: "num_steps", label: "Training Epochs", min: 5, max: 100, step: 5, color: "accent-primary" },
                  { key: "num_responses", label: "Batch Size (K responses)", min: 2, max: 16, step: 1, color: "accent-accent" },
                  { key: "epsilon", label: "PPO Clip Epsilon", min: 0.05, max: 0.5, step: 0.05, color: "accent-pink-500" },
                  { key: "kl_coef", label: "KL Divergence Penalty (\u03B2)", min: 0, max: 1, step: 0.05, color: "accent-amber-500" },
                ].map((p) => (
                  <div key={p.key}>
                    <div className="flex justify-between items-end mb-2">
                      <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">{p.label}</label>
                      <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">{(ppoConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step}
                      value={(ppoConfig as any)[p.key]}
                      onChange={(e) => setPPOConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))}
                      className={`w-full ${p.color} h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20`} />
                  </div>
                ))}
              </div>
              <button onClick={() => ppoMutation.mutate()} disabled={ppoMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-pink-600 hover:bg-pink-500 text-white font-medium disabled:opacity-50 transition-all shadow-lg shadow-pink-500/20 text-sm mt-4">
                {ppoMutation.isPending ? "Optimizing Policy..." : "Initialize PPO Run"}
              </button>
            </div>

            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">PPO Telemetry Curves</h2>
              </div>
              {ppoMutation.data ? (
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { id: "policy_loss", label: "Policy Loss", color: "from-blue-600 to-blue-400", hex: "text-blue-400", border: 'border-blue-500/20' },
                    { id: "value_loss", label: "Value Loss", color: "from-purple-600 to-purple-400", hex: "text-purple-400", border: 'border-purple-500/20' },
                    { id: "entropy", label: "Entropy", color: "from-emerald-600 to-emerald-400", hex: "text-emerald-400", border: 'border-emerald-500/20' },
                    { id: "kl_div", label: "KL Divergence", color: "from-amber-600 to-amber-400", hex: "text-amber-400", border: 'border-amber-500/20' },
                    { id: "mean_reward", label: "Mean Reward", color: "from-primary to-accent", hex: "text-primary", border: 'border-primary/20', colSpan: "col-span-2" }
                  ].map((metric) => (
                    <motion.div
                      key={metric.id}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className={cn("bg-black/20 border rounded-xl p-4 flex flex-col", metric.colSpan, metric.border)}
                    >
                      <div className="flex justify-between items-center mb-3">
                        <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">{metric.label}</div>
                        <div className={`font-mono text-sm font-bold ${metric.hex}`}>
                          {((ppoMutation.data as any).steps?.slice(-1)[0])?.[metric.id]?.toFixed(4)}
                        </div>
                      </div>
                      <div className="mt-auto flex items-end gap-[1px] h-16 bg-black/40 rounded p-1">
                        {((ppoMutation.data as any).steps || []).map((s: any, i: number) => {
                          const vals = ((ppoMutation.data as any).steps || []).map((x: any) => x[metric.id]);
                          const max = Math.max(...vals); const min = Math.min(...vals);
                          const h = ((s[metric.id] - min) / (max - min || 1)) * 100;
                          return (
                            <motion.div
                              initial={{ height: 0 }}
                              animate={{ height: `${Math.max(h, 2)}%` }}
                              transition={{ duration: 0.5, delay: i * 0.02 }}
                              key={i}
                              className={`flex-1 bg-gradient-to-t ${metric.color} rounded-t-sm opacity-80`}
                            />
                          );
                        })}
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="flex-1 min-h-[300px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                  Execute PPO to project optimization traces
                </div>
              )}
            </div>

            <div className="glass rounded-2xl p-6 col-span-1 md:col-span-2 mt-2">
              <h2 className="text-lg font-semibold mb-6 border-b border-white/10 pb-4 text-foreground/90">Architecture Pipeline (PPO)</h2>
              <div className="flex items-center justify-center gap-2 md:gap-4 text-sm flex-wrap px-4">
                {[
                  { label: "Prompt Dataset", icon: "📁", color: "bg-blue-500/10 text-blue-400 border-blue-500/20" },
                  { label: "SFT Model", icon: "🧠", color: "bg-purple-500/10 text-purple-400 border-purple-500/20" },
                  { label: "Generation", icon: "⚡", color: "bg-amber-500/10 text-amber-400 border-amber-500/20" },
                  { label: "RM Scoring", icon: "🏆", color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" },
                  { label: "PPO Update", icon: "🔄", color: "bg-pink-500/10 text-pink-400 border-pink-500/20" }
                ].map((step, i) => (
                  <div key={i} className="flex items-center">
                    <motion.div
                      whileHover={{ y: -2 }}
                      className={`rounded-xl px-4 py-3 font-medium border flex items-center gap-2 shadow-lg custom-glow text-xs uppercase tracking-widest ${step.color}`}
                    >
                      <span className="text-lg opacity-80">{step.icon}</span> {step.label}
                    </motion.div>
                    {i < 4 && (
                      <div className="text-muted-foreground/30 mx-2 md:mx-4 hidden sm:block">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14" /><path d="m12 5 7 7-7 7" /></svg>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === "dpo" && (
          <motion.div
            key="dpo"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">DPO Configuration</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  Direct Preference Optimization bypasses the reward model, optimizing policy directly on paired preferences.
                </p>
              </div>
              <div className="space-y-5">
                {[
                  { key: "num_steps", label: "Training Epochs", min: 5, max: 100, step: 5, color: "accent-primary" },
                  { key: "beta", label: "Beta (\u03B2 - KL Scaling)", min: 0.01, max: 1, step: 0.01, color: "accent-emerald-500" },
                ].map((p) => (
                  <div key={p.key}>
                    <div className="flex justify-between items-end mb-2">
                      <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">{p.label}</label>
                      <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">{(dpoConfig as any)[p.key]}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step}
                      value={(dpoConfig as any)[p.key]}
                      onChange={(e) => setDPOConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))}
                      className={`w-full ${p.color} h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20`} />
                  </div>
                ))}
              </div>
              <button onClick={() => dpoMutation.mutate()} disabled={dpoMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-accent hover:bg-accent/90 text-accent-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-accent/20 text-sm mt-4">
                {dpoMutation.isPending ? "Solving MLE..." : "Execute DPO Flow"}
              </button>

              <div className="bg-black/20 border border-white/5 rounded-xl p-5 mt-6">
                <h3 className="text-[11px] font-bold uppercase tracking-widest text-primary mb-3">DPO vs PPO Paradigm</h3>
                <ul className="text-xs space-y-2.5 text-muted-foreground">
                  <li className="flex gap-2"><span className="text-emerald-500">✓</span> No complex reward model mapping</li>
                  <li className="flex gap-2"><span className="text-emerald-500">✓</span> Direct cross-entropy objective on data</li>
                  <li className="flex gap-2"><span className="text-emerald-500">✓</span> Extreme training stability (MLE vs RL)</li>
                  <li className="flex gap-2"><span className="text-emerald-500">✓</span> Implicit mapping theorem utilization</li>
                </ul>
              </div>
            </div>

            <div className="glass rounded-2xl p-6 flex flex-col">
              <h2 className="text-lg font-semibold text-foreground/90 pb-4 border-b border-white/10 mb-4">DPO Training Gradients</h2>
              {dpoMutation.data ? (
                <div className="space-y-6 flex-1 flex flex-col">
                  <div className="grid grid-cols-3 gap-3 text-xs font-mono bg-black/40 border border-white/5 p-3 rounded-xl">
                    <div className="font-bold text-muted-foreground/80 uppercase">Step</div>
                    <div className="font-bold text-muted-foreground/80 uppercase text-center">NLL Loss</div>
                    <div className="font-bold text-accent uppercase text-right">Accuracy</div>
                    {((dpoMutation.data as any).steps || []).map((s: any, i: number) => (
                      <div key={i} className="contents">
                        <div className="py-1 text-foreground/80">{s.step}</div>
                        <div className="py-1 text-center text-blue-400">{s.loss?.toFixed(4)}</div>
                        <div className="py-1 text-right text-accent-400">{(s.accuracy * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>

                  <div className="bg-black/20 border border-white/5 rounded-xl p-5 flex-1 flex flex-col">
                    <div className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground mb-4">Loss Descent Trajectory</div>
                    <div className="flex-1 flex items-end gap-[2px] min-h-[120px] bg-black/40 rounded p-2">
                      {((dpoMutation.data as any).steps || []).map((s: any, i: number) => {
                        const vals = ((dpoMutation.data as any).steps || []).map((x: any) => x.loss);
                        const max = Math.max(...vals); const min = Math.min(...vals);
                        const h = ((s.loss - min) / (max - min || 1)) * 100;
                        return (
                          <motion.div
                            initial={{ height: 0 }}
                            animate={{ height: `${Math.max(100 - h, 2)}%` }}
                            transition={{ duration: 0.6, delay: i * 0.05 }}
                            key={i}
                            className="flex-1 bg-gradient-to-t from-accent/80 to-accent-400/80 rounded-t-sm opacity-90"
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex-1 min-h-[300px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                  Run DPO alignment to map cross-entropy distributions
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
