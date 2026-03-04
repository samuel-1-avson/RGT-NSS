"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { rlhfApi } from "@/lib/api";

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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">RLHF Lab</h1>
        <p className="text-muted-foreground">
          Explore Reinforcement Learning from Human Feedback — train reward models, run PPO optimization,
          and experiment with Direct Preference Optimization.
        </p>
      </div>

      <div className="flex gap-2 border-b border-border pb-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 border-b-2 border-primary-500"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "reward" && (
        <div className="grid grid-cols-2 gap-6">
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
        </div>
      )}

      {activeTab === "ppo" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">PPO Configuration</h2>
            <p className="text-sm text-muted-foreground">RL-based fine-tuning with clipped surrogate objective and KL penalty.</p>
            {[
              { key: "num_steps", label: "Training Steps", min: 5, max: 100, step: 5 },
              { key: "num_responses", label: "Responses per Prompt", min: 2, max: 16, step: 1 },
              { key: "epsilon", label: "Clip Epsilon", min: 0.05, max: 0.5, step: 0.05 },
              { key: "kl_coef", label: "KL Coefficient", min: 0, max: 1, step: 0.05 },
            ].map((p) => (
              <div key={p.key}>
                <label className="text-sm font-medium mb-1 block">
                  {p.label}: <span className="text-primary-500">{(ppoConfig as any)[p.key]}</span>
                </label>
                <input type="range" min={p.min} max={p.max} step={p.step}
                  value={(ppoConfig as any)[p.key]}
                  onChange={(e) => setPPOConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))}
                  className="w-full" />
              </div>
            ))}
            <button onClick={() => ppoMutation.mutate()} disabled={ppoMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {ppoMutation.isPending ? "Training PPO..." : "Run PPO Training"}
            </button>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">PPO Training Curves</h2>
            {ppoMutation.data ? (
              <div className="grid grid-cols-2 gap-3">
                {["policy_loss", "value_loss", "entropy", "kl_div", "mean_reward"].map((metric) => (
                  <div key={metric} className="bg-background rounded-xl p-3">
                    <div className="text-xs text-muted-foreground capitalize">{metric.replace("_", " ")}</div>
                    <div className="font-mono text-lg font-bold text-primary-500">
                      {((ppoMutation.data as any).steps?.slice(-1)[0])?.[metric]?.toFixed(4)}
                    </div>
                    <div className="mt-2 flex items-end gap-0.5 h-12">
                      {((ppoMutation.data as any).steps || []).map((s: any, i: number) => {
                        const vals = ((ppoMutation.data as any).steps || []).map((x: any) => x[metric]);
                        const max = Math.max(...vals); const min = Math.min(...vals);
                        const h = ((s[metric] - min) / (max - min || 1)) * 100;
                        return <div key={i} className="flex-1 bg-primary-500/60 rounded-t" style={{ height: `${Math.max(h, 2)}%` }} />;
                      })}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-12">Run PPO training to see real curves</div>
            )}
          </div>
          <div className="glass rounded-2xl p-6 col-span-2">
            <h2 className="text-lg font-semibold mb-4">PPO Pipeline</h2>
            <div className="flex items-center justify-center gap-3 text-sm flex-wrap">
              {["Prompt", "→", "SFT Model", "→", "Generate K responses", "→", "Reward Model scores", "→", "PPO Update"].map((step, i) => (
                <div key={i} className={step === "→" ? "text-muted-foreground text-xl" : "bg-background rounded-xl px-4 py-3 font-medium border border-border"}>
                  {step}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === "dpo" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">DPO Configuration</h2>
            <p className="text-sm text-muted-foreground">
              Direct Preference Optimization learns from preferences without a separate reward model.
            </p>
            {[
              { key: "num_steps", label: "Training Steps", min: 5, max: 100, step: 5 },
              { key: "beta", label: "Beta (KL weight)", min: 0.01, max: 1, step: 0.01 },
            ].map((p) => (
              <div key={p.key}>
                <label className="text-sm font-medium mb-1 block">
                  {p.label}: <span className="text-primary-500">{(dpoConfig as any)[p.key]}</span>
                </label>
                <input type="range" min={p.min} max={p.max} step={p.step}
                  value={(dpoConfig as any)[p.key]}
                  onChange={(e) => setDPOConfig((prev) => ({ ...prev, [p.key]: parseFloat(e.target.value) }))}
                  className="w-full" />
              </div>
            ))}
            <button onClick={() => dpoMutation.mutate()} disabled={dpoMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {dpoMutation.isPending ? "Training DPO..." : "Run DPO Training"}
            </button>
            <div className="bg-background rounded-xl p-4">
              <h3 className="text-sm font-semibold mb-2">DPO vs PPO</h3>
              <ul className="text-xs space-y-1 text-muted-foreground">
                <li>✓ No separate reward model needed</li>
                <li>✓ Simpler training pipeline</li>
                <li>✓ More stable optimization</li>
                <li>✓ Policy is its own implicit reward</li>
              </ul>
            </div>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">DPO Training Results</h2>
            {dpoMutation.data ? (
              <div className="space-y-3">
                <div className="grid grid-cols-3 gap-2 text-xs font-mono bg-background p-3 rounded-xl">
                  <div className="font-semibold">Step</div><div className="font-semibold">Loss</div><div className="font-semibold">Accuracy</div>
                  {((dpoMutation.data as any).steps || []).map((s: any, i: number) => (
                    <div key={i} className="contents">
                      <div>{s.step}</div><div>{s.loss?.toFixed(4)}</div><div>{(s.accuracy * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
                <div className="bg-background rounded-xl p-3">
                  <div className="text-xs text-muted-foreground mb-2">Loss Curve</div>
                  <div className="flex items-end gap-0.5 h-24">
                    {((dpoMutation.data as any).steps || []).map((s: any, i: number) => {
                      const vals = ((dpoMutation.data as any).steps || []).map((x: any) => x.loss);
                      const max = Math.max(...vals); const min = Math.min(...vals);
                      const h = ((s.loss - min) / (max - min || 1)) * 100;
                      return <div key={i} className="flex-1 bg-accent-500/60 rounded-t" style={{ height: `${Math.max(100 - h, 2)}%` }} />;
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-16">Run DPO training to see results</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
