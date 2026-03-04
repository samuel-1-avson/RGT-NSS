"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { safetyApi } from "@/lib/api";

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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">AI Safety Center</h1>
        <p className="text-muted-foreground">
          Evaluate model safety, run red-team attacks, and apply Constitutional AI principles.
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

      {activeTab === "evaluate" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Safety Evaluator</h2>
            <p className="text-sm text-muted-foreground">Check text for safety concerns across multiple risk categories.</p>
            <textarea value={evalText} onChange={(e) => setEvalText(e.target.value)} rows={5}
              className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm font-mono resize-none" />
            <button onClick={() => evalMutation.mutate()} disabled={evalMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {evalMutation.isPending ? "Evaluating..." : "Evaluate Safety"}
            </button>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Safety Results</h2>
            {evalMutation.data ? (
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className={`text-3xl ${(evalMutation.data as any).safe ? "text-green-500" : "text-red-500"}`}>
                    {(evalMutation.data as any).safe ? "✓ Safe" : "✗ Unsafe"}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Overall score: <span className="font-mono font-bold">{(evalMutation.data as any).overall_score?.toFixed(2)}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  {((evalMutation.data as any).categories || []).map((c: any) => (
                    <div key={c.category} className="bg-background rounded-xl p-3 flex items-center gap-3">
                      <span className={`text-sm ${c.flagged ? "text-red-500" : "text-green-500"}`}>{c.flagged ? "⚠️" : "✓"}</span>
                      <span className="text-sm flex-1">{c.category}</span>
                      <div className="w-20 bg-muted rounded-full h-2 overflow-hidden">
                        <div className={`h-full rounded-full ${c.score > 0.5 ? "bg-red-500" : c.score > 0.2 ? "bg-yellow-500" : "bg-green-500"}`}
                          style={{ width: `${Math.min(c.score * 100, 100)}%` }} />
                      </div>
                      <span className="font-mono text-xs w-10 text-right">{c.score?.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (<div className="text-center text-muted-foreground py-16">Enter text and evaluate its safety</div>)}
          </div>

          {categories.data && (
            <div className="glass rounded-2xl p-6 space-y-4 col-span-2">
              <h2 className="text-lg font-semibold">Safety Risk Categories</h2>
              <div className="grid grid-cols-4 gap-3">
                {((categories.data as any).categories || []).map((c: any) => (
                  <div key={c.name} className="bg-background rounded-xl p-3">
                    <h3 className="font-semibold text-sm">{c.name}</h3>
                    <p className="text-xs text-muted-foreground mt-1">{c.description}</p>
                    <div className="flex gap-1 flex-wrap mt-2">
                      {(c.keywords || []).slice(0, 3).map((k: string) => (
                        <span key={k} className="px-1.5 py-0.5 rounded bg-red-100 dark:bg-red-900/20 text-red-600 text-[10px]">{k}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "redteam" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Red Team Suite</h2>
            <p className="text-sm text-muted-foreground">
              Run automated adversarial attacks to probe model safety boundaries. This covers common jailbreak and manipulation attempts.
            </p>
            <button onClick={() => redteamMutation.mutate()} disabled={redteamMutation.isPending}
              className="px-6 py-2 rounded-xl bg-red-600 text-white font-medium hover:bg-red-700 disabled:opacity-50 transition-colors">
              {redteamMutation.isPending ? "Running Red Team..." : "Run Red Team Suite"}
            </button>
          </div>
          {redteamMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Red Team Results</h2>
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-background rounded-xl p-3">
                  <div className="text-xs text-muted-foreground">Total Tests</div>
                  <div className="font-mono text-lg font-bold">{(redteamMutation.data as any).total_tests}</div>
                </div>
                <div className="bg-background rounded-xl p-3">
                  <div className="text-xs text-muted-foreground">Scenarios Tested</div>
                  <div className="font-mono text-lg font-bold">{(redteamMutation.data as any).scenarios_tested}</div>
                </div>
              </div>
              <div className="space-y-3">
                {((redteamMutation.data as any).results || []).map((r: any, i: number) => (
                  <div key={i} className="bg-background rounded-xl p-4 space-y-2">
                    <div className="flex justify-between items-center">
                      <h3 className="font-semibold text-sm">{r.scenario}</h3>
                      <span className={`text-xs px-2 py-0.5 rounded ${r.passed ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                        {r.passed ? "Passed" : "Failed"}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">{r.description}</p>
                    <div className="flex justify-between text-xs">
                      <span>Safety Score: <span className="font-mono">{r.safety_score?.toFixed(2)}</span></span>
                      <span>Attack Success: <span className="font-mono">{(r.attack_success_rate * 100)?.toFixed(0)}%</span></span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "constitutional" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Constitutional AI</h2>
            <p className="text-sm text-muted-foreground">
              Apply AI-generated critiques based on constitutional principles, then revise responses.
            </p>
            <div>
              <label className="text-xs font-medium mb-1 block">Model Response</label>
              <textarea value={constResponse} onChange={(e) => setConstResponse(e.target.value)} rows={4}
                className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm resize-none" />
            </div>
            <button onClick={() => constMutation.mutate()} disabled={constMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {constMutation.isPending ? "Applying..." : "Apply Constitutional Review"}
            </button>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Constitutional Review</h2>
            {constMutation.data ? (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {((constMutation.data as any).reviews || []).map((r: any, i: number) => (
                  <div key={i} className="bg-background rounded-xl p-3 space-y-2">
                    <div className="text-xs font-semibold text-primary-500">{r.principle}</div>
                    <div><span className="text-xs font-medium">Critique:</span>
                      <p className="text-xs text-muted-foreground">{r.critique}</p></div>
                    <div><span className="text-xs font-medium">Revision:</span>
                      <p className="text-xs text-green-600 dark:text-green-400">{r.revision}</p></div>
                  </div>
                ))}
              </div>
            ) : (<div className="text-center text-muted-foreground py-16">Submit a response for constitutional review</div>)}
          </div>

          {principles.data && (
            <div className="glass rounded-2xl p-6 space-y-4 col-span-2">
              <h2 className="text-lg font-semibold">Constitutional Principles</h2>
              <div className="grid grid-cols-2 gap-3">
                {((principles.data as any).principles || []).map((p: any, i: number) => (
                  <div key={i} className="bg-background rounded-xl p-3">
                    <h3 className="font-semibold text-sm text-primary-500">{p.name}</h3>
                    <p className="text-xs text-muted-foreground mt-1">{p.principle}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
