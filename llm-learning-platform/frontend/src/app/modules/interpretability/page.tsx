"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { interpretabilityApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function InterpretabilityPage() {
  const [activeTab, setActiveTab] = useState<"logit-lens" | "patching" | "neurons" | "circuits">("logit-lens");
  const [logitText, setLogitText] = useState("The capital of France is");
  const [logitLayers, setLogitLayers] = useState(6);
  const [cleanText, setCleanText] = useState("The Eiffel Tower is in Paris");
  const [corrText, setCorrText] = useState("The Eiffel Tower is in London");
  const [neuronText, setNeuronText] = useState("Hello world");
  const [neuronLayer, setNeuronLayer] = useState(0);
  const [circuitText, setCircuitText] = useState("The cat sat on the mat");
  const [circuitLayers, setCircuitLayers] = useState(6);
  const [circuitHeads, setCircuitHeads] = useState(8);

  const toolsInfo = useQuery<any>({ queryKey: ["interp-tools"], queryFn: () => interpretabilityApi.tools() });

  const logitMutation = useMutation<any>({ mutationFn: () => interpretabilityApi.logitLens({ text: logitText, num_layers: logitLayers, top_k: 5 }) });
  const patchMutation = useMutation<any>({ mutationFn: () => interpretabilityApi.activationPatching({ clean_text: cleanText, corrupted_text: corrText }) });
  const neuronMutation = useMutation<any>({ mutationFn: () => interpretabilityApi.neurons({ text: neuronText, layer: neuronLayer, top_k: 10 }) });
  const circuitMutation = useMutation<any>({ mutationFn: () => interpretabilityApi.circuits({ text: circuitText, num_layers: circuitLayers, num_heads: circuitHeads }) });

  const tabs = [
    { id: "logit-lens" as const, label: "Logit Lens", icon: "🔍" },
    { id: "patching" as const, label: "Activation Patching", icon: "🔬" },
    { id: "neurons" as const, label: "Neuron Analysis", icon: "🧠" },
    { id: "circuits" as const, label: "Circuit Tracing", icon: "⚡" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Mechanistic Interpretability</h1>
        <p className="text-muted-foreground max-w-3xl">
          Peer inside transformers with logit lens, activation patching, neuron analysis, and circuit tracing.
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
                layoutId="activeTabInterp"
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
        {activeTab === "logit-lens" && (
          <motion.div
            key="logit-lens"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Logit Lens</h2>
              <p className="text-sm text-muted-foreground">Project intermediate layer representations through the unembedding matrix to see how predictions evolve layer by layer.</p>
              <div><label className="text-xs font-medium mb-1 block">Input Text</label>
                <input value={logitText} onChange={(e) => setLogitText(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm" /></div>
              <div>
                <label className="text-sm font-medium mb-1 block">Layers: <span className="text-primary-500">{logitLayers}</span></label>
                <input type="range" min={2} max={24} step={1} value={logitLayers}
                  onChange={(e) => setLogitLayers(parseInt(e.target.value))} className="w-full" />
              </div>
              <button onClick={() => logitMutation.mutate()} disabled={logitMutation.isPending}
                className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {logitMutation.isPending ? "Probing..." : "Run Logit Lens"}
              </button>
            </div>
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Layer-by-Layer Predictions</h2>
              {logitMutation.data ? (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {((logitMutation.data as any).layers || []).map((layer: any) => (
                    <div key={layer.layer} className="bg-background rounded-xl p-3">
                      <div className="text-xs font-semibold text-muted-foreground mb-1">Layer {layer.layer}</div>
                      <div className="flex gap-2 flex-wrap">
                        {(layer.top_tokens || []).map((t: any, i: number) => (
                          <div key={i} className="px-2 py-1 rounded bg-primary-100 dark:bg-primary-900/30 text-xs font-mono">
                            {t.token} <span className="text-muted-foreground">({t.prob?.toFixed(3)})</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (<div className="text-center text-muted-foreground py-16">Run logit lens to see layer predictions</div>)}
            </div>
          </motion.div>
        )}

        {activeTab === "patching" && (
          <div className="grid grid-cols-2 gap-6">
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Activation Patching</h2>
              <p className="text-sm text-muted-foreground">Replace activations from a clean run with a corrupted run to measure causal importance.</p>
              <div><label className="text-xs font-medium mb-1 block text-green-500">Clean Text</label>
                <input value={cleanText} onChange={(e) => setCleanText(e.target.value)}
                  className="w-full rounded-xl border border-green-300 dark:border-green-800 bg-background px-3 py-2 text-sm" /></div>
              <div><label className="text-xs font-medium mb-1 block text-red-500">Corrupted Text</label>
                <input value={corrText} onChange={(e) => setCorrText(e.target.value)}
                  className="w-full rounded-xl border border-red-300 dark:border-red-800 bg-background px-3 py-2 text-sm" /></div>
              <button onClick={() => patchMutation.mutate()} disabled={patchMutation.isPending}
                className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {patchMutation.isPending ? "Patching..." : "Run Patching"}
              </button>
            </div>
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Patching Results</h2>
              {patchMutation.data ? (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {((patchMutation.data as any).components || []).map((c: any, i: number) => (
                    <div key={i} className="bg-background rounded-xl p-3 flex justify-between items-center">
                      <span className="text-sm font-mono">{c.component}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-muted rounded-full h-2 overflow-hidden">
                          <div className={`h-full rounded-full ${c.importance > 0.5 ? "bg-red-500" : c.importance > 0.2 ? "bg-yellow-500" : "bg-green-500"}`}
                            style={{ width: `${Math.min(c.importance * 100, 100)}%` }} />
                        </div>
                        <span className="font-mono text-xs w-12 text-right">{c.importance?.toFixed(3)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (<div className="text-center text-muted-foreground py-16">Run patching to see component importance</div>)}
            </div>
          </div>
        )}

        {activeTab === "neurons" && (
          <div className="grid grid-cols-2 gap-6">
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Neuron Analysis</h2>
              <p className="text-sm text-muted-foreground">Examine individual neuron activation patterns in MLP layers.</p>
              <div><label className="text-xs font-medium mb-1 block">Input Text</label>
                <input value={neuronText} onChange={(e) => setNeuronText(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm" /></div>
              <div>
                <label className="text-sm font-medium mb-1 block">Layer: <span className="text-primary-500">{neuronLayer}</span></label>
                <input type="range" min={0} max={11} step={1} value={neuronLayer}
                  onChange={(e) => setNeuronLayer(parseInt(e.target.value))} className="w-full" />
              </div>
              <button onClick={() => neuronMutation.mutate()} disabled={neuronMutation.isPending}
                className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {neuronMutation.isPending ? "Analyzing..." : "Analyze Neurons"}
              </button>
            </div>
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Neuron Activations</h2>
              {neuronMutation.data ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-background rounded-xl p-3">
                      <div className="text-xs text-muted-foreground">Dead Neurons</div>
                      <div className="font-mono font-bold text-red-500">{(neuronMutation.data as any).dead_neurons}</div>
                    </div>
                    <div className="bg-background rounded-xl p-3">
                      <div className="text-xs text-muted-foreground">Dead %</div>
                      <div className="font-mono font-bold">{(neuronMutation.data as any).dead_pct?.toFixed(1)}%</div>
                    </div>
                  </div>
                  <div className="text-xs font-semibold">Top Activated Neurons</div>
                  <div className="space-y-1">
                    {((neuronMutation.data as any).top_neurons || []).map((n: any, i: number) => (
                      <div key={i} className="flex items-center gap-2 bg-background rounded-lg p-2">
                        <span className="font-mono text-xs w-16">#{n.index}</span>
                        <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
                          <div className="bg-primary-500 h-full rounded-full" style={{ width: `${Math.min(n.activation * 20, 100)}%` }} />
                        </div>
                        <span className="font-mono text-xs w-16 text-right">{n.activation?.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (<div className="text-center text-muted-foreground py-16">Run analysis to see neuron activations</div>)}
            </div>
          </div>
        )}

        {activeTab === "circuits" && (
          <div className="grid grid-cols-2 gap-6">
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Circuit Tracing</h2>
              <p className="text-sm text-muted-foreground">Trace information flow through attention heads and MLP layers.</p>
              <div><label className="text-xs font-medium mb-1 block">Input Text</label>
                <input value={circuitText} onChange={(e) => setCircuitText(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm" /></div>
              {[{ key: "circuitLayers", label: "Layers", min: 2, max: 24, val: circuitLayers, set: setCircuitLayers },
              { key: "circuitHeads", label: "Heads", min: 1, max: 32, val: circuitHeads, set: setCircuitHeads }].map((p) => (
                <div key={p.key}>
                  <label className="text-sm font-medium mb-1 block">{p.label}: <span className="text-primary-500">{p.val}</span></label>
                  <input type="range" min={p.min} max={p.max} step={1} value={p.val}
                    onChange={(e) => p.set(parseInt(e.target.value))} className="w-full" />
                </div>
              ))}
              <button onClick={() => circuitMutation.mutate()} disabled={circuitMutation.isPending}
                className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
                {circuitMutation.isPending ? "Tracing..." : "Trace Circuit"}
              </button>
            </div>
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Circuit Analysis</h2>
              {circuitMutation.data ? (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  <div className="text-xs font-semibold">Head Importance</div>
                  {((circuitMutation.data as any).head_importance || []).slice(0, 10).map((h: any, i: number) => (
                    <div key={i} className="flex items-center gap-2 bg-background rounded-lg p-2">
                      <span className="font-mono text-xs w-24">L{h.layer}H{h.head}</span>
                      <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
                        <div className="bg-primary-500 h-full rounded-full" style={{ width: `${Math.min(h.importance * 100, 100)}%` }} />
                      </div>
                      <span className="font-mono text-xs w-12 text-right">{h.importance?.toFixed(3)}</span>
                    </div>
                  ))}
                  <div className="text-xs font-semibold mt-3">MLP Importance</div>
                  {((circuitMutation.data as any).mlp_importance || []).map((m: any, i: number) => (
                    <div key={i} className="flex items-center gap-2 bg-background rounded-lg p-2">
                      <span className="font-mono text-xs w-24">MLP L{m.layer}</span>
                      <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
                        <div className="bg-accent-500 h-full rounded-full" style={{ width: `${Math.min(m.importance * 100, 100)}%` }} />
                      </div>
                      <span className="font-mono text-xs w-12 text-right">{m.importance?.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              ) : (<div className="text-center text-muted-foreground py-16">Run circuit tracing to see results</div>)}
            </div>
          </div>
        )}

        {toolsInfo.data && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="glass rounded-2xl p-6 space-y-4 mt-8"
          >
            <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Interpretability Toolkit Guide</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {((toolsInfo.data as any).tools || []).map((t: any) => (
                <div key={t.name} className="bg-black/20 border border-white/5 rounded-xl p-4 space-y-2 hover:bg-white/5 transition-colors group cursor-default">
                  <h3 className="font-semibold text-primary group-hover:text-primary-400 transition-colors uppercase tracking-widest text-xs">{t.name}</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">{t.description}</p>
                  <div className="pt-2 mt-auto border-t border-white/5">
                    <p className="text-[10px]"><span className="text-emerald-500 font-bold uppercase">Use case:</span> <span className="text-foreground/80">{t.use_case}</span></p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
