"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { inferenceApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function InferencePage() {
  const [promptIds, setPromptIds] = useState("2");
  const [maxTokens, setMaxTokens] = useState(50);
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(0);
  const [topP, setTopP] = useState(1.0);
  const [result, setResult] = useState<any>(null);
  const [forwardResult, setForwardResult] = useState<any>(null);

  const generate = useMutation({
    mutationFn: () =>
      inferenceApi.generate({
        prompt_ids: promptIds.split(",").map((s) => parseInt(s.trim())),
        max_new_tokens: maxTokens,
        temperature,
        top_k: topK,
        top_p: topP,
      }),
    onSuccess: (data) => setResult(data),
  });

  const forward = useMutation({
    mutationFn: () =>
      inferenceApi.forward({
        token_ids: promptIds.split(",").map((s) => parseInt(s.trim())),
        store_intermediates: true,
      }),
    onSuccess: (data) => setForwardResult(data),
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Inference Engine</h1>
        <p className="text-muted-foreground max-w-3xl">
          Explore autoregressive text generation with different sampling strategies and see how temperature, top-k, and nucleus sampling affect predicted probabilities.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="glass rounded-2xl p-6 space-y-6 flex flex-col"
        >
          <div className="pb-4 border-b border-white/10">
            <h2 className="text-lg font-semibold text-foreground/90">Generation Settings</h2>
            <p className="text-xs text-muted-foreground mt-1">Configure autoregressive decoding.</p>
          </div>

          <div className="space-y-5 flex-1">
            <div>
              <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 block">Prompt Token IDs</label>
              <input
                type="text"
                value={promptIds}
                onChange={(e) => setPromptIds(e.target.value)}
                className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-2.5 font-mono text-sm focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all placeholder:text-muted-foreground/30"
                placeholder="2, 10, 20"
              />
            </div>

            <div className="space-y-5">
              {[
                { key: 'maxTokens', label: 'Max New Tokens', min: 1, max: 200, step: 1, val: maxTokens, set: setMaxTokens, color: 'accent-primary' },
                { key: 'temperature', label: 'Temperature', min: 0.01, max: 3.0, step: 0.01, val: temperature, set: setTemperature, color: 'accent-amber-500' },
                { key: 'topK', label: 'Top-k Filtering', min: 0, max: 100, step: 1, val: topK, set: setTopK, color: 'accent-primary' },
                { key: 'topP', label: 'Top-p (Nucleus)', min: 0.1, max: 1.0, step: 0.05, val: topP, set: setTopP, color: 'accent-accent' }
              ].map((param) => (
                <div key={param.key}>
                  <div className="flex justify-between items-end mb-2">
                    <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">
                      {param.label}
                    </label>
                    <span className="text-sm font-mono text-foreground/90 bg-white/5 border border-white/10 px-2 py-0.5 rounded shadow-sm">
                      {param.key === 'topK' && param.val === 0 ? "Off" : typeof param.val === 'number' && param.val % 1 !== 0 ? param.val.toFixed(2) : param.val}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={param.min}
                    max={param.max}
                    step={param.step}
                    value={param.val}
                    onChange={(e) => param.set(parseFloat(e.target.value))}
                    className={`w-full ${param.color} h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20`}
                  />
                  {param.key === 'temperature' && (
                    <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-muted-foreground/50 mt-1.5 px-1">
                      <span>Deterministic</span>
                      <span>Creative</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="flex gap-3 pt-4 border-t border-white/10 mt-auto">
            <button
              onClick={() => generate.mutate()}
              disabled={generate.isPending}
              className="flex-[2] py-2.5 bg-primary hover:bg-primary/90 text-primary-foreground font-medium rounded-xl transition-all shadow-lg shadow-primary/20 disabled:opacity-50 text-sm"
            >
              {generate.isPending ? "Decoding..." : "Autoregressive Generation"}
            </button>
            <button
              onClick={() => forward.mutate()}
              disabled={forward.isPending}
              className="flex-1 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 text-foreground/90 rounded-xl text-sm font-medium transition-colors disabled:opacity-50"
            >
              Forward Pass
            </button>
          </div>
        </motion.div>

        {/* Output */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="col-span-1 lg:col-span-2 space-y-6 flex flex-col"
        >
          <AnimatePresence>
            {/* Generated Tokens */}
            {result && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="glass rounded-2xl p-6 space-y-6"
              >
                <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Generation Output</h2>

                <div className="bg-black/20 border border-white/5 rounded-xl p-5 font-mono">
                  <div className="flex flex-wrap gap-2 leading-loose">
                    {result.generated_ids?.map((id: number, i: number) => {
                      const isPrompt = i < promptIds.split(",").length;
                      return (
                        <motion.span
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.01 }}
                          key={i}
                          className={cn(
                            "px-2.5 py-1 rounded inline-flex items-center justify-center text-sm shadow-sm",
                            isPrompt
                              ? "bg-amber-500/10 border border-amber-500/20 text-amber-500"
                              : "bg-primary/20 border border-primary/30 text-primary-foreground"
                          )}
                        >
                          {id}
                        </motion.span>
                      );
                    })}
                  </div>
                </div>

                <div className="flex items-center gap-4 text-[11px] uppercase tracking-widest font-bold text-muted-foreground/60">
                  <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-amber-500/20 border border-amber-500/30 inline-block"></span> Prompt</div>
                  <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-primary/20 border border-primary/30 inline-block"></span> Generated</div>
                  <div className="ml-auto flex gap-4">
                    <span>Generated Tokens: <span className="text-foreground/80">{result.generated_ids?.length - promptIds.split(",").length}</span></span>
                    <span>Total Length: <span className="text-foreground/80">{result.generated_ids?.length}</span></span>
                  </div>
                </div>

                {/* Generation Steps */}
                {result.steps?.length > 0 && (
                  <div className="pt-4 border-t border-white/5">
                    <h3 className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground mb-3 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-primary/50"></span> Step-by-Step Probability (First 10)
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                      {result.steps.slice(0, 10).map((step: any, i: number) => (
                        <div key={i} className="bg-black/40 border border-white/5 rounded-lg p-3 flex flex-col items-center justify-center gap-1 relative overflow-hidden group">
                          <div className="absolute inset-x-0 bottom-0 h-1 bg-primary/10">
                            <div className="h-full bg-primary" style={{ width: `${(step.probability || 0) * 100}%` }} />
                          </div>
                          <div className="text-[10px] text-muted-foreground/50 font-mono uppercase">Step {i + 1}</div>
                          <div className="font-mono text-lg font-bold text-primary-foreground">{step.chosen_token}</div>
                          <div className="text-[10px] font-mono text-muted-foreground">P: {(step.probability || 0).toFixed(4)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* Forward Pass */}
            {forwardResult && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="glass rounded-2xl p-6 space-y-6"
              >
                <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Forward Pass Telemetry</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-black/20 border border-white/5 rounded-xl p-5">
                    <div className="text-[11px] font-bold uppercase tracking-widest text-blue-400 mb-3 flex items-center gap-2">Output Logits Shape</div>
                    <div className="font-mono text-lg bg-black/40 p-3 rounded-lg border border-white/5 inline-block text-foreground/90">[{forwardResult.logits_shape?.join(", ")}]</div>
                  </div>
                  <div className="bg-black/20 border border-white/5 rounded-xl p-5">
                    <div className="text-[11px] font-bold uppercase tracking-widest text-emerald-400 mb-3">Top Next-Token Predictions</div>
                    <div className="space-y-1.5">
                      {forwardResult.next_token_predictions?.slice(0, 5).map((pred: any, i: number) => (
                        <div key={i} className="flex justify-between items-center font-mono text-xs bg-black/40 px-3 py-1.5 rounded border border-white/5">
                          <span className="text-foreground/80">ID: {pred.token_id}</span>
                          <span className="text-emerald-400/80">{pred.logit?.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {forwardResult.layer_shapes && (
                  <div className="pt-2">
                    <h3 className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground mb-3 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-white/20"></span> Activation Tensor Shapes
                    </h3>
                    <div className="bg-black/40 border border-white/5 rounded-xl p-4 font-mono text-xs overflow-x-auto custom-scrollbar">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
                        {Object.entries(forwardResult.layer_shapes).map(([key, val]) => (
                          <div key={key} className="flex justify-between items-center py-1 border-b border-white/5 last:border-0">
                            <span className="text-blue-400/80">{key}</span>
                            <span className="text-emerald-400/80">{String(val)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Educational */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="glass rounded-2xl p-6"
          >
            <h2 className="text-lg font-semibold mb-4 text-foreground/90 border-b border-white/10 pb-2">Sampling Strategies</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-muted-foreground">
              <div className="p-4 rounded-xl bg-black/20 border border-white/5 hover:bg-white/5 transition-colors">
                <h3 className="font-semibold text-primary mb-2 text-xs uppercase tracking-widest">Temperature</h3>
                <p className="leading-relaxed text-xs">Scales logits before softmax. Low values (0.1) make output deterministic; high values (2.0) increase randomness.</p>
              </div>
              <div className="p-4 rounded-xl bg-black/20 border border-white/5 hover:bg-white/5 transition-colors">
                <h3 className="font-semibold text-amber-500 mb-2 text-xs uppercase tracking-widest">Top-k Sampling</h3>
                <p className="leading-relaxed text-xs">Only sample from the k most probable tokens. Prevents selection of very unlikely tokens (long tail).</p>
              </div>
              <div className="p-4 rounded-xl bg-black/20 border border-white/5 hover:bg-white/5 transition-colors">
                <h3 className="font-semibold text-accent mb-2 text-xs uppercase tracking-widest">Top-p (Nucleus)</h3>
                <p className="leading-relaxed text-xs">Sample from the smallest set of tokens whose cumulative probability exceeds p. Adapts to distribution shape dynamically.</p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </motion.div>
  );
}
