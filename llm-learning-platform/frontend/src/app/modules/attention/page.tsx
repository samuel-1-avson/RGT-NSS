"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { visualizationsApi } from "@/lib/api";
import { cn } from "@/lib/utils";

const ATTENTION_TYPES = [
  { value: "full", label: "Full Attention", description: "Standard O(n²) attention" },
  { value: "local", label: "Local Attention", description: "Sliding window pattern" },
  { value: "sparse", label: "Sparse Attention", description: "Strided + local pattern" },
  { value: "linear", label: "Linear Attention", description: "O(n) approximation" },
];

function getValidHeadOptions(dModel: number): number[] {
  return [1, 2, 4, 8, 16].filter((h) => h <= 16 && dModel % h === 0);
}

function getNearestValidHead(dModel: number, requestedHeads: number): number {
  const valid = getValidHeadOptions(dModel);
  const exact = valid.find((h) => h === requestedHeads);
  if (exact) return exact;
  const lower = [...valid].reverse().find((h) => h <= requestedHeads);
  return lower ?? valid[0] ?? 1;
}

function getHeatColor(value: number): string {
  // Viridis-inspired: dark purple → blue → green → yellow
  if (value < 0.1) return "bg-purple-950";
  if (value < 0.2) return "bg-indigo-900";
  if (value < 0.3) return "bg-blue-800";
  if (value < 0.4) return "bg-cyan-700";
  if (value < 0.5) return "bg-teal-600";
  if (value < 0.6) return "bg-emerald-500";
  if (value < 0.7) return "bg-green-400";
  if (value < 0.8) return "bg-lime-400";
  if (value < 0.9) return "bg-yellow-300";
  return "bg-yellow-200";
}

export default function AttentionPage() {
  const [seqLen, setSeqLen] = useState(8);
  const [numHeads, setNumHeads] = useState(4);
  const [dModel, setDModel] = useState(64);
  const [attentionType, setAttentionType] = useState("full");
  const [selectedHead, setSelectedHead] = useState(0);
  const [result, setResult] = useState<any>(null);

  const computeAttention = useMutation({
    mutationFn: () =>
      visualizationsApi.attentionHeatmap({
        seq_len: seqLen,
        num_heads: getNearestValidHead(dModel, numHeads),
        d_model: dModel,
        attention_type: attentionType,
      }),
    onSuccess: (data) => {
      setResult(data);
      setSelectedHead(0);
    },
  });

  const headData = result?.heads?.[selectedHead];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Attention Visualizer</h1>
        <p className="text-muted-foreground max-w-3xl">
          Explore how self-attention works by visualizing attention weights
          across different heads and attention types.
        </p>
      </div>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="glass rounded-2xl p-6 space-y-6"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest mb-3 block">Sequence Length</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={2}
                max={32}
                value={seqLen}
                onChange={(e) => setSeqLen(parseInt(e.target.value))}
                className="w-full accent-primary h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-primary/50"
              />
              <span className="text-sm font-mono text-muted-foreground min-w-[50px]">{seqLen} tok</span>
            </div>
          </div>
          <div>
            <label className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest mb-3 block">Number of Heads</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={1}
                max={16}
                value={numHeads}
                onChange={(e) => {
                  const next = parseInt(e.target.value, 10);
                  setNumHeads(getNearestValidHead(dModel, next));
                }}
                className="w-full accent-primary h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-primary/50"
              />
              <span className="text-sm font-mono text-muted-foreground min-w-[50px]">{numHeads} h</span>
            </div>
          </div>
          <div>
            <label className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest mb-2 block">Model Dimensions (d_model)</label>
            <select
              value={dModel}
              onChange={(e) => {
                const nextDModel = parseInt(e.target.value, 10);
                setDModel(nextDModel);
                setNumHeads((prev) => getNearestValidHead(nextDModel, prev));
              }}
              className="w-full rounded-xl border border-white/10 bg-black/20 px-4 py-2.5 text-sm outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 transition-all text-foreground/90 font-mono"
            >
              {[32, 64, 128, 256].map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Attention Type */}
        <div className="pt-2">
          <label className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest mb-2 block">Attention Architecture</label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {ATTENTION_TYPES.map((t) => (
              <button
                key={t.value}
                onClick={() => setAttentionType(t.value)}
                className={cn(
                  "p-4 rounded-xl border text-left transition-all relative overflow-hidden group",
                  attentionType === t.value
                    ? "border-primary/50 bg-primary/5"
                    : "border-white/10 hover:border-white/20 hover:bg-white/5"
                )}
              >
                {attentionType === t.value && (
                  <motion.div
                    layoutId="activeAttentionType"
                    className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-50"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                <div className="font-semibold text-[15px] mb-1 text-foreground/90 relative z-10">{t.label}</div>
                <div className="text-xs text-muted-foreground leading-relaxed relative z-10">{t.description}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="pt-2">
          <button
            onClick={() => computeAttention.mutate()}
            disabled={computeAttention.isPending}
            className="px-6 py-2.5 bg-primary hover:bg-primary/90 text-primary-foreground rounded-xl font-medium transition-all shadow-lg shadow-primary/20 disabled:opacity-50 disabled:shadow-none text-sm"
          >
            {computeAttention.isPending ? "Computing..." : "Compute Attention Weights"}
          </button>
        </div>
      </motion.div>

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          {/* Heatmap */}
          <div className="col-span-1 md:col-span-2 glass rounded-2xl p-6 border-primary/20">
            <div className="flex items-center justify-between mb-4 pb-2 border-b border-white/10">
              <h2 className="text-lg font-semibold text-primary">
                Attention Heatmap <span className="text-muted-foreground font-normal">| Head {selectedHead}</span>
              </h2>
              <div className="flex gap-2">
                {result.heads?.map((_: any, i: number) => (
                  <button
                    key={i}
                    onClick={() => setSelectedHead(i)}
                    className={cn(
                      "w-8 h-8 rounded-lg text-[11px] font-bold transition-all flex items-center justify-center",
                      i === selectedHead
                        ? "bg-primary text-primary-foreground shadow-sm shadow-primary/20"
                        : "bg-white/5 hover:bg-white/10 text-muted-foreground border border-white/5"
                    )}
                  >
                    H{i}
                  </button>
                ))}
              </div>
            </div>

            {/* Heatmap Grid */}
            {headData && (
              <div className="overflow-x-auto pb-4 custom-scrollbar">
                <div
                  className="grid gap-0.5 min-w-max"
                  style={{
                    gridTemplateColumns: `40px repeat(${seqLen}, minmax(32px, 1fr))`,
                  }}
                >
                  {/* Column headers */}
                  <div />
                  {Array.from({ length: seqLen }, (_, i) => (
                    <div
                      key={i}
                      className="text-center text-[10px] font-mono text-muted-foreground py-1"
                    >
                      K{i}
                    </div>
                  ))}

                  {/* Rows */}
                  {headData.weights?.map((row: number[], qi: number) => (
                    <div key={`row-${qi}`} className="contents">
                      <div
                        className="text-[10px] font-mono text-muted-foreground flex items-center justify-end pr-2"
                      >
                        Q{qi}
                      </div>
                      {row.map((w: number, ki: number) => (
                        <div
                          key={`${qi}-${ki}`}
                          className={cn(
                            "heatmap-cell aspect-square rounded-sm flex items-center justify-center text-[10px] font-mono border border-black/20",
                            getHeatColor(w)
                          )}
                          title={`Q${qi}→K${ki}: ${w.toFixed(4)}`}
                        >
                          {seqLen <= 12 ? (w > 0.05 ? w.toFixed(2).replace('0.', '.') : "") : ""}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Head Statistics */}
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Head Statistics</h2>
            <div className="space-y-3">
              {result.heads?.map((head: any, i: number) => (
                <div
                  key={i}
                  onClick={() => setSelectedHead(i)}
                  className={cn(
                    "p-4 rounded-xl border cursor-pointer transition-all relative overflow-hidden",
                    i === selectedHead
                      ? "border-primary/50 bg-primary/5"
                      : "border-white/10 hover:border-white/20 bg-black/10"
                  )}
                >
                  {i === selectedHead && (
                    <motion.div
                      layoutId="activeHeadStat"
                      className="absolute inset-0 bg-primary/10 border border-primary/20 rounded-xl"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                  <div className="flex justify-between items-center mb-2 relative z-10">
                    <span className="font-semibold text-sm text-foreground/90">Head {i}</span>
                    {i === selectedHead && <span className="w-1.5 h-1.5 rounded-full bg-primary" />}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px] text-muted-foreground font-mono relative z-10">
                    <div className="flex flex-col">
                      <span className="opacity-70">Entropy</span>
                      <span className="text-foreground text-xs mt-0.5">{head.entropy?.toFixed(3)}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="opacity-70">Sparsity</span>
                      <span className="text-foreground text-xs mt-0.5">
                        {(head.sparsity * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Educational Callouts */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-8 border-t border-white/5"
      >
        <div className="p-5 rounded-2xl border-l-2 border-primary bg-primary/5">
          <h3 className="font-semibold text-primary mb-2 text-sm">Self-Attention Mechanism</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Each position creates Query (Q), Key (K), and Value (V) vectors.
            Attention weights are computed as softmax(QK^T / √d_k),
            then used to create a weighted sum of values. This allows words to seamlessly route context to one another.
          </p>
        </div>
        <div className="p-5 rounded-2xl border-l-2 border-accent bg-accent/5">
          <h3 className="font-semibold text-accent mb-2 text-sm">Multi-Head Attention</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Multiple attention heads operate in parallel subspaces, each learning different
            patterns — some heads focus heavily on nearby tokens, while others identify broader syntactic
            rules and long-range semantic dependencies.
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}
