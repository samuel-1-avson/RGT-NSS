"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Attention Visualizer</h1>
        <p className="text-muted-foreground">
          Explore how self-attention works by visualizing attention weights
          across different heads and attention types.
        </p>
      </div>

      {/* Controls */}
      <div className="glass rounded-2xl p-6 space-y-4">
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="text-sm font-medium mb-1 block">Sequence Length</label>
            <input
              type="range"
              min={2}
              max={32}
              value={seqLen}
              onChange={(e) => setSeqLen(parseInt(e.target.value))}
              className="w-full"
            />
            <span className="text-sm text-muted-foreground">{seqLen} tokens</span>
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Number of Heads</label>
            <input
              type="range"
              min={1}
              max={16}
              value={numHeads}
              onChange={(e) => {
                const next = parseInt(e.target.value, 10);
                setNumHeads(getNearestValidHead(dModel, next));
              }}
              className="w-full"
            />
            <span className="text-sm text-muted-foreground">{numHeads} heads</span>
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">d_model</label>
            <select
              value={dModel}
              onChange={(e) => {
                const nextDModel = parseInt(e.target.value, 10);
                setDModel(nextDModel);
                setNumHeads((prev) => getNearestValidHead(nextDModel, prev));
              }}
              className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
            >
              {[32, 64, 128, 256].map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Attention Type */}
        <div className="grid grid-cols-4 gap-3">
          {ATTENTION_TYPES.map((t) => (
            <button
              key={t.value}
              onClick={() => setAttentionType(t.value)}
              className={cn(
                "p-3 rounded-xl border text-left transition-all",
                attentionType === t.value
                  ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20"
                  : "border-border hover:border-primary-300"
              )}
            >
              <div className="font-medium text-sm">{t.label}</div>
              <div className="text-xs text-muted-foreground">{t.description}</div>
            </button>
          ))}
        </div>

        <button
          onClick={() => computeAttention.mutate()}
          disabled={computeAttention.isPending}
          className="px-6 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors disabled:opacity-50"
        >
          {computeAttention.isPending ? "Computing..." : "Compute Attention"}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="grid grid-cols-3 gap-6">
          {/* Heatmap */}
          <div className="col-span-2 glass rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">
                Attention Heatmap — Head {selectedHead}
              </h2>
              <div className="flex gap-1">
                {result.heads?.map((_: any, i: number) => (
                  <button
                    key={i}
                    onClick={() => setSelectedHead(i)}
                    className={cn(
                      "w-8 h-8 rounded-lg text-xs font-medium transition-colors",
                      i === selectedHead
                        ? "bg-primary-600 text-white"
                        : "bg-muted hover:bg-muted/80"
                    )}
                  >
                    H{i}
                  </button>
                ))}
              </div>
            </div>

            {/* Heatmap Grid */}
            {headData && (
              <div className="overflow-auto">
                <div
                  className="grid gap-0.5"
                  style={{
                    gridTemplateColumns: `40px repeat(${seqLen}, 1fr)`,
                  }}
                >
                  {/* Column headers */}
                  <div />
                  {Array.from({ length: seqLen }, (_, i) => (
                    <div
                      key={i}
                      className="text-center text-xs text-muted-foreground py-1"
                    >
                      K{i}
                    </div>
                  ))}

                  {/* Rows */}
                  {headData.weights?.map((row: number[], qi: number) => (
                    <>
                      <div
                        key={`label-${qi}`}
                        className="text-xs text-muted-foreground flex items-center"
                      >
                        Q{qi}
                      </div>
                      {row.map((w: number, ki: number) => (
                        <div
                          key={`${qi}-${ki}`}
                          className={cn(
                            "heatmap-cell aspect-square rounded-sm flex items-center justify-center text-[10px] font-mono",
                            getHeatColor(w)
                          )}
                          title={`Q${qi}→K${ki}: ${w.toFixed(4)}`}
                        >
                          {seqLen <= 12 ? w.toFixed(2) : ""}
                        </div>
                      ))}
                    </>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Head Statistics */}
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-xl font-semibold">Head Statistics</h2>
            {result.heads?.map((head: any, i: number) => (
              <div
                key={i}
                onClick={() => setSelectedHead(i)}
                className={cn(
                  "p-3 rounded-xl border cursor-pointer transition-all",
                  i === selectedHead
                    ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20"
                    : "border-border hover:border-primary-300"
                )}
              >
                <div className="flex justify-between mb-1">
                  <span className="font-medium text-sm">Head {i}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                  <div>
                    Entropy:{" "}
                    <span className="text-foreground">{head.entropy?.toFixed(3)}</span>
                  </div>
                  <div>
                    Sparsity:{" "}
                    <span className="text-foreground">
                      {(head.sparsity * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Educational */}
      <div className="glass rounded-2xl p-6">
        <h2 className="text-xl font-semibold mb-4">Understanding Attention</h2>
        <div className="grid grid-cols-2 gap-6 text-sm text-muted-foreground">
          <div>
            <h3 className="font-medium text-foreground mb-1">Self-Attention Mechanism</h3>
            <p>
              Each position creates Query (Q), Key (K), and Value (V) vectors.
              Attention weights are computed as softmax(QK^T / √d_k),
              then used to create a weighted sum of values.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-foreground mb-1">Multi-Head Attention</h3>
            <p>
              Multiple attention heads run in parallel, each learning different
              patterns — some attend to nearby tokens, others to syntactic
              structure or semantic relationships.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
