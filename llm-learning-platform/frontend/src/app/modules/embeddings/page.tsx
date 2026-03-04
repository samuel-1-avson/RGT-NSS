"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { embeddingsApi, visualizationsApi } from "@/lib/api";
import { cn } from "@/lib/utils";

const ENCODING_TYPES = [
  { value: "sinusoidal", label: "Sinusoidal", description: "Fixed sin/cos (Vaswani 2017)" },
  { value: "learned", label: "Learned", description: "Trainable embeddings" },
  { value: "rope", label: "RoPE", description: "Rotary Position Embedding" },
  { value: "alibi", label: "ALiBi", description: "Attention Linear Biases" },
];

export default function EmbeddingsPage() {
  const [tokenIds, setTokenIds] = useState("1, 5, 10, 20, 30, 40, 50");
  const [dModel, setDModel] = useState(64);
  const [encoding, setEncoding] = useState("sinusoidal");
  const [geometryResult, setGeometryResult] = useState<any>(null);
  const [similarityResult, setSimilarityResult] = useState<any>(null);
  const [scatterData, setScatterData] = useState<any>(null);

  const ids = tokenIds.split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n));

  const analyzeGeometry = useMutation({
    mutationFn: () => embeddingsApi.geometry(ids),
    onSuccess: (data) => setGeometryResult(data),
  });

  const findSimilar = useMutation({
    mutationFn: () =>
      embeddingsApi.similarity({ query_id: ids[0] || 1, d_model: dModel, top_k: 10 }),
    onSuccess: (data) => setSimilarityResult(data),
  });

  const getScatter = useMutation({
    mutationFn: () =>
      visualizationsApi.embeddingScatter({
        token_ids: ids,
        d_model: dModel,
        n_components: 3,
      }),
    onSuccess: (data) => setScatterData(data),
  });

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Embedding Explorer</h1>
        <p className="text-muted-foreground">
          Visualize how tokens are represented as vectors in high-dimensional space, and how positional information is encoded.
        </p>
      </div>

      {/* Controls */}
      <div className="glass rounded-2xl p-6 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium mb-1 block">Token IDs (comma-separated)</label>
            <input
              type="text"
              value={tokenIds}
              onChange={(e) => setTokenIds(e.target.value)}
              className="w-full rounded-xl border border-border bg-background px-4 py-2 font-mono text-sm focus:ring-2 focus:ring-primary-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Embedding Dimension (d_model)</label>
            <input
              type="number"
              value={dModel}
              onChange={(e) => setDModel(parseInt(e.target.value) || 64)}
              min={8}
              max={1024}
              className="w-full rounded-xl border border-border bg-background px-4 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Positional Encoding Selection */}
        <div>
          <label className="text-sm font-medium mb-2 block">Positional Encoding</label>
          <div className="grid grid-cols-4 gap-3">
            {ENCODING_TYPES.map((enc) => (
              <button
                key={enc.value}
                onClick={() => setEncoding(enc.value)}
                className={cn(
                  "p-3 rounded-xl border text-left transition-all",
                  encoding === enc.value
                    ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20"
                    : "border-border hover:border-primary-300"
                )}
              >
                <div className="font-medium text-sm">{enc.label}</div>
                <div className="text-xs text-muted-foreground">{enc.description}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={() => analyzeGeometry.mutate()}
            className="px-6 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors"
          >
            Analyze Geometry
          </button>
          <button
            onClick={() => findSimilar.mutate()}
            className="px-6 py-2.5 glass rounded-xl font-medium hover:bg-white/90 dark:hover:bg-gray-800/90 transition-colors"
          >
            Find Similar
          </button>
          <button
            onClick={() => getScatter.mutate()}
            className="px-6 py-2.5 glass rounded-xl font-medium hover:bg-white/90 dark:hover:bg-gray-800/90 transition-colors"
          >
            3D Scatter Plot
          </button>
        </div>
      </div>

      {/* Geometry Analysis */}
      {geometryResult && (
        <div className="glass rounded-2xl p-6">
          <h2 className="text-xl font-semibold mb-4">Embedding Space Geometry</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              { label: "Mean Norm", value: geometryResult.mean_norm?.toFixed(3) },
              { label: "Std Norm", value: geometryResult.std_norm?.toFixed(3) },
              { label: "Isotropy", value: geometryResult.isotropy_score?.toFixed(3) },
              { label: "Effective Dim", value: geometryResult.effective_dimensionality?.toFixed(1) },
              { label: "Mean Cos Sim", value: geometryResult.mean_cosine_similarity?.toFixed(3) },
            ].map((m) => (
              <div key={m.label} className="text-center p-3 rounded-xl bg-muted/50">
                <div className="text-2xl font-bold gradient-text">{m.value}</div>
                <div className="text-xs text-muted-foreground">{m.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Similarity Results */}
      {similarityResult && (
        <div className="glass rounded-2xl p-6">
          <h2 className="text-xl font-semibold mb-4">
            Most Similar to Token {similarityResult.query_id}
          </h2>
          <div className="space-y-2">
            {similarityResult.similar_tokens?.map((item: any, i: number) => (
              <div key={i} className="flex items-center gap-3">
                <span className="text-sm font-mono w-16">ID: {item.token_id}</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-primary-500 to-accent-500 rounded-full"
                    style={{ width: `${Math.max(item.score * 100, 0)}%` }}
                  />
                </div>
                <span className="text-sm text-muted-foreground w-16 text-right">
                  {item.score?.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 3D Scatter Preview */}
      {scatterData && (
        <div className="glass rounded-2xl p-6">
          <h2 className="text-xl font-semibold mb-4">3D Embedding Space</h2>
          <div className="bg-gray-900 rounded-xl p-4 h-64 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <p className="text-sm mb-2">
                {scatterData.points?.length} tokens projected to 3D via PCA
              </p>
              <p className="text-xs">
                Three.js 3D visualization renders here with orbit controls
              </p>
              <div className="mt-4 grid grid-cols-5 gap-2 text-xs">
                {scatterData.points?.slice(0, 10).map((p: any) => (
                  <div key={p.token_id} className="bg-gray-800 rounded p-1">
                    <span className="text-primary-400">#{p.token_id}</span>
                    <br />
                    ({p.x?.toFixed(1)}, {p.y?.toFixed(1)}, {p.z?.toFixed(1)})
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
