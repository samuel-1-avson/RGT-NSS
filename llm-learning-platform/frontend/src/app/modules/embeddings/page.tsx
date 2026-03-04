"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Embedding Explorer</h1>
        <p className="text-muted-foreground max-w-3xl">
          Visualize how tokens are represented as vectors in high-dimensional space, and how positional information is encoded.
        </p>
      </div>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="glass rounded-2xl p-6 space-y-4"
      >
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest mb-1 block">Token IDs (comma-separated)</label>
            <input
              type="text"
              value={tokenIds}
              onChange={(e) => setTokenIds(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-black/20 px-4 py-2.5 font-mono text-sm focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all text-foreground/90"
            />
          </div>
          <div>
            <label className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest mb-1 block">Embedding Dimension (d_model)</label>
            <input
              type="number"
              value={dModel}
              onChange={(e) => setDModel(parseInt(e.target.value) || 64)}
              min={8}
              max={1024}
              className="w-full rounded-xl border border-white/10 bg-black/20 px-4 py-2.5 text-sm focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all text-foreground/90"
            />
          </div>
        </div>

        {/* Positional Encoding Selection */}
        <div className="pt-2">
          <label className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest mb-2 block">Positional Encoding</label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {ENCODING_TYPES.map((enc) => (
              <button
                key={enc.value}
                onClick={() => setEncoding(enc.value)}
                className={cn(
                  "p-4 rounded-xl border text-left transition-all relative overflow-hidden group",
                  encoding === enc.value
                    ? "border-primary/50 bg-primary/5"
                    : "border-white/10 hover:border-white/20 hover:bg-white/5"
                )}
              >
                {encoding === enc.value && (
                  <motion.div
                    layoutId="activeEncoding"
                    className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-50"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                <div className="font-semibold text-[15px] mb-1 text-foreground/90 relative z-10">{enc.label}</div>
                <div className="text-xs text-muted-foreground leading-relaxed relative z-10">{enc.description}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap gap-3 pt-4">
          <button
            onClick={() => analyzeGeometry.mutate()}
            className="px-6 py-2.5 bg-primary hover:bg-primary/90 text-primary-foreground rounded-xl font-medium transition-all shadow-lg shadow-primary/20 text-sm"
          >
            Analyze Geometry
          </button>
          <button
            onClick={() => findSimilar.mutate()}
            className="px-6 py-2.5 glass rounded-xl font-medium transition-all hover:bg-white/10 text-sm border-white/10"
          >
            Find Similar
          </button>
          <button
            onClick={() => getScatter.mutate()}
            className="px-6 py-2.5 glass rounded-xl font-medium transition-all hover:bg-white/10 text-sm border-white/10"
          >
            3D Scatter Plot
          </button>
        </div>
      </motion.div>

      {/* Geometry Analysis */}
      {geometryResult && (
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass rounded-2xl p-6 border-primary/20"
        >
          <h2 className="text-lg font-semibold text-primary mb-4 pb-2 border-b border-white/10">Embedding Space Geometry</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              { label: "Mean Norm", value: geometryResult.mean_norm?.toFixed(3) },
              { label: "Std Norm", value: geometryResult.std_norm?.toFixed(3) },
              { label: "Isotropy", value: geometryResult.isotropy_score?.toFixed(3) },
              { label: "Effective Dim", value: geometryResult.effective_dimensionality?.toFixed(1) },
              { label: "Mean Cos Sim", value: geometryResult.mean_cosine_similarity?.toFixed(3) },
            ].map((m) => (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                key={m.label}
                className="text-center p-4 rounded-xl bg-black/20 border border-white/5"
              >
                <div className="text-2xl font-bold gradient-text mb-1">{m.value}</div>
                <div className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">{m.label}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Similarity Results */}
      {similarityResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-2xl p-6"
        >
          <h2 className="text-lg font-semibold text-foreground/90 mb-4 pb-2 border-b border-white/10">
            Most Similar to Token {similarityResult.query_id}
          </h2>
          <div className="space-y-3">
            {similarityResult.similar_tokens?.map((item: any, i: number) => (
              <div key={i} className="flex items-center gap-4 bg-black/10 p-2 rounded-lg border border-white/5">
                <span className="text-xs font-mono w-16 text-muted-foreground bg-white/5 px-2 py-1 rounded text-center">ID: {item.token_id}</span>
                <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(item.score * 100, 0)}%` }}
                    transition={{ duration: 1, delay: i * 0.1, ease: "easeOut" }}
                    className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                  />
                </div>
                <span className="text-xs font-mono text-muted-foreground w-16 text-right">
                  {item.score?.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* 3D Scatter Preview */}
      {scatterData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-2xl p-6"
        >
          <h2 className="text-lg font-semibold text-foreground/90 mb-4 pb-2 border-b border-white/10">3D Embedding Space</h2>
          <div className="bg-black/40 rounded-xl p-4 min-h-64 flex flex-col items-center justify-center border border-white/5">
            <div className="text-center text-muted-foreground">
              <p className="text-sm mb-2 text-foreground/80 font-medium">
                {scatterData.points?.length} tokens projected to 3D via PCA
              </p>
              <p className="text-xs opacity-70 mb-4">
                Three.js 3D visualization renders here with orbit controls
              </p>
              <div className="flex flex-wrap justify-center gap-2 text-xs">
                {scatterData.points?.slice(0, 5).map((p: any) => (
                  <div key={p.token_id} className="bg-white/5 border border-white/5 rounded-md p-2">
                    <span className="text-primary font-bold">#{p.token_id}</span>
                    <div className="font-mono mt-1 opacity-70">
                      [{p.x?.toFixed(1)}, {p.y?.toFixed(1)}, {p.z?.toFixed(1)}]
                    </div>
                  </div>
                ))}
                {scatterData.points?.length > 5 && (
                  <div className="bg-white/5 border border-white/5 rounded-md p-2 flex items-center justify-center text-xs italic opacity-70">
                    +{scatterData.points.length - 5} more
                  </div>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
