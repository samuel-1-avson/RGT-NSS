"use client";

import { useState, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { trainingApi, visualizationsApi } from "@/lib/api";
import { useTrainingStore } from "@/lib/stores";
import { cn, formatNumber, formatDuration } from "@/lib/utils";

export default function TrainingPage() {
  const store = useTrainingStore();
  const [config, setConfig] = useState({
    model_preset: "micro",
    num_epochs: 5,
    batch_size: 32,
    learning_rate: 3e-4,
    weight_decay: 0.01,
    warmup_steps: 50,
    max_steps: 200,
    grad_clip: 1.0,
  });
  const [lossCurveData, setLossCurveData] = useState<any>(null);

  const startTraining = useMutation({
    mutationFn: () => trainingApi.start(config),
    onSuccess: (data: any) => {
      store.setSession(data.session_id);
      // Fetch completed metrics
      fetchMetrics.mutate(data.session_id);
    },
  });

  const fetchMetrics = useMutation({
    mutationFn: (sessionId: string) => trainingApi.metrics(sessionId),
    onSuccess: (data: any) => {
      store.setMetrics(data.metrics);
      store.setStatus("completed");
    },
  });

  const previewLossCurve = useMutation({
    mutationFn: () => {
      const normalizedSteps = Math.max(1, Math.floor(Number(config.max_steps) || 1));
      return visualizationsApi.lossCurve({ num_steps: normalizedSteps });
    },
    onSuccess: (data) => setLossCurveData(data),
  });

  // Get latest metrics
  const metrics = store.metrics;
  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Training Dashboard</h1>
        <p className="text-muted-foreground max-w-3xl">
          Train models end-to-end with real-time loss curves, gradient monitoring, and hyperparameter controls.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="glass rounded-2xl p-6 space-y-6 flex flex-col h-full"
        >
          <div className="pb-4 border-b border-white/10">
            <h2 className="text-lg font-semibold text-foreground/90">Training Config</h2>
            <p className="text-xs text-muted-foreground mt-1">Configure your hyperparameter sweep.</p>
          </div>

          <div className="space-y-6 flex-1">
            <div>
              <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 block">Model Architecture Preset</label>
              <div className="relative">
                <select
                  value={config.model_preset}
                  onChange={(e) => setConfig((p) => ({ ...p, model_preset: e.target.value }))}
                  className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-2.5 text-sm appearance-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all text-foreground/90 font-medium"
                >
                  {["nano", "micro", "mini", "small"].map((p) => (
                    <option key={p} value={p}>{p.toUpperCase()}</option>
                  ))}
                </select>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-muted-foreground">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
                </div>
              </div>
            </div>

            <div className="space-y-5">
              {[
                { key: "num_epochs", label: "Training Epochs", min: 1, max: 100, step: 1, color: "accent-primary" },
                { key: "batch_size", label: "Batch Size (Samples)", min: 4, max: 256, step: 4, color: "accent-accent" },
                { key: "max_steps", label: "Max Training Steps", min: 50, max: 5000, step: 50, color: "accent-primary" },
                { key: "warmup_steps", label: "LR Warmup Steps", min: 0, max: 500, step: 10, color: "accent-accent" },
                { key: "grad_clip", label: "Gradient Clipping", min: 0.1, max: 10, step: 0.1, color: "accent-emerald-500" },
              ].map((param) => (
                <div key={param.key}>
                  <div className="flex justify-between items-end mb-2">
                    <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">
                      {param.label}
                    </label>
                    <span className="text-sm font-mono text-foreground/90 bg-white/5 px-2 py-0.5 rounded border border-white/10">{(config as any)[param.key]}</span>
                  </div>
                  <input
                    type="range"
                    min={param.min}
                    max={param.max}
                    step={param.step}
                    value={(config as any)[param.key]}
                    onChange={(e) =>
                      setConfig((prev) => ({ ...prev, [param.key]: parseFloat(e.target.value) }))
                    }
                    className={`w-full ${param.color} h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-white/20`}
                  />
                </div>
              ))}

              <div className="pt-2">
                <div className="flex justify-between items-end mb-2">
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block">
                    Base Learning Rate
                  </label>
                  <span className="text-sm font-mono text-primary bg-primary/10 border border-primary/20 px-2 py-0.5 rounded shadow-sm">{config.learning_rate.toExponential(1)}</span>
                </div>
                <input
                  type="range"
                  min={-5}
                  max={-1}
                  step={0.1}
                  value={Math.log10(config.learning_rate)}
                  onChange={(e) =>
                    setConfig((prev) => ({
                      ...prev,
                      learning_rate: Math.pow(10, parseFloat(e.target.value)),
                    }))
                  }
                  className="w-full accent-primary h-1.5 bg-black/40 rounded-full appearance-none outline-none focus:ring-1 focus:ring-primary/50"
                />
              </div>
            </div>
          </div>

          <div className="flex gap-3 pt-6 border-t border-white/10 mt-auto">
            <button
              onClick={() => startTraining.mutate()}
              disabled={startTraining.isPending}
              className="flex-[2] py-2.5 bg-primary hover:bg-primary/90 text-primary-foreground rounded-xl font-medium transition-all shadow-lg shadow-primary/20 disabled:opacity-50 text-sm"
            >
              {startTraining.isPending ? "Initializing..." : "Start Run"}
            </button>
            <button
              onClick={() => previewLossCurve.mutate()}
              className="flex-1 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-sm font-medium transition-colors text-foreground/90 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={previewLossCurve.isPending}
            >
              Preview
            </button>
          </div>
        </motion.div>

        {/* Live Metrics */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="col-span-1 lg:col-span-2 space-y-6 flex flex-col"
        >
          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Loss Metrics", value: latestMetric?.loss?.toFixed(4) || "—", color: "text-blue-400", bg: "bg-blue-500/10", border: "border-blue-500/20" },
              { label: "Perplexity Diff", value: latestMetric?.perplexity?.toFixed(1) || "—", color: "text-purple-400", bg: "bg-purple-500/10", border: "border-purple-500/20" },
              { label: "Gradient Norm", value: latestMetric?.grad_norm?.toFixed(4) || "—", color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20" },
              { label: "Throughput (t/s)", value: latestMetric?.tokens_per_sec?.toFixed(0) || "—", color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" },
            ].map((s, i) => (
              <motion.div
                whileHover={{ y: -2 }}
                key={s.label}
                className={cn("glass rounded-xl p-5 text-center flex flex-col justify-center border", s.bg, s.border)}
              >
                <div className={`text-2xl md:text-3xl font-bold font-mono tracking-tight mb-1 ${s.color}`}>{s.value}</div>
                <div className="text-[10px] uppercase tracking-widest font-bold text-muted-foreground/80">{s.label}</div>
              </motion.div>
            ))}
          </div>

          {/* Loss Curve */}
          <div className="glass rounded-2xl p-6 flex-1 flex flex-col min-h-[300px]">
            <div className="flex justify-between items-center mb-4 border-b border-white/10 pb-4">
              <h2 className="text-lg font-semibold text-foreground/90">Loss Descent Trajectory</h2>
              <div className="flex items-center gap-2">
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                </span>
                <span className="text-[10px] uppercase font-bold text-primary tracking-widest">Live Telemetry</span>
              </div>
            </div>

            <div className="flex-1 min-h-[200px] bg-black/40 rounded-xl p-6 flex flex-col justify-end gap-px overflow-hidden relative border border-white/5">

              {/* Grid Lines */}
              <div className="absolute inset-x-0 bottom-1/4 border-b border-dashed border-white/5 pointer-events-none w-full" />
              <div className="absolute inset-x-0 bottom-2/4 border-b border-dashed border-white/5 pointer-events-none w-full" />
              <div className="absolute inset-x-0 bottom-3/4 border-b border-dashed border-white/5 pointer-events-none w-full" />

              <div className="flex items-end gap-[2px] h-full w-full relative z-10">
                {metrics.length > 0 ? (
                  metrics.slice(-100).map((m, i) => {
                    const maxLoss = Math.max(...metrics.map((mm) => mm.loss));
                    const height = Math.max((m.loss / maxLoss) * 100, 2);
                    return (
                      <motion.div
                        initial={{ height: 0 }}
                        animate={{ height: `${height}%` }}
                        key={i}
                        className="flex-1 bg-gradient-to-t from-primary/80 to-accent/80 rounded-t-sm min-w-[3px] hover:brightness-125 transition-all group relative cursor-pointer"
                        title={`Step ${m.step}: Loss ${m.loss.toFixed(4)}`}
                      >
                        <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-black text-white text-[10px] font-mono py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
                          {m.loss.toFixed(4)}
                        </div>
                      </motion.div>
                    );
                  })
                ) : lossCurveData ? (
                  lossCurveData.train_loss?.slice(-100).map((loss: number, i: number) => {
                    const maxLoss = Math.max(...lossCurveData.train_loss);
                    const height = Math.max((loss / maxLoss) * 100, 2);
                    return (
                      <motion.div
                        initial={{ height: 0 }}
                        animate={{ height: `${height}%` }}
                        key={i}
                        className="flex-1 bg-gradient-to-t from-blue-600/60 to-cyan-400/60 rounded-t-sm min-w-[3px] opacity-70"
                      />
                    );
                  })
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-muted-foreground/50 text-sm font-medium">
                    Initialize training run to map loss trajectory
                  </div>
                )}
              </div>
            </div>
            {metrics.length > 0 && (
              <div className="flex justify-between mt-3 px-2 text-[10px] uppercase tracking-widest font-bold text-muted-foreground/60">
                <span>Step Origin</span>
                <span>Current: {metrics[metrics.length - 1]?.step}</span>
              </div>
            )}
          </div>

          {/* Learning Rate Schedule */}
          {metrics.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass rounded-2xl p-6"
            >
              <h2 className="text-lg font-semibold mb-4 text-foreground/90 border-b border-white/10 pb-2">LR Decay Schedule</h2>
              <div className="h-24 bg-black/40 border border-white/5 rounded-xl p-4 flex items-end gap-[2px] overflow-hidden">
                {metrics.slice(-100).map((m, i) => {
                  const maxLR = Math.max(...metrics.map((mm) => mm.learning_rate));
                  const height = maxLR > 0 ? Math.max((m.learning_rate / maxLR) * 100, 2) : 2;
                  return (
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: `${height}%` }}
                      key={i}
                      className="flex-1 bg-gradient-to-t from-emerald-600/80 to-emerald-400/80 rounded-t-sm min-w-[3px]"
                    />
                  );
                })}
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
}
