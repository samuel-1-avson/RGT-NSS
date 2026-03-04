"use client";

import { useState, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Training Dashboard</h1>
        <p className="text-muted-foreground">
          Train models end-to-end with real-time loss curves, gradient monitoring, and hyperparameter controls.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Configuration */}
        <div className="glass rounded-2xl p-6 space-y-4">
          <h2 className="text-lg font-semibold">Training Config</h2>

          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium mb-1 block">Model Preset</label>
              <select
                value={config.model_preset}
                onChange={(e) => setConfig((p) => ({ ...p, model_preset: e.target.value }))}
                className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
              >
                {["nano", "micro", "mini", "small"].map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            {[
              { key: "num_epochs", label: "Epochs", min: 1, max: 100, step: 1 },
              { key: "batch_size", label: "Batch Size", min: 4, max: 256, step: 4 },
              { key: "max_steps", label: "Max Steps", min: 50, max: 5000, step: 50 },
              { key: "warmup_steps", label: "Warmup Steps", min: 0, max: 500, step: 10 },
              { key: "grad_clip", label: "Gradient Clip", min: 0.1, max: 10, step: 0.1 },
            ].map((param) => (
              <div key={param.key}>
                <label className="text-sm font-medium mb-1 block">
                  {param.label}: <span className="text-primary-500">{(config as any)[param.key]}</span>
                </label>
                <input
                  type="range"
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  value={(config as any)[param.key]}
                  onChange={(e) =>
                    setConfig((prev) => ({ ...prev, [param.key]: parseFloat(e.target.value) }))
                  }
                  className="w-full"
                />
              </div>
            ))}

            <div>
              <label className="text-sm font-medium mb-1 block">
                Learning Rate: <span className="text-primary-500">{config.learning_rate.toExponential(1)}</span>
              </label>
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
                className="w-full"
              />
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => startTraining.mutate()}
              disabled={startTraining.isPending}
              className="flex-1 px-4 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors disabled:opacity-50"
            >
              {startTraining.isPending ? "Training..." : "Start Training"}
            </button>
            <button
              onClick={() => previewLossCurve.mutate()}
              className="px-4 py-2.5 glass rounded-xl text-sm hover:bg-white/90 dark:hover:bg-gray-800/90 transition-colors"
            >
              Preview
            </button>
          </div>
        </div>

        {/* Live Metrics */}
        <div className="col-span-2 space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: "Loss", value: latestMetric?.loss?.toFixed(4) || "—", color: "text-blue-500" },
              { label: "Perplexity", value: latestMetric?.perplexity?.toFixed(1) || "—", color: "text-purple-500" },
              { label: "Grad Norm", value: latestMetric?.grad_norm?.toFixed(4) || "—", color: "text-emerald-500" },
              { label: "Tokens/sec", value: latestMetric?.tokens_per_sec?.toFixed(0) || "—", color: "text-amber-500" },
            ].map((s) => (
              <div key={s.label} className="glass rounded-xl p-4 text-center">
                <div className={`text-2xl font-bold font-mono ${s.color}`}>{s.value}</div>
                <div className="text-xs text-muted-foreground">{s.label}</div>
              </div>
            ))}
          </div>

          {/* Loss Curve */}
          <div className="glass rounded-2xl p-6">
            <h2 className="text-lg font-semibold mb-4">Loss Curve</h2>
            <div className="h-64 bg-gray-900 rounded-xl p-4 flex items-end gap-px overflow-hidden">
              {metrics.length > 0 ? (
                metrics.slice(-100).map((m, i) => {
                  const maxLoss = Math.max(...metrics.map((mm) => mm.loss));
                  const height = Math.max((m.loss / maxLoss) * 100, 2);
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-gradient-to-t from-primary-600 to-primary-400 rounded-t-sm min-w-[2px]"
                      style={{ height: `${height}%` }}
                      title={`Step ${m.step}: Loss ${m.loss.toFixed(4)}`}
                    />
                  );
                })
              ) : lossCurveData ? (
                lossCurveData.train_loss?.slice(-100).map((loss: number, i: number) => {
                  const maxLoss = Math.max(...lossCurveData.train_loss);
                  const height = Math.max((loss / maxLoss) * 100, 2);
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-gradient-to-t from-blue-600/50 to-blue-400/30 rounded-t-sm min-w-[2px]"
                      style={{ height: `${height}%` }}
                    />
                  );
                })
              ) : (
                <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
                  Start training or preview to see the loss curve
                </div>
              )}
            </div>
            {metrics.length > 0 && (
              <div className="flex justify-between mt-2 text-xs text-muted-foreground">
                <span>Step 0</span>
                <span>Step {metrics[metrics.length - 1]?.step}</span>
              </div>
            )}
          </div>

          {/* Learning Rate Schedule */}
          {metrics.length > 0 && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Learning Rate Schedule</h2>
              <div className="h-32 bg-gray-900 rounded-xl p-4 flex items-end gap-px overflow-hidden">
                {metrics.slice(-100).map((m, i) => {
                  const maxLR = Math.max(...metrics.map((mm) => mm.learning_rate));
                  const height = maxLR > 0 ? Math.max((m.learning_rate / maxLR) * 100, 2) : 2;
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-gradient-to-t from-emerald-600 to-emerald-400 rounded-t-sm min-w-[2px]"
                      style={{ height: `${height}%` }}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
