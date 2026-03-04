"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Inference Playground</h1>
        <p className="text-muted-foreground">
          Explore autoregressive text generation with different sampling strategies and see how temperature, top-k, and top-p affect output.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Controls */}
        <div className="glass rounded-2xl p-6 space-y-4">
          <h2 className="text-lg font-semibold">Generation Settings</h2>

          <div>
            <label className="text-sm font-medium mb-1 block">Prompt Token IDs</label>
            <input
              type="text"
              value={promptIds}
              onChange={(e) => setPromptIds(e.target.value)}
              className="w-full rounded-xl border border-border bg-background px-4 py-2 font-mono text-sm focus:ring-2 focus:ring-primary-500 focus:outline-none"
              placeholder="2, 10, 20"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-1 block">
              Max New Tokens: <span className="text-primary-500">{maxTokens}</span>
            </label>
            <input
              type="range"
              min={1}
              max={200}
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-1 block">
              Temperature: <span className="text-primary-500">{temperature.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.01}
              max={3.0}
              step={0.01}
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Deterministic</span>
              <span>Creative</span>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium mb-1 block">
              Top-k: <span className="text-primary-500">{topK === 0 ? "Off" : topK}</span>
            </label>
            <input
              type="range"
              min={0}
              max={100}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-1 block">
              Top-p (Nucleus): <span className="text-primary-500">{topP.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={topP}
              onChange={(e) => setTopP(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => generate.mutate()}
              disabled={generate.isPending}
              className="flex-1 px-4 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors disabled:opacity-50"
            >
              {generate.isPending ? "Generating..." : "Generate"}
            </button>
            <button
              onClick={() => forward.mutate()}
              disabled={forward.isPending}
              className="px-4 py-2.5 glass rounded-xl text-sm hover:bg-white/90 dark:hover:bg-gray-800/90 transition-colors disabled:opacity-50"
            >
              Forward Pass
            </button>
          </div>
        </div>

        {/* Output */}
        <div className="col-span-2 space-y-6">
          {/* Generated Tokens */}
          {result && (
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Generated Output</h2>
              <div className="flex flex-wrap gap-1">
                {result.generated_ids?.map((id: number, i: number) => {
                  const isPrompt = i < promptIds.split(",").length;
                  return (
                    <span
                      key={i}
                      className={cn(
                        "px-2 py-1 rounded-lg text-sm font-mono",
                        isPrompt
                          ? "bg-amber-100 dark:bg-amber-900/40 border border-amber-300 dark:border-amber-700"
                          : "bg-primary-100 dark:bg-primary-900/40 border border-primary-300 dark:border-primary-700"
                      )}
                    >
                      {id}
                    </span>
                  );
                })}
              </div>

              <div className="text-sm text-muted-foreground">
                <span className="text-amber-500">Prompt</span> |{" "}
                <span className="text-primary-500">Generated</span> |{" "}
                Total: {result.generated_ids?.length} tokens
              </div>

              {/* Generation Steps */}
              {result.steps?.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium mb-2">Generation Steps (first 10)</h3>
                  <div className="space-y-1">
                    {result.steps.map((step: any, i: number) => (
                      <div key={i} className="flex items-center gap-3 text-xs font-mono">
                        <span className="text-muted-foreground w-12">Step {i}</span>
                        <span>Token: {step.chosen_token}</span>
                        <span className="text-muted-foreground">
                          prob: {step.probability?.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Forward Pass */}
          {forwardResult && (
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-semibold">Forward Pass Analysis</h2>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="p-3 rounded-xl bg-muted/50">
                  <div className="text-muted-foreground mb-1">Logits Shape</div>
                  <div className="font-mono">[{forwardResult.logits_shape?.join(", ")}]</div>
                </div>
                <div className="p-3 rounded-xl bg-muted/50">
                  <div className="text-muted-foreground mb-1">Top Predictions</div>
                  <div className="space-y-1">
                    {forwardResult.next_token_predictions?.slice(0, 5).map((pred: any, i: number) => (
                      <div key={i} className="flex justify-between font-mono text-xs">
                        <span>Token {pred.token_id}</span>
                        <span className="text-muted-foreground">{pred.logit?.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {forwardResult.layer_shapes && (
                <div>
                  <h3 className="text-sm font-medium mb-2">Intermediate Shapes</h3>
                  <div className="code-block">
                    {Object.entries(forwardResult.layer_shapes).map(([key, val]) => (
                      <div key={key}>
                        <span className="text-blue-400">{key}</span>:{" "}
                        <span className="text-green-400">{String(val)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Educational */}
          <div className="glass rounded-2xl p-6">
            <h2 className="text-lg font-semibold mb-4">Sampling Strategies</h2>
            <div className="grid grid-cols-3 gap-4 text-sm text-muted-foreground">
              <div>
                <h3 className="font-medium text-foreground mb-1">Temperature</h3>
                <p>Scales logits before softmax. Low values (0.1) make output deterministic; high values (2.0) increase randomness.</p>
              </div>
              <div>
                <h3 className="font-medium text-foreground mb-1">Top-k Sampling</h3>
                <p>Only sample from the k most probable tokens. Prevents selection of very unlikely tokens.</p>
              </div>
              <div>
                <h3 className="font-medium text-foreground mb-1">Top-p (Nucleus)</h3>
                <p>Sample from the smallest set of tokens whose cumulative probability exceeds p. Adapts dynamically.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
