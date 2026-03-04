"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { tokenizationApi } from "@/lib/api";
import { cn } from "@/lib/utils";

const STRATEGIES = [
  { value: "character", label: "Character", description: "Split into individual characters" },
  { value: "word", label: "Word", description: "Split on whitespace and punctuation" },
  { value: "bpe", label: "BPE", description: "Byte Pair Encoding — learned subword merges" },
  { value: "wordpiece", label: "WordPiece", description: "BERT-style subword tokenization" },
];

const SAMPLE_TEXTS = [
  "The quick brown fox jumps over the lazy dog.",
  "Transformer architectures revolutionized natural language processing.",
  "GPT models use byte-pair encoding for subword tokenization.",
  "Attention is all you need — the foundational paper for modern LLMs.",
];

// Color palette for token visualization
const TOKEN_COLORS = [
  "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700",
  "bg-purple-100 dark:bg-purple-900/40 border-purple-300 dark:border-purple-700",
  "bg-emerald-100 dark:bg-emerald-900/40 border-emerald-300 dark:border-emerald-700",
  "bg-amber-100 dark:bg-amber-900/40 border-amber-300 dark:border-amber-700",
  "bg-rose-100 dark:bg-rose-900/40 border-rose-300 dark:border-rose-700",
  "bg-cyan-100 dark:bg-cyan-900/40 border-cyan-300 dark:border-cyan-700",
  "bg-indigo-100 dark:bg-indigo-900/40 border-indigo-300 dark:border-indigo-700",
  "bg-pink-100 dark:bg-pink-900/40 border-pink-300 dark:border-pink-700",
];

export default function TokenizationPage() {
  const [text, setText] = useState(SAMPLE_TEXTS[0]);
  const [strategy, setStrategy] = useState("character");
  const [result, setResult] = useState<any>(null);
  const [compareResult, setCompareResult] = useState<any>(null);

  const tokenize = useMutation({
    mutationFn: () => tokenizationApi.tokenize(text, strategy),
    onSuccess: (data) => setResult(data),
  });

  const compare = useMutation({
    mutationFn: () =>
      tokenizationApi.compare(text, STRATEGIES.map((s) => s.value)),
    onSuccess: (data) => setCompareResult(data),
  });

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2">Tokenization Lab</h1>
        <p className="text-muted-foreground">
          Explore how text is broken into tokens — the fundamental building
          blocks that LLMs process.
        </p>
      </div>

      {/* Input Section */}
      <div className="glass rounded-2xl p-6 space-y-4">
        <label className="text-sm font-medium">Input Text</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="w-full h-32 rounded-xl border border-border bg-background p-4 font-mono text-sm resize-none focus:ring-2 focus:ring-primary-500 focus:outline-none"
          placeholder="Enter text to tokenize..."
        />

        {/* Sample texts */}
        <div className="flex flex-wrap gap-2">
          <span className="text-xs text-muted-foreground">Try:</span>
          {SAMPLE_TEXTS.map((sample, i) => (
            <button
              key={i}
              onClick={() => setText(sample)}
              className="text-xs px-2 py-1 rounded-lg bg-muted hover:bg-muted/80 transition-colors truncate max-w-xs"
            >
              {sample.slice(0, 40)}...
            </button>
          ))}
        </div>

        {/* Strategy Selection */}
        <div className="grid grid-cols-4 gap-3">
          {STRATEGIES.map((s) => (
            <button
              key={s.value}
              onClick={() => setStrategy(s.value)}
              className={cn(
                "p-3 rounded-xl border text-left transition-all",
                strategy === s.value
                  ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20"
                  : "border-border hover:border-primary-300"
              )}
            >
              <div className="font-medium text-sm">{s.label}</div>
              <div className="text-xs text-muted-foreground">{s.description}</div>
            </button>
          ))}
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={() => tokenize.mutate()}
            disabled={tokenize.isPending || !text}
            className="px-6 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors disabled:opacity-50"
          >
            {tokenize.isPending ? "Tokenizing..." : "Tokenize"}
          </button>
          <button
            onClick={() => compare.mutate()}
            disabled={compare.isPending || !text}
            className="px-6 py-2.5 glass rounded-xl font-medium hover:bg-white/90 dark:hover:bg-gray-800/90 transition-colors disabled:opacity-50"
          >
            {compare.isPending ? "Comparing..." : "Compare All Strategies"}
          </button>
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="glass rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Tokenization Result</h2>
            <div className="flex gap-4 text-sm text-muted-foreground">
              <span>
                <strong className="text-foreground">{result.token_count}</strong> tokens
              </span>
              <span>
                Vocab: <strong className="text-foreground">{result.vocab_size}</strong>
              </span>
              <span>
                Compression:{" "}
                <strong className="text-foreground">
                  {result.compression_ratio?.toFixed(2)}x
                </strong>
              </span>
            </div>
          </div>

          {/* Token Visualization */}
          <div className="flex flex-wrap gap-1.5">
            {result.tokens?.map((token: string, i: number) => (
              <span
                key={i}
                className={cn(
                  "px-2 py-1 rounded-lg border text-sm font-mono cursor-default",
                  TOKEN_COLORS[i % TOKEN_COLORS.length]
                )}
                title={`ID: ${result.token_ids?.[i]}`}
              >
                {token.replace(/ /g, "·")}
              </span>
            ))}
          </div>

          {/* Token IDs */}
          <div>
            <h3 className="text-sm font-medium mb-2">Token IDs</h3>
            <div className="code-block">
              [{result.token_ids?.join(", ")}]
            </div>
          </div>
        </div>
      )}

      {/* Comparison Results */}
      {compareResult && (
        <div className="glass rounded-2xl p-6 space-y-4">
          <h2 className="text-xl font-semibold">Strategy Comparison</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {compareResult.comparisons?.map((comp: any) => (
              <div
                key={comp.strategy}
                className="rounded-xl border border-border p-4 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium capitalize">{comp.strategy}</span>
                  <span className="text-sm text-muted-foreground">
                    {comp.num_tokens} tokens
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {comp.tokens?.slice(0, 30).map((t: string, i: number) => (
                    <span
                      key={i}
                      className="px-1.5 py-0.5 rounded text-xs font-mono bg-muted"
                    >
                      {t.replace(/ /g, "·")}
                    </span>
                  ))}
                  {comp.tokens?.length > 30 && (
                    <span className="text-xs text-muted-foreground">
                      +{comp.tokens.length - 30} more
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Educational Content */}
      <div className="glass rounded-2xl p-6 space-y-4">
        <h2 className="text-xl font-semibold">How Tokenization Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-muted-foreground">
          <div>
            <h3 className="font-medium text-foreground mb-1">Character Tokenization</h3>
            <p>
              The simplest approach — each character becomes a token. Results in
              long sequences but a tiny vocabulary. No out-of-vocabulary issues.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-foreground mb-1">BPE (Byte Pair Encoding)</h3>
            <p>
              Used by GPT models. Iteratively merges the most frequent character
              pairs to build a subword vocabulary. Balances vocabulary size with
              sequence length.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-foreground mb-1">WordPiece</h3>
            <p>
              Used by BERT. Similar to BPE but uses likelihood-based scoring
              instead of frequency. Prefixes subwords with ## for continuation.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-foreground mb-1">Word Tokenization</h3>
            <p>
              Splits on whitespace and punctuation. Simple but creates huge
              vocabularies and cannot handle unseen words. Rarely used in modern
              LLMs.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
