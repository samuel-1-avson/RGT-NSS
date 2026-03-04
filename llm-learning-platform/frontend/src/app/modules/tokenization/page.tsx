"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Tokenization Lab</h1>
        <p className="text-muted-foreground max-w-3xl">
          Explore how text is broken into tokens — the fundamental building
          blocks that LLMs process.
        </p>
      </div>

      {/* Input Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="glass rounded-2xl p-6 space-y-4"
      >
        <label className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest">Input Text</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="w-full h-32 rounded-xl border border-white/10 bg-black/20 p-4 font-mono text-sm resize-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 transition-all outline-none text-foreground/90"
          placeholder="Enter text to tokenize..."
        />

        {/* Sample texts */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mr-2">Try:</span>
          {SAMPLE_TEXTS.map((sample, i) => (
            <button
              key={i}
              onClick={() => setText(sample)}
              className="text-xs px-2.5 py-1.5 rounded-md bg-white/5 border border-white/5 hover:bg-white/10 transition-colors truncate max-w-[200px] text-muted-foreground hover:text-foreground"
            >
              {sample.slice(0, 30)}...
            </button>
          ))}
        </div>

        {/* Strategy Selection */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-4">
          {STRATEGIES.map((s) => (
            <button
              key={s.value}
              onClick={() => setStrategy(s.value)}
              className={cn(
                "p-4 rounded-xl border text-left transition-all relative overflow-hidden group",
                strategy === s.value
                  ? "border-primary/50 bg-primary/5"
                  : "border-white/10 hover:border-white/20 hover:bg-white/5"
              )}
            >
              {strategy === s.value && (
                <motion.div
                  layoutId="activeStrategy"
                  className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-50"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                />
              )}
              <div className="font-semibold text-[15px] mb-1 text-foreground/90 relative z-10">{s.label}</div>
              <div className="text-xs text-muted-foreground leading-relaxed relative z-10">{s.description}</div>
            </button>
          ))}
        </div>

        {/* Actions */}
        <div className="flex gap-3 pt-2">
          <button
            onClick={() => tokenize.mutate()}
            disabled={tokenize.isPending || !text}
            className="px-6 py-2.5 bg-primary hover:bg-primary/90 text-primary-foreground rounded-xl font-medium transition-all shadow-lg shadow-primary/20 disabled:opacity-50 disabled:shadow-none text-sm"
          >
            {tokenize.isPending ? "Tokenizing..." : "Tokenize Sequence"}
          </button>
          <button
            onClick={() => compare.mutate()}
            disabled={compare.isPending || !text}
            className="px-6 py-2.5 glass rounded-xl font-medium transition-all disabled:opacity-50 hover:bg-white/10 text-sm"
          >
            {compare.isPending ? "Comparing..." : "Compare Strategies"}
          </button>
        </div>
      </motion.div>

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass rounded-2xl p-6 space-y-6 border-primary/20"
        >
          <div className="flex items-center justify-between pb-4 border-b border-white/10">
            <h2 className="text-lg font-semibold text-primary">Tokenization Result</h2>
            <div className="flex gap-6 text-sm">
              <div className="flex flex-col items-end">
                <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">Tokens</span>
                <span className="font-mono text-foreground text-lg">{result.token_count}</span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">Vocab Size</span>
                <span className="font-mono text-foreground text-lg">{result.vocab_size}</span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">Compression</span>
                <span className="font-mono text-accent text-lg">{result.compression_ratio?.toFixed(2)}x</span>
              </div>
            </div>
          </div>

          {/* Token Visualization */}
          <div className="flex flex-wrap gap-1.5 p-4 bg-black/20 rounded-xl border border-white/5">
            {result.tokens?.map((token: string, i: number) => (
              <motion.span
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.01 }}
                key={i}
                className={cn(
                  "px-2 py-1 rounded-md border text-sm font-mono cursor-default shadow-sm",
                  TOKEN_COLORS[i % TOKEN_COLORS.length]
                )}
                title={`ID: ${result.token_ids?.[i]}`}
              >
                {token.replace(/ /g, "·")}
              </motion.span>
            ))}
          </div>

          {/* Token IDs */}
          <div>
            <h3 className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Token IDs Sequence</h3>
            <div className="code-block bg-black/40 border-white/10 text-muted-foreground">
              [{result.token_ids?.join(", ")}]
            </div>
          </div>
        </motion.div>
      )}

      {/* Comparison Results */}
      {compareResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-2xl p-6 space-y-4"
        >
          <h2 className="text-lg font-semibold text-foreground/90">Strategy Comparison</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {compareResult.comparisons?.map((comp: any) => (
              <div
                key={comp.strategy}
                className="rounded-xl border border-white/10 bg-black/10 p-5 space-y-3"
              >
                <div className="flex items-center justify-between pb-2 border-b border-white/5">
                  <span className="font-semibold text-primary capitalize">{comp.strategy}</span>
                  <span className="text-xs font-mono text-muted-foreground bg-white/5 px-2 py-0.5 rounded">
                    {comp.num_tokens} tokens
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {comp.tokens?.slice(0, 30).map((t: string, i: number) => (
                    <span
                      key={i}
                      className="px-1.5 py-0.5 rounded textxs font-mono bg-white/5 text-foreground/80 border border-white/5"
                    >
                      {t.replace(/ /g, "·")}
                    </span>
                  ))}
                  {comp.tokens?.length > 30 && (
                    <span className="text-xs text-muted-foreground self-center ml-2 italic">
                      +{comp.tokens.length - 30} more
                    </span>
                  )}
                </div>
              </div>
            ))}
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
          <h3 className="font-semibold text-primary mb-2 text-sm">Character Tokenization</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            The simplest approach — each character becomes a token. Results in
            long sequences but a tiny vocabulary. Eliminates unknown word (OOV) tokens completely.
          </p>
        </div>

        <div className="p-5 rounded-2xl border-l-2 border-accent bg-accent/5">
          <h3 className="font-semibold text-accent mb-2 text-sm">BPE (Byte Pair Encoding)</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Used by GPT frameworks. Iteratively merges the most frequent character
            pairs to build a subword vocabulary, optimizing the balance between vocabulary size and sequence length.
          </p>
        </div>

        <div className="p-5 rounded-2xl border-l-2 border-orange-500/50 bg-orange-500/5">
          <h3 className="font-semibold text-orange-400 mb-2 text-sm">WordPiece / BERT</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Standard format for BERT. Similar to BPE but uses likelihood-based scoring
            instead of pure frequency. Prefixes subwords with `##` to denote continuation.
          </p>
        </div>

        <div className="p-5 rounded-2xl border-l-2 border-emerald-500/50 bg-emerald-500/5">
          <h3 className="font-semibold text-emerald-400 mb-2 text-sm">Word Tokenization</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Splits strictly on whitespace and punctuation markers. While human-readable, it creates unmanageably huge
            vocabularies and lacks the ability to handle unseen words. Rarely used in modern LLMs.
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}
