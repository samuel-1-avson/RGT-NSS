"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, Eye, Play, Info } from "lucide-react";
import Link from "next/link";

export default function AttentionPage() {
  const [query, setQuery] = useState("The");
  const [sentence, setSentence] = useState(["The", "cat", "sat", "on", "the", "mat"]);
  const [attentionWeights, setAttentionWeights] = useState([0.3, 0.15, 0.25, 0.1, 0.12, 0.08]);

  const computeAttention = (qIdx: number) => {
    // Simulate attention based on position proximity
    const weights = sentence.map((_, i) => {
      const proximity = Math.exp(-Math.abs(i - qIdx) / 2);
      const random = Math.random() * 0.2;
      return proximity + random;
    });
    const sum = weights.reduce((a, b) => a + b, 0);
    return weights.map(w => w / sum);
  };

  const handleCompute = () => {
    const qIdx = sentence.indexOf(query) || 0;
    setAttentionWeights(computeAttention(qIdx));
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-slate-800/50">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-4">
            <Link
              href="/learn/"
              className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back to Learning</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Hero */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 text-amber-300 text-sm font-medium mb-6">
              <Eye className="w-4 h-4" />
              <span>Attention Mechanism</span>
            </div>
            <h1 className="text-5xl font-bold mb-6">
              The Power of <span className="text-amber-400">Attention</span>
            </h1>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto">
              Discover how transformers focus on what matters when processing language.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Interactive Demo */}
      <section className="py-20 border-y border-slate-800/50">
        <div className="max-w-4xl mx-auto px-6">
          <div className="bg-slate-900/50 rounded-3xl p-8 border border-slate-700/50">
            <h2 className="text-2xl font-bold mb-6">Attention Visualizer</h2>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm text-slate-400 mb-2">Select Query Word</label>
                <select
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="w-full p-3 bg-slate-800 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
                >
                  {sentence.map((word) => (
                    <option key={word} value={word}>{word}</option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleCompute}
                className="flex items-center gap-2 px-6 py-3 bg-amber-600 text-white rounded-xl font-medium hover:bg-amber-500 transition-colors"
              >
                <Play className="w-4 h-4" />
                Compute Attention
              </button>

              <div>
                <label className="block text-sm text-slate-400 mb-3">
                  Attention Weights for "{query}":
                </label>
                <div className="space-y-3">
                  {sentence.map((word, i) => (
                    <div key={word} className="flex items-center gap-4">
                      <span className="w-16 text-slate-300 font-medium">{word}</span>
                      <div className="flex-1 h-8 bg-slate-800 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${attentionWeights[i] * 100}%` }}
                          transition={{ duration: 0.5, delay: i * 0.05 }}
                          className="h-full bg-gradient-to-r from-amber-500 to-orange-500"
                        />
                      </div>
                      <span className="w-16 text-right text-amber-400 font-mono">
                        {(attentionWeights[i] * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Content */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-6 space-y-12">
          <div className="flex gap-6">
            <div className="w-12 h-12 rounded-xl bg-amber-500/20 flex items-center justify-center shrink-0">
              <Info className="w-6 h-6 text-amber-400" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-white mb-2">What is Attention?</h3>
              <p className="text-slate-400 leading-relaxed">
                Attention is a mechanism that allows models to focus on different parts of the input 
                when producing each part of the output. It computes a weighted sum of all input positions, 
                where the weights determine how much each position contributes.
              </p>
            </div>
          </div>

          <div className="p-6 bg-slate-900/50 rounded-2xl border border-slate-700/50">
            <h3 className="text-lg font-bold text-white mb-4">Self-Attention Formula</h3>
            <div className="p-4 bg-slate-950 rounded-xl font-mono text-center text-lg">
              <span className="text-amber-400">Attention</span>
              <span className="text-slate-400">(Q, K, V) = </span>
              <span className="text-slate-300">softmax</span>
              <span className="text-slate-400">(</span>
              <div className="inline-flex flex-col items-center mx-1">
                <span className="text-violet-400 border-b border-slate-600 px-2">QK<sup>T</sup></span>
                <span className="text-slate-500 text-sm">√d<sub>k</sub></span>
              </div>
              <span className="text-slate-400">)</span>
              <span className="text-emerald-400">V</span>
            </div>
          </div>

          <div className="flex gap-6">
            <div className="w-12 h-12 rounded-xl bg-violet-500/20 flex items-center justify-center shrink-0">
              <span className="text-2xl">🎯</span>
            </div>
            <div>
              <h3 className="text-xl font-bold text-white mb-2">Why It Matters</h3>
              <ul className="space-y-2 text-slate-400">
                <li>• Captures long-range dependencies in text</li>
                <li>• Parallel computation (unlike RNNs)</li>
                <li>• Interpretable - we can visualize what the model focuses on</li>
                <li>• Foundation of modern NLP (BERT, GPT, T5)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Navigation */}
      <section className="py-12 border-t border-slate-800/50">
        <div className="max-w-4xl mx-auto px-6 flex justify-between">
          <Link
            href="/learn/tokenization/"
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Previous: Tokenization</span>
          </Link>
          <Link
            href="/learn/transformer/"
            className="flex items-center gap-2 text-amber-400 hover:text-amber-300 transition-colors"
          >
            <span>Next: Transformer</span>
            <ArrowLeft className="w-5 h-5 rotate-180" />
          </Link>
        </div>
      </section>
    </main>
  );
}
