"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, Terminal, Play, Info } from "lucide-react";
import Link from "next/link";

// Simple tokenization demo
const tokenizeText = (text: string): string[] => {
  // Simple whitespace and punctuation tokenization
  return text
    .replace(/([.,!?;:])/g, " $1 ")
    .split(/\s+/)
    .filter(t => t.length > 0);
};

export default function TokenizationPage() {
  const [inputText, setInputText] = useState("Hello, world! This is tokenization.");
  const [tokens, setTokens] = useState<string[]>(["Hello", ",", "world", "!", "This", "is", "tokenization", "."]);

  const handleTokenize = () => {
    setTokens(tokenizeText(inputText));
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
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 text-emerald-300 text-sm font-medium mb-6">
              <Terminal className="w-4 h-4" />
              <span>Tokenization</span>
            </div>
            <h1 className="text-5xl font-bold mb-6">
              From Text to <span className="text-emerald-400">Tokens</span>
            </h1>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto">
              Learn how language models break down text into pieces they can understand.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Interactive Demo */}
      <section className="py-20 border-y border-slate-800/50">
        <div className="max-w-4xl mx-auto px-6">
          <div className="bg-slate-900/50 rounded-3xl p-8 border border-slate-700/50">
            <h2 className="text-2xl font-bold mb-6">Try It Out</h2>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm text-slate-400 mb-2">Input Text</label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="w-full p-4 bg-slate-800 rounded-xl text-white resize-none focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  rows={3}
                />
              </div>

              <button
                onClick={handleTokenize}
                className="flex items-center gap-2 px-6 py-3 bg-emerald-600 text-white rounded-xl font-medium hover:bg-emerald-500 transition-colors"
              >
                <Play className="w-4 h-4" />
                Tokenize
              </button>

              <div>
                <label className="block text-sm text-slate-400 mb-2">Tokens ({tokens.length})</label>
                <div className="flex flex-wrap gap-2 p-4 bg-slate-800 rounded-xl">
                  {tokens.map((token, i) => (
                    <motion.span
                      key={i}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.05 }}
                      className="px-3 py-1.5 bg-emerald-500/20 text-emerald-300 rounded-lg font-mono text-sm"
                    >
                      {token}
                    </motion.span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Content Sections */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-6 space-y-12">
          <div className="flex gap-6">
            <div className="w-12 h-12 rounded-xl bg-emerald-500/20 flex items-center justify-center shrink-0">
              <Info className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-white mb-2">What is Tokenization?</h3>
              <p className="text-slate-400 leading-relaxed">
                Tokenization is the process of breaking down text into smaller pieces called tokens. 
                These tokens can be words, subwords, or even characters. Language models process 
                tokens rather than raw text, making tokenization a crucial first step.
              </p>
            </div>
          </div>

          <div className="flex gap-6">
            <div className="w-12 h-12 rounded-xl bg-violet-500/20 flex items-center justify-center shrink-0">
              <span className="text-2xl">🔤</span>
            </div>
            <div>
              <h3 className="text-xl font-bold text-white mb-2">Types of Tokenization</h3>
              <div className="space-y-3 text-slate-400">
                <p><strong className="text-white">Word-based:</strong> Split by spaces and punctuation. Simple but creates large vocabularies.</p>
                <p><strong className="text-white">Subword (BPE):</strong> Breaks words into common subwords. Used by GPT-2, BERT.</p>
                <p><strong className="text-white">Character-based:</strong> Split into individual characters. Small vocabulary but long sequences.</p>
              </div>
            </div>
          </div>

          <div className="p-6 bg-slate-900/50 rounded-2xl border border-slate-700/50">
            <h3 className="text-lg font-bold text-white mb-4">Example: BPE Tokenization</h3>
            <div className="space-y-2 font-mono text-sm">
              <div className="flex gap-4">
                <span className="text-slate-500">Input:</span>
                <span className="text-white">"unhappiness"</span>
              </div>
              <div className="flex gap-4">
                <span className="text-slate-500">Tokens:</span>
                <span className="text-emerald-400">["un", "happiness"]</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Navigation */}
      <section className="py-12 border-t border-slate-800/50">
        <div className="max-w-4xl mx-auto px-6 flex justify-between">
          <Link
            href="/learn/"
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>All Modules</span>
          </Link>
          <Link
            href="/learn/transformer/"
            className="flex items-center gap-2 text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            <span>Next: Transformer</span>
            <ArrowLeft className="w-5 h-5 rotate-180" />
          </Link>
        </div>
      </section>
    </main>
  );
}
