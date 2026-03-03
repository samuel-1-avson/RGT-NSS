'use client';

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Type, Split, Hash, Play, BookOpen, Loader2,
  ArrowRight, Layers, BarChart3, GitMerge, Sparkles, Info
} from 'lucide-react';
import Link from 'next/link';
import ModuleNavBar from '@/components/ModuleNavBar';
import { api } from '@/utils/api';
import toast from 'react-hot-toast';

interface MergeStep {
  step: number;
  pair: [string, string];
  merged: string;
  frequency: number;
}

const TOKEN_COLORS = [
  'bg-blue-100 text-blue-800 ring-blue-200',
  'bg-purple-100 text-purple-800 ring-purple-200',
  'bg-amber-100 text-amber-800 ring-amber-200',
  'bg-green-100 text-green-800 ring-green-200',
  'bg-pink-100 text-pink-800 ring-pink-200',
  'bg-cyan-100 text-cyan-800 ring-cyan-200',
  'bg-rose-100 text-rose-800 ring-rose-200',
  'bg-emerald-100 text-emerald-800 ring-emerald-200',
];

export default function TokenizationPage() {
  const [inputText, setInputText] = useState('The transformer model learns to generate text by predicting the next token.');
  const [isProcessing, setIsProcessing] = useState(false);

  // Results for all 3 strategies
  const [charResult, setCharResult] = useState<any>(null);
  const [wordResult, setWordResult] = useState<any>(null);
  const [bpeResult, setBpeResult] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'compare' | 'bpe-merges'>('compare');

  // BPE controls
  const [numMerges, setNumMerges] = useState(30);
  const [highlightedMerge, setHighlightedMerge] = useState<number | null>(null);

  const handleTokenize = useCallback(async () => {
    if (!inputText.trim()) {
      toast.error('Please enter some text');
      return;
    }

    setIsProcessing(true);
    try {
      // Run all three strategies in parallel
      const [charData, wordData, bpeData] = await Promise.all([
        api.tokenize({ text: inputText, strategy: 'character' }),
        api.tokenize({ text: inputText, strategy: 'word' }),
        api.computeBpeTokenize(inputText, numMerges),
      ]);
      setCharResult(charData);
      setWordResult(wordData);
      setBpeResult(bpeData);
    } catch (error) {
      toast.error('Failed to tokenize');
      console.error(error);
    } finally {
      setIsProcessing(false);
    }
  }, [inputText, numMerges]);

  const displayToken = (token: string) => token === ' ' ? '␣' : token;

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/20 to-indigo-50/20">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-50">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/learn" className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors">
                <ArrowLeft className="h-5 w-5" />
                <span className="text-sm font-medium">Back to Learn</span>
              </Link>
              <div className="h-6 w-px bg-slate-300" />
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                  <Type className="h-4 w-4 text-white" />
                </div>
                <h1 className="text-lg font-semibold text-slate-900">Tokenization Lab</h1>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-500">Module 1 of 6</span>
              <span className="text-slate-300">|</span>
              <Link href="/learn/embeddings" className="text-sm text-slate-500 hover:text-slate-700">Embeddings →</Link>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Theory Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-8 mb-8"
        >
          <div className="flex items-center gap-3 mb-4">
            <BookOpen className="h-5 w-5 text-blue-500" />
            <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Understanding Tokenization
            </h2>
          </div>
          <p className="text-slate-600 leading-relaxed mb-5">
            Tokenization converts raw text into a sequence of tokens that a language model can process.
            The choice of tokenization strategy significantly affects vocabulary size, sequence length,
            and the model&apos;s ability to handle rare words. Modern LLMs use <strong>BPE (Byte-Pair Encoding)</strong>,
            which iteratively merges the most frequent character pairs.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { icon: Type, title: 'Character', desc: 'Each character = 1 token. Tiny vocab (~100) but long sequences.', color: 'text-cyan-500', bg: 'bg-cyan-50' },
              { icon: Split, title: 'Word', desc: 'Each word = 1 token. Huge vocab, can\'t handle unknown words.', color: 'text-amber-500', bg: 'bg-amber-50' },
              { icon: Hash, title: 'BPE (Subword)', desc: 'Iteratively merges frequent pairs. Best of both worlds.', color: 'text-indigo-500', bg: 'bg-indigo-50' },
            ].map((item) => (
              <div key={item.title} className={`${item.bg} rounded-xl p-4 ring-1 ring-slate-100`}>
                <item.icon className={`h-5 w-5 ${item.color} mb-2`} />
                <h4 className="font-semibold text-slate-800 text-sm mb-1">{item.title}</h4>
                <p className="text-xs text-slate-500">{item.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Input Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6 mb-6"
        >
          <h3 className="font-semibold text-slate-900 mb-3">Input Text</h3>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            className="w-full h-24 px-4 py-3 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none text-sm"
            placeholder="Enter text to tokenize..."
          />
          <div className="flex items-center gap-4 mt-3">
            <button
              onClick={handleTokenize}
              disabled={isProcessing}
              className="flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-medium hover:from-blue-500 hover:to-indigo-500 disabled:opacity-50 transition-all shadow-lg shadow-blue-200"
            >
              {isProcessing ? (
                <><Loader2 className="h-4 w-4 animate-spin" /> Processing...</>
              ) : (
                <><Play className="h-4 w-4" /> Tokenize All Strategies</>
              )}
            </button>
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <label className="whitespace-nowrap">BPE Merges:</label>
              <input
                type="range"
                min="5"
                max="100"
                value={numMerges}
                onChange={(e) => setNumMerges(parseInt(e.target.value))}
                className="w-32"
              />
              <span className="font-mono text-xs text-slate-700 min-w-[24px]">{numMerges}</span>
            </div>
          </div>
        </motion.div>

        {/* Results Tabs */}
        {(charResult || wordResult || bpeResult) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="flex gap-2">
              <button
                onClick={() => setActiveTab('compare')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${activeTab === 'compare'
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-200'
                    : 'bg-white text-slate-600 hover:bg-slate-50 ring-1 ring-slate-200'
                  }`}
              >
                <BarChart3 className="h-4 w-4" /> Side-by-Side Comparison
              </button>
              <button
                onClick={() => setActiveTab('bpe-merges')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${activeTab === 'bpe-merges'
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-200'
                    : 'bg-white text-slate-600 hover:bg-slate-50 ring-1 ring-slate-200'
                  }`}
              >
                <GitMerge className="h-4 w-4" /> BPE Merge Visualization
              </button>
            </div>

            <AnimatePresence mode="wait">
              {activeTab === 'compare' && (
                <motion.div
                  key="compare"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  {/* Summary Stats Bar */}
                  <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-5 mb-6">
                    <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-amber-500" />
                      Strategy Comparison
                    </h3>
                    <div className="grid grid-cols-3 gap-4">
                      {[
                        { name: 'Character', data: charResult, gradient: 'from-cyan-500 to-blue-500' },
                        { name: 'Word', data: wordResult, gradient: 'from-amber-500 to-orange-500' },
                        { name: 'BPE', data: bpeResult, gradient: 'from-indigo-500 to-purple-500' },
                      ].map((strategy) => (
                        <div key={strategy.name} className="text-center">
                          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gradient-to-r ${strategy.gradient} text-white text-xs font-medium mb-3`}>
                            {strategy.name}
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <div className="bg-slate-50 rounded-lg p-2">
                              <p className="text-lg font-bold text-slate-800">{strategy.data?.num_tokens || '—'}</p>
                              <p className="text-[10px] text-slate-500">Tokens</p>
                            </div>
                            <div className="bg-slate-50 rounded-lg p-2">
                              <p className="text-lg font-bold text-slate-800">{strategy.data?.vocab_size || strategy.data?.vocabulary?.length || '—'}</p>
                              <p className="text-[10px] text-slate-500">Vocab</p>
                            </div>
                          </div>
                          {strategy.data?.compression_ratio && (
                            <p className="text-[10px] text-slate-400 mt-2">{strategy.data.compression_ratio}× compression</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Side-by-Side Token Display */}
                  <div className="grid gap-6 md:grid-cols-3">
                    {[
                      { name: 'Character', data: charResult, gradient: 'from-cyan-500 to-blue-500', colors: TOKEN_COLORS[5] },
                      { name: 'Word', data: wordResult, gradient: 'from-amber-500 to-orange-500', colors: TOKEN_COLORS[2] },
                      { name: 'BPE (Subword)', data: bpeResult, gradient: 'from-indigo-500 to-purple-500', colors: TOKEN_COLORS[1] },
                    ].map((strategy) => (
                      <div key={strategy.name} className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 overflow-hidden">
                        <div className={`h-1 bg-gradient-to-r ${strategy.gradient}`} />
                        <div className="p-5">
                          <h4 className="font-semibold text-slate-800 text-sm mb-3">{strategy.name}</h4>

                          {/* Tokens */}
                          <div className="mb-4">
                            <p className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider mb-2">Tokens</p>
                            <div className="flex flex-wrap gap-1.5">
                              {strategy.data?.tokens?.map((token: string, i: number) => (
                                <motion.span
                                  key={i}
                                  initial={{ opacity: 0, scale: 0.8 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  transition={{ delay: i * 0.01 }}
                                  className={`px-2 py-1 rounded-lg text-xs font-medium ring-1 ${strategy.colors}`}
                                >
                                  {displayToken(token)}
                                </motion.span>
                              ))}
                            </div>
                          </div>

                          {/* Token IDs */}
                          <div>
                            <p className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider mb-2">Token IDs</p>
                            <div className="flex flex-wrap gap-1">
                              {strategy.data?.token_ids?.map((id: number, i: number) => (
                                <span key={i} className="px-1.5 py-0.5 bg-slate-100 rounded text-[10px] font-mono text-slate-600">
                                  {id}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}

              {activeTab === 'bpe-merges' && bpeResult?.merge_history && (
                <motion.div
                  key="bpe-merges"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="space-y-6"
                >
                  {/* BPE Explanation */}
                  <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6">
                    <div className="flex items-start gap-3">
                      <Info className="h-5 w-5 text-indigo-500 mt-0.5" />
                      <div>
                        <h3 className="font-semibold text-slate-900">How BPE Works</h3>
                        <p className="text-sm text-slate-500 mt-1">
                          BPE starts with individual characters, then <strong>iteratively merges the most frequent adjacent pair</strong> into a new token.
                          This process continues for a fixed number of merges. Hover over a merge step below to see it highlighted.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Merge Steps Timeline */}
                  <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 overflow-hidden">
                    <div className="p-5 border-b border-slate-100">
                      <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                        <GitMerge className="h-5 w-5 text-indigo-500" />
                        Merge History ({bpeResult.merge_history.length} merges)
                      </h3>
                      <p className="text-xs text-slate-500 mt-1">
                        Started with {inputText.length} characters → compressed to {bpeResult.num_tokens} tokens ({bpeResult.compression_ratio}× compression)
                      </p>
                    </div>

                    <div className="divide-y divide-slate-50 max-h-[500px] overflow-y-auto">
                      {bpeResult.merge_history.map((merge: MergeStep) => (
                        <motion.div
                          key={merge.step}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: merge.step * 0.02 }}
                          onMouseEnter={() => setHighlightedMerge(merge.step)}
                          onMouseLeave={() => setHighlightedMerge(null)}
                          className={`flex items-center gap-4 px-5 py-3 transition-colors ${highlightedMerge === merge.step ? 'bg-indigo-50' : 'hover:bg-slate-50'
                            }`}
                        >
                          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                            {merge.step}
                          </div>
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            <span className="px-2 py-1 bg-red-50 text-red-700 rounded-md text-sm font-mono ring-1 ring-red-100">
                              &quot;{displayToken(merge.pair[0])}&quot;
                            </span>
                            <span className="text-slate-400 text-xs">+</span>
                            <span className="px-2 py-1 bg-red-50 text-red-700 rounded-md text-sm font-mono ring-1 ring-red-100">
                              &quot;{displayToken(merge.pair[1])}&quot;
                            </span>
                            <ArrowRight className="h-4 w-4 text-indigo-400 flex-shrink-0" />
                            <span className="px-2 py-1 bg-green-50 text-green-700 rounded-md text-sm font-mono font-bold ring-1 ring-green-100">
                              &quot;{displayToken(merge.merged)}&quot;
                            </span>
                          </div>
                          <div className="text-right flex-shrink-0">
                            <p className="text-xs font-semibold text-indigo-600">×{merge.frequency}</p>
                            <p className="text-[10px] text-slate-400">occurrences</p>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>

                  {/* Final BPE Result */}
                  <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6">
                    <h3 className="font-semibold text-slate-900 mb-4">Final BPE Tokens</h3>
                    <div className="flex flex-wrap gap-2">
                      {bpeResult.tokens?.map((token: string, i: number) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: i * 0.02 }}
                          className={`px-3 py-1.5 rounded-lg text-sm font-medium ring-1 ${TOKEN_COLORS[i % TOKEN_COLORS.length]
                            }`}
                        >
                          <span className="font-mono">{displayToken(token)}</span>
                          <span className="ml-1.5 text-[10px] opacity-60">id:{bpeResult.token_ids[i]}</span>
                        </motion.div>
                      ))}
                    </div>

                    {/* Vocabulary */}
                    {bpeResult.vocabulary?.length <= 80 && (
                      <div className="mt-5 pt-5 border-t border-slate-100">
                        <p className="text-xs font-semibold text-slate-500 mb-2">
                          BPE Vocabulary ({bpeResult.vocab_size} tokens)
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {bpeResult.vocabulary.map((v: string, i: number) => (
                            <span key={i} className="px-1.5 py-0.5 bg-slate-100 rounded text-[10px] font-mono text-slate-600">
                              {displayToken(v)}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}

        {/* Empty state */}
        {!charResult && !wordResult && !bpeResult && (
          <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-12 text-center">
            <Type className="h-12 w-12 text-slate-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-slate-900 mb-2">Ready to Tokenize</h3>
            <p className="text-slate-500 text-sm max-w-md mx-auto">
              Enter text above and click &quot;Tokenize All Strategies&quot; to see how Character, Word, and BPE tokenization compare side-by-side.
            </p>
          </div>
        )}
      </div>
      <ModuleNavBar />
    </main>
  );
}
