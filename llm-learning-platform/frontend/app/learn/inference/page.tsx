'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Wand2, Play, Settings,
  Loader2, BarChart3, Sparkles, Thermometer
} from 'lucide-react';
import Link from 'next/link';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api } from '@/utils/api';
import { useModelStore } from '@/stores/modelStore';
import toast from 'react-hot-toast';
import ModuleNavBar from '@/components/ModuleNavBar';

export default function InferencePage() {
  const { currentModel } = useModelStore();

  // Input state
  const [prompt, setPrompt] = useState('To be or not to be');
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(50);

  // Output state
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  // Sampling visualization
  const [samplingData, setSamplingData] = useState<any>(null);
  const [loadingSampling, setLoadingSampling] = useState(false);

  // Compute sampling distribution
  const computeSampling = useCallback(async () => {
    setLoadingSampling(true);
    try {
      const data = await api.computeSampling({
        text: prompt,
        temperature,
        top_k: topK,
        top_p: topP,
        vocab_size: 128,
      });
      setSamplingData(data);
    } catch (err) {
      console.error('Sampling compute error:', err);
    } finally {
      setLoadingSampling(false);
    }
  }, [prompt, temperature, topK, topP]);

  // Auto-compute sampling when params change
  useEffect(() => {
    const timer = setTimeout(() => {
      computeSampling();
    }, 300);
    return () => clearTimeout(timer);
  }, [temperature, topK, topP, prompt]);

  // Generate text using backend
  const generateText = async () => {
    if (!currentModel) {
      toast.error('Create a model first from the Training Dashboard (/train)');
      return;
    }
    setIsGenerating(true);
    setGeneratedText('');
    try {
      const response = await api.generate({
        model_id: currentModel.model_id,
        prompt,
        max_new_tokens: maxTokens,
        temperature,
        top_k: topK,
        top_p: topP,
        repetition_penalty: 1.0,
      });
      setGeneratedText(response.generated_text);
      toast.success(`Generated ${response.tokens_generated} tokens`);
    } catch (err: any) {
      toast.error(err.message || 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  // Prepare chart data
  const originalDistData = (samplingData?.original_distribution || []).slice(0, 20).map((d: any) => ({
    token: d.token,
    probability: d.probability,
  }));

  const topKData = (samplingData?.top_k_distribution || []).slice(0, 15).map((d: any) => ({
    token: d.token,
    probability: d.probability,
  }));

  const nucleusData = (samplingData?.nucleus_distribution || []).slice(0, 15).map((d: any) => ({
    token: d.token,
    probability: d.probability,
  }));

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-violet-50/20 to-pink-50/20">
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
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-violet-500 to-pink-600 flex items-center justify-center">
                  <Wand2 className="h-4 w-4 text-white" />
                </div>
                <h1 className="text-lg font-semibold text-slate-900">Inference Playground</h1>
              </div>
            </div>
            <Link href="/learn/training" className="text-sm text-slate-500 hover:text-slate-700">← Training</Link>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Theory */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-8 mb-8"
        >
          <h2 className="text-2xl font-bold bg-gradient-to-r from-violet-600 to-pink-600 bg-clip-text text-transparent mb-4">
            Text Generation & Sampling Strategies
          </h2>
          <p className="text-slate-600 leading-relaxed mb-4">
            During inference, the model predicts a probability distribution over the vocabulary for the next token.
            How we <strong>sample</strong> from this distribution determines the quality and diversity of generated text.
            Temperature, Top-K, and Top-P are the key parameters that control this process.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { title: 'Temperature', desc: 'Controls randomness. Lower = more deterministic, Higher = more creative', color: 'from-red-500 to-orange-500' },
              { title: 'Top-K Sampling', desc: 'Only sample from the K most probable tokens at each step', color: 'from-blue-500 to-cyan-500' },
              { title: 'Nucleus (Top-P)', desc: 'Sample from the smallest set whose cumulative probability exceeds P', color: 'from-green-500 to-emerald-500' },
            ].map((item) => (
              <div key={item.title} className="bg-slate-50 rounded-xl p-4">
                <h4 className={`font-semibold text-transparent bg-gradient-to-r ${item.color} bg-clip-text text-sm mb-1`}>{item.title}</h4>
                <p className="text-slate-500 text-xs">{item.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* Left Panel - Controls */}
          <div className="space-y-6">
            {/* Prompt Input */}
            <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6">
              <h3 className="text-sm font-semibold text-slate-900 mb-3">Prompt</h3>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter a prompt..."
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-violet-500 resize-none font-mono"
                rows={4}
              />
              <button
                onClick={generateText}
                disabled={isGenerating || !prompt.trim()}
                className="w-full mt-3 px-4 py-3 bg-gradient-to-r from-violet-600 to-pink-600 text-white text-sm font-medium rounded-xl hover:from-violet-700 hover:to-pink-700 disabled:opacity-50 transition-all flex items-center justify-center gap-2 shadow-lg shadow-violet-200"
              >
                {isGenerating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                {isGenerating ? 'Generating...' : 'Generate Text'}
              </button>
              {!currentModel && (
                <p className="text-xs text-amber-600 mt-2 text-center">
                  ⚠️ Create a model first at <Link href="/train" className="underline font-medium">/train</Link>
                </p>
              )}
            </div>

            {/* Generation Parameters */}
            <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6">
              <h3 className="text-sm font-semibold text-slate-900 mb-4 flex items-center gap-2">
                <Settings className="h-4 w-4 text-violet-500" />
                Sampling Parameters
              </h3>
              <div className="space-y-5">
                <div>
                  <div className="flex justify-between mb-1">
                    <label className="text-xs font-medium text-slate-600 flex items-center gap-1">
                      <Thermometer className="h-3 w-3 text-red-400" /> Temperature
                    </label>
                    <span className="text-xs font-mono text-slate-400">{temperature.toFixed(2)}</span>
                  </div>
                  <input type="range" min={0.1} max={3.0} step={0.05} value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))} className="w-full" />
                  <div className="flex justify-between text-xs text-slate-400 mt-0.5">
                    <span>Focused</span><span>Creative</span>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <label className="text-xs font-medium text-slate-600">Top-K</label>
                    <span className="text-xs font-mono text-slate-400">{topK}</span>
                  </div>
                  <input type="range" min={1} max={128} step={1} value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))} className="w-full" />
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <label className="text-xs font-medium text-slate-600">Top-P (Nucleus)</label>
                    <span className="text-xs font-mono text-slate-400">{topP.toFixed(2)}</span>
                  </div>
                  <input type="range" min={0.1} max={1.0} step={0.05} value={topP}
                    onChange={(e) => setTopP(parseFloat(e.target.value))} className="w-full" />
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <label className="text-xs font-medium text-slate-600">Max Tokens</label>
                    <span className="text-xs font-mono text-slate-400">{maxTokens}</span>
                  </div>
                  <input type="range" min={10} max={200} step={10} value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value))} className="w-full" />
                </div>
              </div>
            </div>

            {/* Sampling Stats */}
            {samplingData && (
              <div className="bg-gradient-to-br from-violet-50 to-pink-50 rounded-2xl ring-1 ring-violet-100 p-5">
                <h3 className="text-sm font-semibold text-slate-800 mb-3">Distribution Stats</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500">Entropy</span>
                    <span className="text-xs font-mono font-bold text-violet-700">{samplingData.entropy?.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500">Top-1 Probability</span>
                    <span className="text-xs font-mono font-bold text-violet-700">{(samplingData.top_1_prob * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500">Top-5 Cumulative</span>
                    <span className="text-xs font-mono font-bold text-violet-700">{(samplingData.top_5_cumulative * 100)?.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Main Area */}
          <div className="lg:col-span-2 space-y-6">
            {/* Generated Output */}
            {(generatedText || isGenerating) && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6"
              >
                <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-violet-500" />
                  Generated Text
                </h3>
                <div className="bg-slate-50 rounded-xl p-4 font-mono text-sm">
                  <span className="text-slate-500">{prompt}</span>
                  <span className="text-violet-700 font-semibold">{generatedText}</span>
                  {isGenerating && <span className="inline-block w-2 h-4 bg-violet-500 animate-pulse ml-0.5" />}
                </div>
              </motion.div>
            )}

            {/* Original Distribution */}
            {samplingData && (
              <>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="font-semibold text-slate-900">
                        Token Probability Distribution
                      </h3>
                      <p className="text-sm text-slate-500">
                        After applying temperature={temperature.toFixed(2)} — Top 20 tokens shown
                      </p>
                    </div>
                    {loadingSampling && <Loader2 className="h-4 w-4 text-violet-500 animate-spin" />}
                  </div>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={originalDistData} margin={{ bottom: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="token" angle={-45} textAnchor="end" fontSize={10} stroke="#94a3b8" interval={0} height={50} />
                        <YAxis stroke="#94a3b8" fontSize={10} />
                        <Tooltip
                          contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '12px', fontSize: '12px' }}
                          formatter={(value: number) => [(value * 100).toFixed(2) + '%', 'Probability']}
                        />
                        <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                          {originalDistData.map((_: any, i: number) => (
                            <Cell key={i} fill={i === 0 ? '#8b5cf6' : i < 5 ? '#a78bfa' : '#c4b5fd'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </motion.div>

                {/* Comparison: Top-K vs Nucleus */}
                <div className="grid md:grid-cols-2 gap-6">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6"
                  >
                    <h3 className="font-semibold text-slate-900 mb-1">Top-K Filtered (K={topK})</h3>
                    <p className="text-xs text-slate-500 mb-4">Only the {topK} most probable tokens remain</p>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={topKData} margin={{ bottom: 30 }}>
                          <XAxis dataKey="token" angle={-45} textAnchor="end" fontSize={9} stroke="#94a3b8" interval={0} height={40} />
                          <YAxis stroke="#94a3b8" fontSize={9} />
                          <Tooltip formatter={(v: number) => [(v * 100).toFixed(2) + '%']} contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                          <Bar dataKey="probability" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6"
                  >
                    <h3 className="font-semibold text-slate-900 mb-1">Nucleus Filtered (P={topP.toFixed(2)})</h3>
                    <p className="text-xs text-slate-500 mb-4">Smallest set with cumulative probability ≥ {topP.toFixed(2)}</p>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={nucleusData} margin={{ bottom: 30 }}>
                          <XAxis dataKey="token" angle={-45} textAnchor="end" fontSize={9} stroke="#94a3b8" interval={0} height={40} />
                          <YAxis stroke="#94a3b8" fontSize={9} />
                          <Tooltip formatter={(v: number) => [(v * 100).toFixed(2) + '%']} contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                          <Bar dataKey="probability" fill="#10b981" radius={[3, 3, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </motion.div>
                </div>
              </>
            )}

            {!samplingData && (
              <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-12 text-center">
                <BarChart3 className="h-12 w-12 text-slate-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-900 mb-2">Sampling Visualization</h3>
                <p className="text-slate-500 text-sm">
                  Adjust the parameters on the left to see how they affect the token probability distribution.
                  The visualizations update in real-time from the backend.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
      <ModuleNavBar />
    </main>
  );
}
