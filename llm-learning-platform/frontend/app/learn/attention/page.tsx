'use client';

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Eye, Play, Loader2, Settings,
  ToggleLeft, ToggleRight, Info, Zap, BarChart3,
  TrendingUp, Target, Activity, Grid3X3, ArrowRight
} from 'lucide-react';
import Link from 'next/link';
import * as d3 from 'd3';
import { api } from '@/utils/api';
import ModuleNavBar from '@/components/ModuleNavBar';

/* ==========================================================================
   STEP DEFINITIONS WITH EDUCATIONAL CONTENT
   ========================================================================== */

const STEPS = [
  {
    title: 'Input & Tokenize',
    icon: '📝',
    desc: 'Text is split into tokens and mapped to embeddings',
    explanation: 'Each word or character becomes a token. Tokens are converted to dense vectors using a learned embedding table, then positional encoding is added so the model knows the order of tokens.',
  },
  {
    title: 'Q, K, V Projection',
    icon: '🔑',
    desc: 'Project embeddings into Query, Key, Value spaces',
    explanation: 'Each token embedding is multiplied by three learned weight matrices (W_Q, W_K, W_V) to produce Query, Key, and Value vectors. Think of it like: Query = "what am I looking for?", Key = "what do I contain?", Value = "what information do I provide?"',
  },
  {
    title: 'Attention Scores',
    icon: '🎯',
    desc: 'Compute QKᵀ/√d_k — how much each token should attend to others',
    explanation: 'The dot product of Query and Key vectors measures compatibility. Dividing by √d_k prevents the scores from getting too large (which would push softmax into extreme values). Higher score = stronger connection between two tokens.',
  },
  {
    title: 'Softmax & Weights',
    icon: '📊',
    desc: 'Normalize scores into a probability distribution',
    explanation: 'Softmax converts raw scores into probabilities that sum to 1 for each query token. With a causal mask, tokens can only attend to themselves and earlier tokens (preventing information leakage from the future).',
  },
  {
    title: 'Multi-Head View',
    icon: '👁️',
    desc: 'Compare attention patterns across different heads',
    explanation: 'Multiple heads allow the model to attend to different aspects simultaneously — one head might focus on syntax (subject-verb), another on semantics (similar meanings), and another on position (nearby tokens). This is the power of multi-head attention.',
  },
  {
    title: 'Analytics',
    icon: '📈',
    desc: 'Entropy, sparsity, and attention flow metrics',
    explanation: 'Entropy measures how "spread out" attention is (high = uniform, low = focused). Sparsity measures how many weights are near zero. These metrics reveal what each head has learned to focus on.',
  },
];

/* ==========================================================================
   MAIN PAGE COMPONENT
   ========================================================================== */

export default function AttentionPage() {
  const heatmapRef = useRef<SVGSVGElement>(null);
  const vectorRef = useRef<SVGSVGElement>(null);
  const flowRef = useRef<SVGSVGElement>(null);

  // Input state
  const [inputText, setInputText] = useState('The cat sat on the mat');
  const [dModel, setDModel] = useState(64);
  const [numHeads, setNumHeads] = useState(4);
  const [numLayers, setNumLayers] = useState(1);
  const [showCausalMask, setShowCausalMask] = useState(true);

  // Data state
  const [attentionData, setAttentionData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Interaction state
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null);
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);
  const [showRawScores, setShowRawScores] = useState(false);

  // Compute attention
  const computeAttention = useCallback(async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.computeAttention({
        text: inputText,
        d_model: dModel,
        num_heads: numHeads,
        num_layers: numLayers,
        show_causal_mask: showCausalMask,
      });
      setAttentionData(data);
      setSelectedLayer(0);
      setSelectedHead(0);
      setSelectedCell(null);
      setCurrentStep(3); // Jump to softmax weights view
    } catch (err: any) {
      setError(err.message || 'Failed to compute attention');
    } finally {
      setLoading(false);
    }
  }, [inputText, dModel, numHeads, numLayers, showCausalMask]);

  useEffect(() => { computeAttention(); }, []);

  const currentHeadData = attentionData?.layers?.[selectedLayer]?.heads?.[selectedHead];

  // ===== HEATMAP DRAWING =====
  useEffect(() => {
    if (!currentHeadData || !heatmapRef.current || currentStep < 3) return;

    const svg = d3.select(heatmapRef.current);
    svg.selectAll('*').remove();

    const matrix = showRawScores ? currentHeadData.raw_scores : currentHeadData.attention_matrix;
    const tokens = attentionData.tokens;
    const n = tokens.length;

    const margin = { top: 60, right: 30, bottom: 30, left: 80 };
    const containerWidth = heatmapRef.current.clientWidth;
    const cellSize = Math.min(Math.max((containerWidth - margin.left - margin.right) / n, 22), 48);
    const width = cellSize * n + margin.left + margin.right;
    const height = cellSize * n + margin.top + margin.bottom;

    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Color scale
    const allVals = matrix.flat();
    const minVal = showRawScores ? Math.min(...allVals) : 0;
    const maxVal = showRawScores ? Math.max(...allVals) : 1;

    const colorScale = showRawScores
      ? d3.scaleSequential(d3.interpolateRdBu).domain([maxVal, minVal])
      : d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...allVals, 0.01)]);

    // Draw cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const value = matrix[i][j];
        const isSelected = selectedCell && selectedCell[0] === i && selectedCell[1] === j;
        const isRowHighlight = hoveredRow === i;

        g.append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('fill', colorScale(value))
          .attr('stroke', isSelected ? '#f97316' : isRowHighlight ? 'rgba(255,255,255,0.3)' : 'none')
          .attr('stroke-width', isSelected ? 2.5 : isRowHighlight ? 1 : 0)
          .attr('rx', 3)
          .style('cursor', 'pointer')
          .on('mouseover', function () {
            d3.select(this).attr('stroke', '#f97316').attr('stroke-width', 2);
            setHoveredRow(i);
          })
          .on('mouseout', function () {
            if (!isSelected) d3.select(this).attr('stroke', 'none').attr('stroke-width', 0);
            setHoveredRow(null);
          })
          .on('click', () => setSelectedCell([i, j]));

        if (cellSize >= 32) {
          g.append('text')
            .attr('x', j * cellSize + cellSize / 2)
            .attr('y', i * cellSize + cellSize / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('font-size', '9px')
            .attr('font-family', 'monospace')
            .attr('fill', !showRawScores && value > 0.4 ? '#0f172a' : '#94a3b8')
            .text(showRawScores ? value.toFixed(1) : value.toFixed(2));
        }
      }
    }

    // Column labels (Key)
    g.selectAll('.col-label').data(tokens).join('text')
      .attr('class', 'col-label')
      .attr('x', (_: any, i: number) => i * cellSize + cellSize / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px').attr('font-weight', '600').attr('fill', '#94a3b8')
      .text((d: any) => d.length > 6 ? d.slice(0, 5) + '…' : d);

    // Row labels (Query)
    g.selectAll('.row-label').data(tokens).join('text')
      .attr('class', 'row-label')
      .attr('x', -10)
      .attr('y', (_: any, i: number) => i * cellSize + cellSize / 2 + 4)
      .attr('text-anchor', 'end')
      .attr('font-size', '11px').attr('font-weight', '600').attr('fill', '#94a3b8')
      .text((d: any) => d.length > 6 ? d.slice(0, 5) + '…' : d);

    // Axis labels
    svg.append('text')
      .attr('x', margin.left + (cellSize * n) / 2).attr('y', 20)
      .attr('text-anchor', 'middle').attr('font-size', '12px').attr('fill', '#64748b').attr('font-weight', '500')
      .text('Key (attending to) →');
    svg.append('text')
      .attr('x', 16).attr('y', margin.top + (cellSize * n) / 2)
      .attr('text-anchor', 'middle').attr('font-size', '12px').attr('fill', '#64748b').attr('font-weight', '500')
      .attr('transform', `rotate(-90, 16, ${margin.top + (cellSize * n) / 2})`)
      .text('Query (from) →');

  }, [currentHeadData, attentionData, selectedCell, hoveredRow, currentStep, showRawScores]);

  // ===== ATTENTION FLOW LINES =====
  useEffect(() => {
    if (!currentHeadData || !flowRef.current || currentStep < 3) return;

    const svg = d3.select(flowRef.current);
    svg.selectAll('*').remove();

    const tokens = attentionData.tokens;
    const n = tokens.length;
    const matrix = currentHeadData.attention_matrix;
    const queryIdx = hoveredRow ?? (selectedCell ? selectedCell[0] : 0);

    const width = flowRef.current.clientWidth;
    const height = 120;
    svg.attr('width', width).attr('height', height);

    const tokenSpacing = width / (n + 1);
    const yTop = 25;
    const yBottom = 95;

    // Draw query token at top
    tokens.forEach((_: string, i: number) => {
      const x = (i + 1) * tokenSpacing;

      // Token circles
      svg.append('circle')
        .attr('cx', x).attr('cy', yTop).attr('r', 14)
        .attr('fill', i === queryIdx ? '#f97316' : 'rgba(255,255,255,0.1)')
        .attr('stroke', i === queryIdx ? '#fb923c' : 'rgba(255,255,255,0.2)')
        .attr('stroke-width', 1.5);

      svg.append('text')
        .attr('x', x).attr('y', yTop + 4)
        .attr('text-anchor', 'middle').attr('font-size', '10px').attr('fill', i === queryIdx ? '#fff' : '#94a3b8')
        .attr('font-weight', i === queryIdx ? '700' : '400')
        .text(tokens[i]?.length > 4 ? tokens[i].slice(0, 3) : tokens[i]);

      // Key circles at bottom
      svg.append('circle')
        .attr('cx', x).attr('cy', yBottom).attr('r', 14)
        .attr('fill', `rgba(99,102,241,${Math.min(matrix[queryIdx][i] * 1.5, 1)})`)
        .attr('stroke', 'rgba(255,255,255,0.2)').attr('stroke-width', 1);

      svg.append('text')
        .attr('x', x).attr('y', yBottom + 4)
        .attr('text-anchor', 'middle').attr('font-size', '10px').attr('fill', '#e2e8f0')
        .text(tokens[i]?.length > 4 ? tokens[i].slice(0, 3) : tokens[i]);

      // Attention lines
      const weight = matrix[queryIdx][i];
      if (weight > 0.01) {
        svg.append('line')
          .attr('x1', (queryIdx + 1) * tokenSpacing).attr('y1', yTop + 15)
          .attr('x2', x).attr('y2', yBottom - 15)
          .attr('stroke', `rgba(251,146,60,${Math.min(weight * 2, 0.9)})`)
          .attr('stroke-width', Math.max(0.5, weight * 5))
          .attr('stroke-linecap', 'round');
      }
    });

    // Labels
    svg.append('text').attr('x', 8).attr('y', yTop + 4).attr('font-size', '9px').attr('fill', '#64748b').text('Q');
    svg.append('text').attr('x', 8).attr('y', yBottom + 4).attr('font-size', '9px').attr('fill', '#64748b').text('K');
  }, [currentHeadData, attentionData, hoveredRow, selectedCell, currentStep]);

  // ===== Q/K/V VECTOR CHART =====
  useEffect(() => {
    if (!currentHeadData || !vectorRef.current || selectedCell === null) return;

    const svg = d3.select(vectorRef.current);
    svg.selectAll('*').remove();
    const width = vectorRef.current.clientWidth;
    const height = 220;
    const margin = { top: 30, right: 15, bottom: 25, left: 35 };
    svg.attr('width', width).attr('height', height);

    const [qi, ki] = selectedCell;
    const qVec = currentHeadData.q_vectors[qi] || [];
    const kVec = currentHeadData.k_vectors[ki] || [];
    const vVec = currentHeadData.v_vectors[ki] || [];
    const dims = qVec.length;
    if (dims === 0) return;

    const barWidth = Math.min((width - margin.left - margin.right) / (dims * 3 + dims), 10);
    const groupWidth = barWidth * 3 + 4;
    const allValues = [...qVec, ...kVec, ...vVec];
    const maxVal = Math.max(...allValues.map(Math.abs), 0.1);

    const xScale = d3.scaleLinear().domain([0, dims]).range([margin.left, margin.left + dims * groupWidth]);
    const yScale = d3.scaleLinear().domain([-maxVal, maxVal]).range([height - margin.bottom, margin.top]);

    const g = svg.append('g');

    // Zero line
    g.append('line')
      .attr('x1', margin.left).attr('x2', width - margin.right)
      .attr('y1', yScale(0)).attr('y2', yScale(0))
      .attr('stroke', 'rgba(255,255,255,0.1)');

    const colors = { q: '#3b82f6', k: '#ef4444', v: '#10b981' };

    for (let i = 0; i < dims; i++) {
      const x0 = margin.left + i * groupWidth;
      [{ vec: qVec, color: colors.q }, { vec: kVec, color: colors.k }, { vec: vVec, color: colors.v }]
        .forEach((item, vi) => {
          g.append('rect')
            .attr('x', x0 + vi * (barWidth + 1)).attr('width', barWidth)
            .attr('y', item.vec[i] >= 0 ? yScale(item.vec[i]) : yScale(0))
            .attr('height', Math.abs(yScale(item.vec[i]) - yScale(0)))
            .attr('fill', item.color).attr('rx', 1).attr('opacity', 0.85);
        });
    }

    // Legend
    [{ color: colors.q, label: `Q ("${attentionData.tokens[qi]}")` },
    { color: colors.k, label: `K ("${attentionData.tokens[ki]}")` },
    { color: colors.v, label: `V ("${attentionData.tokens[ki]}")` }]
      .forEach((item, i) => {
        const lx = margin.left + i * 120;
        g.append('rect').attr('x', lx).attr('y', 8).attr('width', 10).attr('height', 10).attr('fill', item.color).attr('rx', 2);
        g.append('text').attr('x', lx + 14).attr('y', 17).attr('font-size', '10px').attr('fill', '#94a3b8').text(item.label);
      });
  }, [currentHeadData, selectedCell, attentionData]);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-orange-950/30">
      {/* Header */}
      <header className="bg-slate-900/80 backdrop-blur-xl border-b border-white/5 sticky top-0 z-50">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/learn" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
                <ArrowLeft className="h-5 w-5" />
                <span className="text-sm font-medium">Back to Learn</span>
              </Link>
              <div className="h-6 w-px bg-white/10" />
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-orange-500 to-pink-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
                  <Eye className="h-4 w-4 text-white" />
                </div>
                <h1 className="text-lg font-semibold text-white">Attention Visualizer</h1>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Link href="/learn/embeddings" className="text-sm text-slate-500 hover:text-slate-300">← Embeddings</Link>
              <span className="text-slate-600">|</span>
              <Link href="/learn/transformer" className="text-sm text-slate-500 hover:text-slate-300">Transformer →</Link>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
        {/* Step Progress */}
        <div className="flex items-center gap-2 mb-6 overflow-x-auto pb-2">
          {STEPS.map((step, idx) => (
            <button key={idx} onClick={() => setCurrentStep(idx)}
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium whitespace-nowrap transition-all ${currentStep === idx
                ? 'bg-orange-600 text-white shadow-lg shadow-orange-600/30'
                : currentStep > idx
                  ? 'bg-orange-500/20 text-orange-300'
                  : 'bg-white/5 text-slate-500 ring-1 ring-white/10'
                }`}
            >
              <span className="text-sm">{step.icon}</span>
              {step.title}
            </button>
          ))}
        </div>

        {/* Current Step Explanation */}
        <motion.div key={currentStep} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-orange-500/10 to-pink-500/10 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-6 mb-6"
        >
          <div className="flex items-start gap-4">
            <div className="text-3xl">{STEPS[currentStep].icon}</div>
            <div>
              <h2 className="text-lg font-bold text-white mb-1">
                Step {currentStep + 1}: {STEPS[currentStep].title}
              </h2>
              <p className="text-sm text-slate-300 leading-relaxed">{STEPS[currentStep].explanation}</p>
              {currentStep === 3 && (
                <div className="mt-3 bg-slate-800/50 rounded-lg p-3 font-mono text-xs text-blue-300">
                  Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) × V
                </div>
              )}
            </div>
          </div>
        </motion.div>

        <div className="grid gap-6 lg:grid-cols-4">
          {/* Left Panel */}
          <div className="space-y-5">
            {/* Input */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-5">
              <h3 className="text-sm font-semibold text-white mb-3">Input Text</h3>
              <input type="text" value={inputText} onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && computeAttention()}
                className="w-full px-3 py-2 text-sm bg-white/5 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-orange-500"
                placeholder="Enter text..." />
              <button onClick={computeAttention} disabled={loading || !inputText.trim()}
                className="w-full mt-3 px-4 py-2.5 bg-gradient-to-r from-orange-600 to-pink-600 text-white text-sm font-medium rounded-lg hover:from-orange-500 hover:to-pink-500 disabled:opacity-50 transition-all flex items-center justify-center gap-2 shadow-lg shadow-orange-600/20"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Compute Attention
              </button>
            </div>

            {/* Config */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-5">
              <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <Settings className="h-4 w-4 text-orange-400" /> Model Config
              </h3>
              <div className="space-y-4">
                {[
                  { label: 'd_model', value: dModel, min: 32, max: 256, step: 32, set: setDModel, color: 'text-blue-400' },
                  { label: 'Num Heads', value: numHeads, min: 1, max: 8, step: 1, set: setNumHeads, color: 'text-purple-400' },
                  { label: 'Layers', value: numLayers, min: 1, max: 4, step: 1, set: setNumLayers, color: 'text-green-400' },
                ].map(cfg => (
                  <div key={cfg.label}>
                    <div className="flex justify-between mb-1">
                      <label className="text-xs font-medium text-slate-400">{cfg.label}</label>
                      <span className={`text-xs font-mono ${cfg.color}`}>{cfg.value}</span>
                    </div>
                    <input type="range" min={cfg.min} max={cfg.max} step={cfg.step} value={cfg.value}
                      onChange={(e) => cfg.set(parseInt(e.target.value))} className="w-full accent-orange-500" />
                  </div>
                ))}
                <button onClick={() => setShowCausalMask(!showCausalMask)}
                  className="w-full flex items-center justify-between px-3 py-2 bg-white/5 rounded-lg text-sm ring-1 ring-white/5"
                >
                  <span className="text-slate-300">Causal Mask</span>
                  {showCausalMask
                    ? <ToggleRight className="h-5 w-5 text-orange-400" />
                    : <ToggleLeft className="h-5 w-5 text-slate-500" />}
                </button>
              </div>
            </div>

            {/* Layer/Head Selector */}
            {attentionData && (
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-5">
                <h3 className="text-sm font-semibold text-white mb-3">Select Head & Layer</h3>
                {numLayers > 1 && (
                  <div className="mb-3">
                    <label className="text-xs text-slate-500 block mb-1">Layer</label>
                    <div className="flex gap-1">
                      {Array.from({ length: numLayers }).map((_, i) => (
                        <button key={i} onClick={() => setSelectedLayer(i)}
                          className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-all ${selectedLayer === i
                            ? 'bg-orange-600 text-white' : 'bg-white/5 text-slate-400 hover:bg-white/10'}`}
                        >L{i}</button>
                      ))}
                    </div>
                  </div>
                )}
                <div>
                  <label className="text-xs text-slate-500 block mb-1">Head</label>
                  <div className="flex flex-wrap gap-1">
                    {Array.from({ length: numHeads }).map((_, i) => (
                      <button key={i} onClick={() => setSelectedHead(i)}
                        className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-all ${selectedHead === i
                          ? 'bg-orange-600 text-white' : 'bg-white/5 text-slate-400 hover:bg-white/10'}`}
                      >H{i}</button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Cell Info */}
            {selectedCell && currentHeadData && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="bg-gradient-to-br from-orange-500/10 to-pink-500/10 rounded-2xl p-5 ring-1 ring-orange-500/20"
              >
                <h4 className="text-xs font-semibold text-slate-400 mb-2">Selected Cell</h4>
                <p className="text-sm text-white mb-1">
                  <span className="font-semibold text-blue-400">&quot;{attentionData.tokens[selectedCell[0]]}&quot;</span>
                  {' → '}
                  <span className="font-semibold text-red-400">&quot;{attentionData.tokens[selectedCell[1]]}&quot;</span>
                </p>
                <p className="text-2xl font-bold text-white font-mono">
                  {currentHeadData.attention_matrix[selectedCell[0]][selectedCell[1]].toFixed(4)}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Raw score: {currentHeadData.raw_scores[selectedCell[0]][selectedCell[1]].toFixed(3)}
                </p>
              </motion.div>
            )}

            {/* Head Entropy Badge */}
            {currentHeadData && (
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-5">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-purple-400" /> Head Analytics
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-400">Avg Entropy</span>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-green-500 to-yellow-500 rounded-full"
                          style={{ width: `${(currentHeadData.avg_entropy / currentHeadData.max_entropy) * 100}%` }} />
                      </div>
                      <span className="text-xs text-white font-mono">{currentHeadData.avg_entropy.toFixed(2)}</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-400">Sparsity</span>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                          style={{ width: `${currentHeadData.sparsity * 100}%` }} />
                      </div>
                      <span className="text-xs text-white font-mono">{(currentHeadData.sparsity * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-400">Max Entropy</span>
                    <span className="text-xs text-slate-500 font-mono">{currentHeadData.max_entropy.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Main Area */}
          <div className="lg:col-span-3 space-y-6">
            {error && (
              <div className="bg-red-500/10 text-red-300 text-sm rounded-xl p-4 ring-1 ring-red-500/20">{error}</div>
            )}

            {/* Attention Flow */}
            {currentHeadData && currentStep >= 3 && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 overflow-hidden"
              >
                <div className="p-4 border-b border-white/5">
                  <h3 className="font-semibold text-white flex items-center gap-2">
                    <Zap className="h-4 w-4 text-orange-400" /> Attention Flow
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Query token (top) → Key tokens (bottom). Line thickness = attention weight.
                    {hoveredRow !== null ? ` Showing: "${attentionData.tokens[hoveredRow]}"` : selectedCell ? ` Showing: "${attentionData.tokens[selectedCell[0]]}"` : ' Hover over heatmap rows to explore.'}
                  </p>
                </div>
                <div className="p-4">
                  <svg ref={flowRef} width="100%" height={120} />
                </div>
              </motion.div>
            )}

            {/* Heatmap */}
            {currentStep >= 3 && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 overflow-hidden"
              >
                <div className="p-4 border-b border-white/5 flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-white">
                      {showRawScores ? 'Raw Scores (QKᵀ/√d_k)' : 'Attention Weights (after Softmax)'}
                      {' — '}L{selectedLayer}H{selectedHead}
                    </h3>
                    <p className="text-sm text-slate-500">
                      {attentionData ? `${attentionData.seq_len} tokens • d_model=${attentionData.d_model} • head_dim=${attentionData.head_dim}` : 'Loading...'}
                      {showCausalMask && ' • causal'}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <button onClick={() => setShowRawScores(!showRawScores)}
                      className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-all ${showRawScores
                        ? 'bg-orange-600 text-white' : 'bg-white/5 text-slate-400 hover:bg-white/10 ring-1 ring-white/10'}`}
                    >
                      {showRawScores ? 'Raw Scores' : 'Softmax'}
                    </button>
                    {loading && <Loader2 className="h-5 w-5 text-orange-400 animate-spin" />}
                  </div>
                </div>
                <div className="p-4 overflow-x-auto">
                  <svg ref={heatmapRef} width="100%" height={500} />
                </div>
              </motion.div>
            )}

            {/* Token Input/Embedding Display (Steps 0-2) */}
            {currentStep <= 2 && attentionData && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-6"
              >
                <h3 className="font-semibold text-white mb-4">
                  {currentStep === 0 ? 'Tokenized Input' : currentStep === 1 ? 'Q, K, V Projections' : 'Projection Matrices'}
                </h3>
                {currentStep === 0 && (
                  <div className="space-y-4">
                    <div className="flex flex-wrap gap-2">
                      {attentionData.tokens.map((t: string, i: number) => (
                        <div key={i} className="px-3 py-2 bg-white/5 rounded-lg ring-1 ring-white/10 text-center">
                          <div className="text-white font-mono text-sm">{t}</div>
                          <div className="text-xs text-slate-500 mt-1">pos {i}</div>
                        </div>
                      ))}
                    </div>
                    {attentionData.token_embeddings && (
                      <div>
                        <p className="text-xs text-slate-500 mb-2">Token embeddings (first 8 dims):</p>
                        <div className="overflow-x-auto">
                          <table className="text-xs font-mono text-slate-400">
                            <tbody>
                              {attentionData.tokens.slice(0, 6).map((t: string, i: number) => (
                                <tr key={i}>
                                  <td className="pr-2 text-slate-500">{t}</td>
                                  {attentionData.token_embeddings[i]?.slice(0, 8).map((v: number, j: number) => (
                                    <td key={j} className={`px-1 ${v > 0 ? 'text-blue-400' : 'text-red-400'}`}>
                                      {v.toFixed(2)}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                )}
                {(currentStep === 1 || currentStep === 2) && currentHeadData && (
                  <div className="space-y-3">
                    <p className="text-xs text-slate-400">
                      Input embeddings (d_model={attentionData.d_model}) are projected through W_Q, W_K, W_V matrices into head_dim={attentionData.head_dim} vectors.
                    </p>
                    <div className="grid grid-cols-3 gap-4">
                      {[
                        { label: 'Query (Q)', color: 'text-blue-400', border: 'ring-blue-500/30', data: currentHeadData.q_vectors },
                        { label: 'Key (K)', color: 'text-red-400', border: 'ring-red-500/30', data: currentHeadData.k_vectors },
                        { label: 'Value (V)', color: 'text-green-400', border: 'ring-green-500/30', data: currentHeadData.v_vectors },
                      ].map(proj => (
                        <div key={proj.label} className={`bg-white/5 rounded-xl p-3 ring-1 ${proj.border}`}>
                          <h4 className={`text-xs font-semibold ${proj.color} mb-2`}>{proj.label}</h4>
                          <div className="space-y-1 overflow-x-auto">
                            {proj.data.slice(0, 4).map((row: number[], ri: number) => (
                              <div key={ri} className="flex gap-1">
                                <span className="text-[10px] text-slate-600 w-6 shrink-0">{attentionData.tokens[ri]}</span>
                                {row.slice(0, 4).map((v: number, ci: number) => (
                                  <span key={ci} className={`text-[10px] font-mono ${v > 0 ? 'text-slate-400' : 'text-slate-500'}`}>
                                    {v.toFixed(2)}
                                  </span>
                                ))}
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* Q/K/V Vectors */}
            {selectedCell && currentHeadData && currentStep >= 3 && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10"
              >
                <div className="p-4 border-b border-white/5">
                  <h3 className="font-semibold text-white">Q, K, V Vectors</h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Click a cell to see the Query, Key, Value vectors for that token pair
                  </p>
                </div>
                <div className="p-4"><svg ref={vectorRef} width="100%" height={220} /></div>
              </motion.div>
            )}

            {/* Multi-Head Comparison */}
            {attentionData && currentStep >= 4 && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-6"
              >
                <h3 className="font-semibold text-white mb-1">Multi-Head Comparison</h3>
                <p className="text-xs text-slate-500 mb-4">Each head learns different attention patterns</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {attentionData.layers[selectedLayer]?.heads.map((head: any, hIdx: number) => (
                    <div key={hIdx} onClick={() => setSelectedHead(hIdx)}
                      className={`cursor-pointer rounded-xl p-3 transition-all ${selectedHead === hIdx
                        ? 'ring-2 ring-orange-500 bg-orange-500/10' : 'bg-white/5 hover:bg-white/10'}`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-xs font-semibold text-white">Head {hIdx}</h4>
                        <span className="text-[10px] text-slate-500">H={head.avg_entropy?.toFixed(2)}</span>
                      </div>
                      <div className="grid gap-px" style={{
                        gridTemplateColumns: `repeat(${Math.min(attentionData.seq_len, 8)}, 1fr)`
                      }}>
                        {head.attention_matrix.slice(0, 8).flatMap((row: number[], ri: number) =>
                          row.slice(0, 8).map((val: number, ci: number) => (
                            <div key={`${ri}-${ci}`} className="aspect-square rounded-sm"
                              style={{ backgroundColor: `rgba(251, 146, 60, ${val})`, minWidth: '4px', minHeight: '4px' }} />
                          ))
                        )}
                      </div>
                      <div className="flex justify-between mt-2 text-[10px] text-slate-500">
                        <span>S: {(head.sparsity * 100).toFixed(0)}%</span>
                        <span>E: {head.avg_entropy?.toFixed(1)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Analytics Dashboard */}
            {attentionData && currentStep >= 5 && currentHeadData && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="space-y-4"
              >
                {/* Entropy per token */}
                <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-6">
                  <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-green-400" /> Entropy per Token (L{selectedLayer}H{selectedHead})
                  </h3>
                  <p className="text-xs text-slate-500 mb-4">
                    High entropy = attention spread across many tokens. Low = focused on one token.
                  </p>
                  <div className="space-y-2">
                    {attentionData.tokens.map((t: string, i: number) => (
                      <div key={i} className="flex items-center gap-3">
                        <span className="text-xs text-slate-400 font-mono w-12 text-right truncate">{t}</span>
                        <div className="flex-1 h-4 bg-white/5 rounded-full overflow-hidden">
                          <div className="h-full rounded-full bg-gradient-to-r from-green-600 to-yellow-500"
                            style={{ width: `${(currentHeadData.entropy_per_token[i] / currentHeadData.max_entropy) * 100}%` }} />
                        </div>
                        <span className="text-xs text-slate-400 font-mono w-8">{currentHeadData.entropy_per_token[i].toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Attention Received */}
                <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-6">
                  <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                    <Target className="h-4 w-4 text-blue-400" /> Attention Received per Token
                  </h3>
                  <p className="text-xs text-slate-500 mb-4">
                    How much total attention each token receives from all others. High = &quot;important&quot; token.
                  </p>
                  <div className="space-y-2">
                    {attentionData.tokens.map((t: string, i: number) => {
                      const maxReceived = Math.max(...(currentHeadData.attention_received || [1]));
                      return (
                        <div key={i} className="flex items-center gap-3">
                          <span className="text-xs text-slate-400 font-mono w-12 text-right truncate">{t}</span>
                          <div className="flex-1 h-4 bg-white/5 rounded-full overflow-hidden">
                            <div className="h-full rounded-full bg-gradient-to-r from-blue-600 to-purple-500"
                              style={{ width: `${(currentHeadData.attention_received[i] / maxReceived) * 100}%` }} />
                          </div>
                          <span className="text-xs text-slate-400 font-mono w-8">
                            {currentHeadData.attention_received[i].toFixed(2)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Dominant Attention */}
                <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-6">
                  <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                    <Grid3X3 className="h-4 w-4 text-orange-400" /> Dominant Attention Target
                  </h3>
                  <p className="text-xs text-slate-500 mb-4">
                    For each query, which key token receives the highest attention weight.
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {attentionData.tokens.map((t: string, i: number) => (
                      <div key={i} className="flex items-center gap-2 bg-white/5 rounded-lg px-3 py-2">
                        <span className="text-xs text-orange-400 font-mono">{t}</span>
                        <ArrowRight className="h-3 w-3 text-slate-600" />
                        <span className="text-xs text-blue-400 font-mono font-bold">
                          {attentionData.tokens[currentHeadData.dominant_indices[i]]}
                        </span>
                        <span className="text-[10px] text-slate-500 ml-auto">
                          {(currentHeadData.dominant_weights[i] * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step Navigation */}
            <div className="flex items-center justify-between">
              <button onClick={() => setCurrentStep(Math.max(0, currentStep - 1))} disabled={currentStep === 0}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-400 bg-white/5 rounded-lg ring-1 ring-white/10 hover:bg-white/10 disabled:opacity-40 transition-all"
              >
                ← Previous
              </button>
              <span className="text-xs text-slate-500">Step {currentStep + 1} of {STEPS.length}</span>
              <button onClick={() => setCurrentStep(Math.min(STEPS.length - 1, currentStep + 1))} disabled={currentStep === STEPS.length - 1}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-400 bg-white/5 rounded-lg ring-1 ring-white/10 hover:bg-white/10 disabled:opacity-40 transition-all"
              >
                Next →
              </button>
            </div>
          </div>
        </div>
      </div>
      <ModuleNavBar />
    </main>
  );
}
