'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Zap, Play, Pause, Loader2, Database, Brain
} from 'lucide-react';
import Link from 'next/link';
import { api } from '@/utils/api';
import ModuleNavBar from '@/components/ModuleNavBar';

/* ==========================================================================
   ARCHITECTURE CONFIGURATION
   ========================================================================== */

const ARCH_LAYERS = [
  { id: 'input', name: 'Input Embeddings', color: 'from-blue-500 to-cyan-500', icon: '📝', desc: 'Token + Position embeddings' },
  { id: 'norm1', name: 'RMSNorm (Pre-Attention)', color: 'from-slate-500 to-slate-400', icon: '⚖️', desc: 'Pre-attention normalization' },
  { id: 'attention', name: 'Multi-Head Self-Attention', color: 'from-amber-500 to-orange-500', icon: '🧠', desc: 'Core attention mechanism' },
  { id: 'add1', name: 'Add (Residual 1)', color: 'from-emerald-500 to-teal-500', icon: '➕', desc: 'First residual connection' },
  { id: 'norm2', name: 'RMSNorm (Pre-MLP)', color: 'from-slate-500 to-slate-400', icon: '⚖️', desc: 'Pre-MLP normalization' },
  { id: 'mlp', name: 'Feedforward (MLP)', color: 'from-indigo-500 to-purple-500', icon: '🚀', desc: 'Position-wise FFN' },
  { id: 'add2', name: 'Add (Residual 2)', color: 'from-emerald-500 to-teal-500', icon: '➕', desc: 'Second residual connection' },
  { id: 'output', name: 'Block Output', color: 'from-slate-600 to-slate-700', icon: '📤', desc: 'To next layer or logits' },
];

const COMPONENT_DETAILS: Record<string, any> = {
  input: {
    title: 'Input Embeddings',
    desc: 'Converts discrete tokens into dense vectors and adds positional encoding so the model knows the order of words.',
    details: ['x = TokenEmb(tokens) + PosEmb(positions)', 'Shape: (seq_len, d_model)'],
    formula: 'x = W_e[tokens] + P_e'
  },
  norm1: {
    title: 'RMSNorm (Pre-Attention)',
    desc: 'Normalizes the input activations to stabilize training. RMSNorm is a modern, faster alternative to LayerNorm used in Llama.',
    details: ['Calculates Root Mean Square', 'Applies learnable scale parameter γ'],
    formula: 'RMSNorm(x) = (x / √(mean(x²) + ε)) ⊙ γ'
  },
  attention: {
    title: 'Multi-Head Self-Attention',
    desc: 'The core mechanism. Each token projects into Query, Key, and Value vectors to attend to other tokens and gather context.',
    details: ['Computes QKᵀ matrix for attention scores', 'Applies causal mask (for autoregressive models)', 'Softmax converts scores to weights'],
    formula: 'Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V'
  },
  add1: {
    title: 'Residual Connection 1',
    desc: 'Adds the original input directly to the attention output. This "skip connection" allows gradients to flow easily during training.',
    details: ['Prevents vanishing gradients'],
    formula: 'x₁ = x + Attention(RMSNorm(x))'
  },
  norm2: {
    title: 'RMSNorm (Pre-MLP)',
    desc: 'Second normalization layer before the feedforward network.',
    details: ['Independent learnable parameters from Norm 1'],
    formula: 'RMSNorm(x₁) = (x₁ / √(mean(x₁²) + ε)) ⊙ γ₂'
  },
  mlp: {
    title: 'Feedforward Network (MLP)',
    desc: 'A two-layer position-wise network that processes each token independently, expanding dimensions (usually 4x) then projecting back.',
    details: ['Usually Uses GELU or SwiGLU activation', 'Expands d_model → 4*d_model → d_model'],
    formula: 'MLP(x) = GELU(xW₁ + b₁)W₂ + b₂'
  },
  add2: {
    title: 'Residual Connection 2',
    desc: 'Adds the attention output to the MLP output, completing the Transformer Block.',
    details: ['Output is ready for the next block'],
    formula: 'x₂ = x₁ + MLP(RMSNorm(x₁))'
  },
  output: {
    title: 'Block Output',
    desc: 'The final contextualized representations of the tokens from this block.',
    details: ['Passed to the next identical Transformer block', 'Or to the final LayerNorm and language modeling head'],
    formula: 'Output = x₂'
  }
};

/* ==========================================================================
   MAIN COMPONENT
   ========================================================================== */

export default function TransformerPage() {
  const [activeLayer, setActiveLayer] = useState<string | null>('attention');

  // Forward pass state
  const [inputText, setInputText] = useState('Visualize the transformer');
  const [forwardData, setForwardData] = useState<any>(null);
  const [loadingForward, setLoadingForward] = useState(false);
  const [activeStepIndex, setActiveStepIndex] = useState<number>(-1);
  const [isPlaying, setIsPlaying] = useState(false);

  // Map backend forward pass steps (0-13) to our architecture layers
  const stepToLayerMap: Record<number, string> = {
    0: 'input',
    1: 'norm1',
    2: 'attention', 3: 'attention', 4: 'attention', 5: 'attention', 6: 'attention',
    7: 'add1',
    8: 'norm2',
    9: 'mlp', 10: 'mlp', 11: 'mlp',
    12: 'add2',
    13: 'output'
  };

  const computeForward = useCallback(async () => {
    if (!inputText.trim()) return;
    setLoadingForward(true);
    setIsPlaying(false);
    setActiveStepIndex(-1);

    try {
      // Mock call to get the sequence length and parameters
      const data = await api.computeForwardStep({
        text: inputText,
        d_model: 64,
        num_heads: 4,
        num_layers: 1,
        step: 0
      });
      setForwardData(data);

      // Start animation sequence
      setIsPlaying(true);
      setActiveStepIndex(0);
      setActiveLayer('input');
    } catch (err) {
      console.error('Forward pass error:', err);
    } finally {
      setLoadingForward(false);
    }
  }, [inputText]);

  // Animation Loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setActiveStepIndex((prev) => {
        const next = prev + 1;
        if (next > 13) {
          setIsPlaying(false);
          setActiveLayer('output');
          return -1; // End
        }
        setActiveLayer(stepToLayerMap[next]);
        return next;
      });
    }, 1500); // 1.5s per step

    return () => clearInterval(interval);
  }, [isPlaying]);

  const activeMappedLayerIndex = ARCH_LAYERS.findIndex(l => l.id === activeLayer);

  return (
    <main className="min-h-screen bg-slate-950 flex flex-col font-sans text-slate-200 selection:bg-indigo-500/30">
      {/* Header */}
      <header className="bg-slate-900/80 backdrop-blur-xl border-b border-white/5 sticky top-0 z-50 shrink-0">
        <div className="mx-auto w-full px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/learn" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
                <ArrowLeft className="h-5 w-5" />
                <span className="text-sm font-medium">Back to Learn</span>
              </Link>
              <div className="h-6 w-px bg-white/10" />
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                  <Database className="h-4 w-4 text-white" />
                </div>
                <h1 className="text-lg font-semibold text-white">Transformer 2D Architecture</h1>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Link href="/learn/attention" className="text-sm text-slate-500 hover:text-slate-300">← Attention</Link>
              <span className="text-slate-600">|</span>
              <Link href="/learn/training" className="text-sm text-slate-500 hover:text-slate-300">Training →</Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main UI Layout */}
      <div className="flex-1 flex overflow-hidden max-w-[1600px] mx-auto w-full">

        {/* Left Side: 2D Interactive Architecture */}
        <div className="flex-1 relative bg-transparent flex items-center justify-center overflow-auto custom-scrollbar p-8">

          <div className="relative w-full max-w-2xl py-12 flex flex-col items-center">

            {/* SVG Background for Connections */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none z-0" style={{ preserveAspectRatio: "none" }}>
              {/* Main Stem Line */}
              <line
                x1="50%" y1="5%"
                x2="50%" y2="95%"
                stroke="#334155"
                strokeWidth="4"
                strokeLinecap="round"
              />

              {/* Highlight overlay for main stem when animating */}
              {activeStepIndex >= 0 && (
                <motion.line
                  x1="50%" y1="5%"
                  x2="50%" y2="95%"
                  stroke="url(#pulseGradient)"
                  strokeWidth="6"
                  strokeLinecap="round"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: (activeMappedLayerIndex + 1) / ARCH_LAYERS.length }}
                  transition={{ duration: 0.5 }}
                />
              )}

              {/* Residual 1: Input -> Add1 */}
              <path
                d="M 30% 12% C 15% 12%, 15% 48%, 30% 48%"
                fill="none"
                stroke={['add1', 'norm1', 'attention', 'input'].includes(activeLayer || '') ? "#10b981" : "#334155"}
                strokeWidth={['add1', 'norm1', 'attention', 'input'].includes(activeLayer || '') ? "4" : "2"}
                strokeDasharray="8 6"
              />

              {/* Residual 2: Add1 -> Add2 */}
              <path
                d="M 70% 48% C 85% 48%, 85% 82%, 70% 82%"
                fill="none"
                stroke={['add2', 'norm2', 'mlp', 'add1'].includes(activeLayer || '') ? "#10b981" : "#334155"}
                strokeWidth={['add2', 'norm2', 'mlp', 'add1'].includes(activeLayer || '') ? "4" : "2"}
                strokeDasharray="8 6"
              />

              <defs>
                <linearGradient id="pulseGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#818cf8" />
                  <stop offset="50%" stopColor="#c084fc" />
                  <stop offset="100%" stopColor="#818cf8" />
                </linearGradient>
              </defs>
            </svg>

            {/* Architecture Nodes */}
            <div className="flex flex-col items-center gap-8 w-full z-10">
              {ARCH_LAYERS.map((layer, index) => {
                const isActive = layer.id === activeLayer;
                const isPassed = activeStepIndex >= 0 && index <= activeMappedLayerIndex;

                return (
                  <motion.div
                    key={layer.id}
                    onClick={() => { setActiveLayer(layer.id); setIsPlaying(false); setActiveStepIndex(-1); }}
                    whileHover={{ scale: 1.02, y: -2 }}
                    className={`relative cursor-pointer w-[60%] lg:w-[45%] flex items-center justify-between p-4 rounded-2xl backdrop-blur-xl border transition-all duration-300 ${isActive
                        ? 'bg-slate-800/90 border-white/20 shadow-[0_0_30px_rgba(255,255,255,0.1)]'
                        : isPassed
                          ? 'bg-slate-800/60 border-indigo-500/30 shadow-[0_0_20px_rgba(99,102,241,0.15)]'
                          : 'bg-slate-900/60 border-white/5 hover:border-white/10'
                      }`}
                  >
                    {/* Glowing background indicator when active */}
                    {isActive && (
                      <div className={`absolute inset-0 rounded-2xl bg-gradient-to-r ${layer.color} opacity-10 blur-md pointer-events-none`} />
                    )}

                    <div className="flex items-center gap-4 z-10 w-full">
                      <div className={`h-12 w-12 shrink-0 rounded-xl bg-gradient-to-br ${layer.color} flex items-center justify-center text-xl shadow-inner ring-1 ring-white/20`}>
                        {layer.icon}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h3 className={`font-bold tracking-wide text-sm ${isActive ? 'text-white' : 'text-slate-300'}`}>
                            {layer.name}
                          </h3>
                        </div>
                        <p className="text-xs text-slate-400 mt-1 line-clamp-1">{layer.desc}</p>
                      </div>
                    </div>

                    {/* Active Pulse Ring */}
                    {isActive && (
                      <div className="absolute -inset-0.5 rounded-2xl ring-2 ring-indigo-500/50 animate-pulse pointer-events-none" />
                    )}
                  </motion.div>
                );
              })}
            </div>

          </div>
        </div>

        {/* Right Side: Educational & Control Panel */}
        <div className="w-[450px] bg-slate-900/90 backdrop-blur-xl border-l border-white/5 flex flex-col shrink-0 overflow-y-auto z-20 custom-scrollbar shadow-2xl">

          <div className="p-6 space-y-6">

            {/* Live Forward Pass Runner */}
            <div className="bg-slate-950/50 rounded-2xl p-5 ring-1 ring-white/10 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 to-purple-500 opacity-50" />

              <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
                <Zap className="h-4 w-4 text-amber-400" /> Live Forward Pass
              </h3>
              <p className="text-xs text-slate-400 mb-4">Watch data flow through the architecture layer by layer.</p>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="flex-1 px-3 py-2 text-sm bg-slate-900 border border-white/10 rounded-lg text-white focus:ring-2 focus:ring-indigo-500 outline-none"
                  placeholder="Enter text to process..."
                  onKeyDown={(e) => e.key === 'Enter' && computeForward()}
                />
                <button
                  onClick={isPlaying ? () => { setIsPlaying(false); setActiveStepIndex(-1); } : computeForward}
                  disabled={loadingForward}
                  className={`px-4 py-2 text-white text-sm font-medium rounded-lg transition-all flex items-center gap-2 ${isPlaying
                      ? 'bg-slate-700 hover:bg-slate-600'
                      : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 shadow-lg shadow-indigo-500/25'
                    }`}
                >
                  {loadingForward ? <Loader2 className="h-4 w-4 animate-spin" /> :
                    isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  {isPlaying ? 'Stop' : 'Run'}
                </button>
              </div>

              {/* Progress Bar (visible during animation) */}
              <AnimatePresence>
                {activeStepIndex >= 0 && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="mt-4 overflow-hidden">
                    <div className="flex justify-between text-[10px] text-slate-400 mb-1 font-medium tracking-wide">
                      <span>STEP {activeStepIndex + 1}/14</span>
                      <span className="text-indigo-300">{stepToLayerMap[activeStepIndex].toUpperCase()}</span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500"
                        animate={{ width: `${(Math.max(0, activeStepIndex) / 13) * 100}%` }}
                        transition={{ ease: "linear", duration: 1.5 }}
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Active Layer Info Panel */}
            <AnimatePresence mode="wait">
              <motion.div
                key={activeLayer || 'none'}
                initial={{ opacity: 0, y: 10, filter: 'blur(4px)' }}
                animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                exit={{ opacity: 0, scale: 0.98, filter: 'blur(4px)' }}
                transition={{ duration: 0.2 }}
                className="bg-slate-800/50 backdrop-blur-md rounded-2xl p-6 ring-1 ring-white/10 shadow-xl"
              >
                {activeLayer ? (
                  <>
                    <div className="flex items-start gap-4 mb-5">
                      <div className={`h-12 w-12 rounded-xl bg-gradient-to-br ${ARCH_LAYERS.find(l => l.id === activeLayer)?.color} flex items-center justify-center text-2xl shadow-lg ring-1 ring-white/20 shrink-0`}>
                        {ARCH_LAYERS.find(l => l.id === activeLayer)?.icon}
                      </div>
                      <div>
                        <h2 className="text-xl font-bold text-white leading-tight mb-1">
                          {COMPONENT_DETAILS[activeLayer].title}
                        </h2>
                        <span className="text-[10px] uppercase tracking-wider text-indigo-400 font-bold bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                          Architecture Node
                        </span>
                      </div>
                    </div>

                    <p className="text-sm text-slate-300 leading-relaxed mb-6">
                      {COMPONENT_DETAILS[activeLayer].desc}
                    </p>

                    <div className="space-y-4">
                      {/* Key Details */}
                      <div>
                        <h4 className="text-[11px] font-bold text-slate-500 mb-2 uppercase tracking-wider">Key Operations</h4>
                        <ul className="space-y-2">
                          {COMPONENT_DETAILS[activeLayer].details.map((detail: string, i: number) => (
                            <li key={i} className="text-sm text-slate-300 flex items-start gap-3 bg-slate-900/50 p-2.5 rounded-lg border border-white/5">
                              <span className="text-indigo-400 shrink-0 mt-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-indigo-500/10 text-[10px] font-bold">
                                {i + 1}
                              </span>
                              <span className="leading-snug">{detail}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Math Formula */}
                      <div className="bg-slate-950 rounded-xl p-4 ring-1 ring-white/10 border-l-2 border-l-amber-500">
                        <h4 className="text-[10px] font-bold text-amber-500 mb-2 flex items-center gap-1.5 uppercase tracking-wider">
                          <MathIcon /> Equation
                        </h4>
                        <div className="font-mono text-sm text-amber-100/90 overflow-x-auto custom-scrollbar pb-1">
                          {COMPONENT_DETAILS[activeLayer].formula}
                        </div>
                      </div>

                      {/* Tensor Shape display */}
                      <div className="bg-indigo-950/30 rounded-xl p-4 ring-1 ring-indigo-500/20 flex flex-col gap-2">
                        <div className="flex items-center gap-2">
                          <Database className="h-4 w-4 text-emerald-400" />
                          <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Output Tensor Shape</h4>
                        </div>
                        <div className="font-mono text-sm text-emerald-300 bg-emerald-500/10 p-2 rounded border border-emerald-500/20 text-center font-semibold tracking-wide">
                          {forwardData?.tokens ? `(${forwardData.tokens.length}, 64)` : '(seq_len, d_model)'}
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="h-64 flex flex-col items-center justify-center text-slate-500 text-center">
                    <MouseClickIcon className="h-12 w-12 text-slate-700 mb-4" />
                    <p>Select any layer in the diagram<br />to view its architectural details.</p>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>

            {/* Architecture Explainer */}
            <div className="bg-gradient-to-r from-blue-900/20 to-cyan-900/20 rounded-2xl p-5 border border-blue-500/20">
              <h3 className="text-sm font-bold text-blue-400 mb-2 flex items-center gap-2">
                <Brain className="h-4 w-4" /> Why this structure?
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                The Transformer block elegantly balances two tasks: <strong>information gathering</strong> and <strong>information processing</strong>.
                The Attention layer acts as a routing mechanism, moving data between tokens. The MLP layer then processes that aggregated data
                independently at each position. Residual connections ensure the original signal isn't lost during this deep transformation.
              </p>
            </div>

          </div>
        </div>
      </div>
      <ModuleNavBar />
    </main>
  );
}

// Helpers
function MathIcon() { return <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" /></svg> }
function MouseClickIcon({ className }: { className?: string }) { return <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M9 11a3 3 0 1 0 6 0 3 3 0 0 0-6 0" /><path d="M17.657 16.657L13.414 20.9a2 2 0 0 1-2.827 0l-4.244-4.243a8 8 0 1 1 11.314 0z" /></svg> }
