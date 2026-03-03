"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence, useScroll, useTransform, useSpring } from "framer-motion";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  RotateCcw,
  Info,
  BookOpen,
  Code,
  Zap,
  Brain,
  Layers,
  Eye,
  ArrowRight,
  Matrix,
  Settings,
  Sparkles,
  Terminal,
  Lightbulb,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
} from "lucide-react";
import Link from "next/link";

// ============================================
// TYPES & INTERFACES
// ============================================
interface Token {
  id: number;
  text: string;
  embedding: number[];
  position: number;
}

interface AttentionHead {
  id: number;
  name: string;
  query: number[][];
  key: number[][];
  value: number[][];
  attentionWeights: number[][];
}

interface Section {
  id: string;
  title: string;
  icon: React.ReactNode;
}

// ============================================
// CONSTANTS & DATA
// ============================================
const SECTIONS: Section[] = [
  { id: "intro", title: "Introduction", icon: <BookOpen className="w-5 h-5" /> },
  { id: "architecture", title: "Architecture", icon: <Layers className="w-5 h-5" /> },
  { id: "embeddings", title: "Embeddings", icon: <Matrix className="w-5 h-5" /> },
  { id: "attention", title: "Self-Attention", icon: <Eye className="w-5 h-5" /> },
  { id: "multihead", title: "Multi-Head Attention", icon: <Brain className="w-5 h-5" /> },
  { id: "ffn", title: "Feed-Forward", icon: <Zap className="w-5 h-5" /> },
  { id: "code", title: "Code Implementation", icon: <Code className="w-5 h-5" /> },
  { id: "playground", title: "Interactive Playground", icon: <Sparkles className="w-5 h-5" /> },
];

const SAMPLE_TOKENS = ["The", "transformer", "architecture", "revolutionized", "NLP"];

const EMBEDDING_DIM = 8;

// ============================================
// UTILITY FUNCTIONS
// ============================================
const generateRandomEmbedding = (dim: number): number[] =>
  Array.from({ length: dim }, () => Math.random() * 2 - 1);

const softmax = (arr: number[]): number[] => {
  const max = Math.max(...arr);
  const exp = arr.map((x) => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((x) => x / sum);
};

const formatNumber = (num: number, decimals: number = 2): string =>
  num.toFixed(decimals);

// ============================================
// COMPONENTS
// ============================================

// --- Navigation Sidebar ---
const NavigationSidebar = ({ activeSection, onSectionClick }: { activeSection: string; onSectionClick: (id: string) => void }) => (
  <motion.nav
    initial={{ x: -100, opacity: 0 }}
    animate={{ x: 0, opacity: 1 }}
    className="fixed left-4 top-1/2 -translate-y-1/2 z-50 hidden lg:block"
  >
    <div className="bg-slate-900/90 backdrop-blur-xl rounded-2xl p-4 border border-slate-700/50 shadow-2xl">
      <div className="space-y-2">
        {SECTIONS.map((section) => (
          <button
            key={section.id}
            onClick={() => onSectionClick(section.id)}
            className={`flex items-center gap-3 w-full px-4 py-3 rounded-xl transition-all duration-300 group ${
              activeSection === section.id
                ? "bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-lg shadow-violet-500/25"
                : "text-slate-400 hover:text-white hover:bg-slate-800/50"
            }`}
          >
            <span className={`transition-transform duration-300 ${activeSection === section.id ? "scale-110" : "group-hover:scale-110"}`}>
              {section.icon}
            </span>
            <span className="text-sm font-medium whitespace-nowrap">{section.title}</span>
            {activeSection === section.id && (
              <motion.div
                layoutId="activeIndicator"
                className="absolute right-2 w-1.5 h-1.5 rounded-full bg-white"
              />
            )}
          </button>
        ))}
      </div>
    </div>
  </motion.nav>
);

// --- Hero Section ---
const HeroSection = () => {
  const { scrollY } = useScroll();
  const y1 = useTransform(scrollY, [0, 500], [0, 200]);
  const y2 = useTransform(scrollY, [0, 500], [0, -150]);
  const opacity = useTransform(scrollY, [0, 300], [1, 0]);

  return (
    <motion.section
      id="intro"
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
      style={{ opacity }}
    >
      {/* Animated Background */}
      <div className="absolute inset-0 bg-slate-950">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(124,58,237,0.15),transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,rgba(99,102,241,0.15),transparent_50%)]" />
        
        {/* Floating Elements */}
        <motion.div
          style={{ y: y1 }}
          className="absolute top-20 left-10 w-64 h-64 bg-violet-500/10 rounded-full blur-3xl"
          animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
          transition={{ duration: 8, repeat: Infinity }}
        />
        <motion.div
          style={{ y: y2 }}
          className="absolute bottom-20 right-10 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl"
          animate={{ scale: [1.2, 1, 1.2], opacity: [0.5, 0.3, 0.5] }}
          transition={{ duration: 10, repeat: Infinity }}
        />
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-300 text-sm font-medium mb-8"
        >
          <Sparkles className="w-4 h-4" />
          <span>The Architecture That Changed Everything</span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="text-6xl md:text-8xl font-bold mb-6"
        >
          <span className="bg-gradient-to-r from-white via-violet-200 to-indigo-200 bg-clip-text text-transparent">
            Transformer
          </span>
          <br />
          <span className="bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
            Architecture
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-xl md:text-2xl text-slate-400 max-w-3xl mx-auto mb-12 leading-relaxed"
        >
          Discover the revolutionary neural network architecture that powers GPT, BERT, 
          and modern AI. From attention mechanisms to multi-head magic.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="flex flex-wrap justify-center gap-4"
        >
          <Link
            href="#architecture"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold hover:shadow-lg hover:shadow-violet-500/25 transition-all duration-300 hover:scale-105"
          >
            <BookOpen className="w-5 h-5" />
            Start Learning
          </Link>
          <Link
            href="#playground"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-slate-800/80 border border-slate-700 text-white font-semibold hover:bg-slate-700/80 transition-all duration-300"
          >
            <Sparkles className="w-5 h-5" />
            Try Playground
          </Link>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto"
        >
          {[
            { value: "2017", label: "Year Introduced", icon: <BookOpen className="w-5 h-5" /> },
            { value: "170M", label: "Original Parameters", icon: <Layers className="w-5 h-5" /> },
            { value: "175B", label: "GPT-3 Parameters", icon: <Zap className="w-5 h-5" /> },
            { value: "∞", label: "Possibilities", icon: <Sparkles className="w-5 h-5" /> },
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 + i * 0.1 }}
              className="text-center p-4 rounded-2xl bg-slate-800/30 border border-slate-700/30"
            >
              <div className="flex justify-center mb-2 text-violet-400">{stat.icon}</div>
              <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
              <div className="text-sm text-slate-400">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="flex flex-col items-center gap-2 text-slate-500"
        >
          <span className="text-sm">Scroll to explore</span>
          <ChevronDown className="w-6 h-6" />
        </motion.div>
      </motion.div>
    </motion.section>
  );
};

// --- Architecture Overview ---
const ArchitectureSection = () => {
  const [activeLayer, setActiveLayer] = useState<string | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);

  const layers = [
    { id: "output", name: "Output Probabilities", color: "from-emerald-500 to-teal-500", description: "Final predictions for next token" },
    { id: "linear", name: "Linear + Softmax", color: "from-cyan-500 to-blue-500", description: "Project to vocabulary size and normalize" },
    { id: "decoder", name: "Decoder Stack (N×)", color: "from-violet-500 to-purple-500", description: "Multiple transformer blocks" },
    { id: "ffn", name: "Feed Forward", color: "from-amber-500 to-orange-500", description: "Position-wise fully connected" },
    { id: "addnorm2", name: "Add & Norm", color: "from-slate-500 to-gray-500", description: "Residual connection + Layer normalization" },
    { id: "multihead", name: "Multi-Head Attention", color: "from-rose-500 to-pink-500", description: "Parallel attention heads" },
    { id: "addnorm1", name: "Add & Norm", color: "from-slate-500 to-gray-500", description: "Residual connection + Layer normalization" },
    { id: "embedding", name: "Input Embedding", color: "from-indigo-500 to-blue-500", description: "Token to vector representation" },
    { id: "positional", name: "Positional Encoding", color: "from-fuchsia-500 to-pink-500", description: "Add position information" },
  ];

  const startAnimation = () => {
    setIsAnimating(true);
    setAnimationStep(0);
    let step = 0;
    const interval = setInterval(() => {
      step++;
      setAnimationStep(step);
      if (step >= layers.length) {
        clearInterval(interval);
        setTimeout(() => {
          setIsAnimating(false);
          setAnimationStep(0);
        }, 1000);
      }
    }, 800);
  };

  return (
    <section id="architecture" className="py-32 relative">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 text-violet-300 text-sm font-medium mb-4">
            <Layers className="w-4 h-4" />
            Core Architecture
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            The Transformer <span className="text-violet-400">Encoder-Decoder</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Understanding the building blocks of modern language models
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-start">
          {/* Interactive Architecture Diagram */}
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="relative"
          >
            <div className="bg-slate-900/50 backdrop-blur rounded-3xl p-8 border border-slate-700/50">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-semibold text-white">Architecture Flow</h3>
                <button
                  onClick={startAnimation}
                  disabled={isAnimating}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-600 text-white text-sm font-medium hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isAnimating ? "Running..." : "Animate Flow"}
                </button>
              </div>

              <div className="space-y-3">
                {layers.map((layer, index) => (
                  <motion.div
                    key={layer.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{
                      opacity: isAnimating ? (index <= animationStep ? 1 : 0.3) : 1,
                      x: 0,
                      scale: isAnimating && index === animationStep ? 1.02 : 1,
                    }}
                    transition={{ delay: index * 0.05 }}
                    onClick={() => setActiveLayer(activeLayer === layer.id ? null : layer.id)}
                    className={`relative p-4 rounded-xl cursor-pointer transition-all duration-300 ${
                      activeLayer === layer.id
                        ? "bg-slate-800 ring-2 ring-violet-500/50"
                        : "bg-slate-800/50 hover:bg-slate-800"
                    }`}
                  >
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${layer.color} flex items-center justify-center text-white font-bold shadow-lg`}>
                        {layers.length - index}
                      </div>
                      <div className="flex-1">
                        <h4 className="text-white font-semibold">{layer.name}</h4>
                        <AnimatePresence>
                          {(activeLayer === layer.id || isAnimating && index === animationStep) && (
                            <motion.p
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className="text-slate-400 text-sm mt-1"
                            >
                              {layer.description}
                            </motion.p>
                          )}
                        </AnimatePresence>
                      </div>
                      <ChevronRight className={`w-5 h-5 text-slate-500 transition-transform ${activeLayer === layer.id ? "rotate-90" : ""}`} />
                    </div>

                    {/* Connection Line */}
                    {index < layers.length - 1 && (
                      <motion.div
                        initial={{ scaleY: 0 }}
                        animate={{ scaleY: 1 }}
                        className="absolute left-10 bottom-0 w-0.5 h-3 bg-gradient-to-b from-slate-600 to-slate-700 -translate-y-full"
                      />
                    )}
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Detailed Info Panel */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="bg-gradient-to-br from-violet-900/20 to-indigo-900/20 rounded-3xl p-8 border border-violet-500/20">
              <h3 className="text-2xl font-bold text-white mb-4">Why Transformers?</h3>
              <div className="space-y-4">
                {[
                  { title: "Parallelization", desc: "Process all tokens simultaneously, unlike RNNs", icon: <Zap className="w-5 h-5 text-amber-400" /> },
                  { title: "Long-range Dependencies", desc: "Attention connects any two positions directly", icon: <Eye className="w-5 h-5 text-cyan-400" /> },
                  { title: "Transfer Learning", desc: "Pre-train once, fine-tune for any task", icon: <Brain className="w-5 h-5 text-violet-400" /> },
                ].map((item, i) => (
                  <motion.div
                    key={item.title}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.1 }}
                    className="flex items-start gap-4 p-4 rounded-xl bg-slate-800/50"
                  >
                    <div className="p-2 rounded-lg bg-slate-700/50">{item.icon}</div>
                    <div>
                      <h4 className="text-white font-semibold">{item.title}</h4>
                      <p className="text-slate-400 text-sm">{item.desc}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Key Innovation */}
            <div className="bg-slate-900/50 rounded-3xl p-8 border border-slate-700/50">
              <div className="flex items-center gap-3 mb-4">
                <Lightbulb className="w-6 h-6 text-amber-400" />
                <h3 className="text-xl font-bold text-white">Key Innovation</h3>
              </div>
              <blockquote className="text-slate-300 italic border-l-4 border-violet-500 pl-4">
                "The Transformer relies entirely on attention mechanisms to draw global dependencies between input and output, 
                dispensing with recurrence and convolutions entirely."
              </blockquote>
              <p className="text-slate-500 text-sm mt-2">— Attention Is All You Need (Vaswani et al., 2017)</p>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

// ============================================
// EMBEDDINGS & POSITIONAL ENCODING SECTION
// ============================================

const EmbeddingsSection = () => {
  const [tokens, setTokens] = useState<Token[]>(
    SAMPLE_TOKENS.map((text, i) => ({
      id: i,
      text,
      embedding: generateRandomEmbedding(EMBEDDING_DIM),
      position: i,
    }))
  );
  const [showPositional, setShowPositional] = useState(false);
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);

  const getPositionalEncoding = (pos: number, dim: number): number => {
    const angle = pos / Math.pow(10000, (2 * (dim % 2)) / EMBEDDING_DIM);
    return dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  };

  const getFinalEmbedding = (token: Token): number[] =>
    token.embedding.map((emb, i) => emb + (showPositional ? getPositionalEncoding(token.position, i) * 0.5 : 0));

  return (
    <section id="embeddings" className="py-32 relative bg-slate-950/50">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 text-indigo-300 text-sm font-medium mb-4">
            <Matrix className="w-4 h-4" />
            Input Representation
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Token <span className="text-indigo-400">Embeddings</span> & Position
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            How words become numbers that computers understand
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Interactive Embedding Visualizer */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="bg-slate-900/50 backdrop-blur rounded-3xl p-8 border border-slate-700/50"
          >
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold text-white">Embedding Visualization</h3>
              <button
                onClick={() => setShowPositional(!showPositional)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  showPositional
                    ? "bg-indigo-600 text-white"
                    : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                }`}
              >
                {showPositional ? "Hide" : "Show"} Positional Encoding
              </button>
            </div>

            <div className="space-y-4">
              {tokens.map((token, tokenIdx) => (
                <motion.div
                  key={token.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: tokenIdx * 0.1 }}
                  onMouseEnter={() => setHoveredToken(tokenIdx)}
                  onMouseLeave={() => setHoveredToken(null)}
                  className={`p-4 rounded-xl transition-all duration-300 ${
                    hoveredToken === tokenIdx
                      ? "bg-slate-800 ring-2 ring-indigo-500/50"
                      : "bg-slate-800/50"
                  }`}
                >
                  <div className="flex items-center gap-4">
                    {/* Token */}
                    <div className="w-24 px-3 py-2 bg-indigo-500/20 rounded-lg text-indigo-300 font-mono text-sm text-center">
                      {token.text}
                    </div>

                    {/* Position */}
                    <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center text-slate-400 font-mono text-sm">
                      {token.position}
                    </div>

                    {/* Embedding Vector */}
                    <div className="flex-1 flex gap-1">
                      {getFinalEmbedding(token).map((val, i) => {
                        const intensity = Math.abs(val);
                        const isPositive = val >= 0;
                        return (
                          <motion.div
                            key={i}
                            initial={{ scaleY: 0 }}
                            animate={{ scaleY: 1 }}
                            transition={{ delay: tokenIdx * 0.1 + i * 0.02 }}
                            className="flex-1 h-12 rounded relative overflow-hidden group"
                          >
                            <div
                              className={`absolute bottom-0 left-0 right-0 transition-all duration-300 ${
                                isPositive
                                  ? "bg-gradient-to-t from-indigo-600 to-indigo-400"
                                  : "bg-gradient-to-b from-rose-600 to-rose-400"
                              }`}
                              style={{ height: `${intensity * 50}%`, top: isPositive ? "auto" : "50%", bottom: isPositive ? "50%" : "auto" }}
                            />
                            <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                              <span className="text-[10px] font-mono text-white bg-slate-900/80 px-1 rounded">
                                {formatNumber(val)}
                              </span>
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Positional Encoding Formula */}
                  <AnimatePresence>
                    {showPositional && hoveredToken === tokenIdx && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="mt-3 pt-3 border-t border-slate-700/50 overflow-hidden"
                      >
                        <p className="text-xs text-slate-400 font-mono mb-2">
                          PE(pos, 2i) = sin(pos / 10000^(2i/d))
                        </p>
                        <div className="flex gap-2">
                          {Array.from({ length: EMBEDDING_DIM }).map((_, i) => {
                            const pe = getPositionalEncoding(token.position, i);
                            return (
                              <div
                                key={i}
                                className={`flex-1 h-8 rounded flex items-center justify-center text-[9px] font-mono ${
                                  pe >= 0 ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"
                                }`}
                              >
                                {formatNumber(pe, 1)}
                              </div>
                            );
                          })}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Explanation */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="bg-gradient-to-br from-indigo-900/20 to-purple-900/20 rounded-3xl p-8 border border-indigo-500/20">
              <h3 className="text-2xl font-bold text-white mb-6">How It Works</h3>
              
              <div className="space-y-6">
                <div className="flex gap-4">
                  <div className="w-12 h-12 rounded-xl bg-indigo-500/20 flex items-center justify-center shrink-0">
                    <span className="text-2xl">🔤</span>
                  </div>
                  <div>
                    <h4 className="text-white font-semibold mb-1">1. Tokenization</h4>
                    <p className="text-slate-400 text-sm">
                      Text is split into tokens (words or subwords). Each token gets a unique ID.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="w-12 h-12 rounded-xl bg-violet-500/20 flex items-center justify-center shrink-0">
                    <span className="text-2xl">📊</span>
                  </div>
                  <div>
                    <h4 className="text-white font-semibold mb-1">2. Embedding Lookup</h4>
                    <p className="text-slate-400 text-sm">
                      Each token ID maps to a high-dimensional vector (typically 512-2048 dimensions) 
                      that captures semantic meaning.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="w-12 h-12 rounded-xl bg-fuchsia-500/20 flex items-center justify-center shrink-0">
                    <span className="text-2xl">📍</span>
                  </div>
                  <div>
                    <h4 className="text-white font-semibold mb-1">3. Positional Encoding</h4>
                    <p className="text-slate-400 text-sm">
                      Since transformers process all tokens in parallel, we add position information 
                      using sine and cosine functions.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Formula Card */}
            <div className="bg-slate-900/50 rounded-3xl p-8 border border-slate-700/50">
              <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Code className="w-5 h-5 text-violet-400" />
                Positional Encoding Formula
              </h4>
              <div className="space-y-3 font-mono text-sm">
                <div className="p-4 bg-slate-950 rounded-xl">
                  <p className="text-emerald-400 mb-2"># Even indices (2i)</p>
                  <p className="text-slate-300">PE(pos, 2i) = sin(pos / 10000^(2i/d_model))</p>
                </div>
                <div className="p-4 bg-slate-950 rounded-xl">
                  <p className="text-amber-400 mb-2"># Odd indices (2i+1)</p>
                  <p className="text-slate-300">PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</p>
                </div>
              </div>
              <p className="text-slate-500 text-sm mt-4">
                This allows the model to learn relative positions easily, as PE(pos+k) 
                can be represented as a linear function of PE(pos).
              </p>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

// ============================================
// SELF-ATTENTION SECTION
// ============================================

const AttentionSection = () => {
  const [inputTokens] = useState(["The", "cat", "sat", "on", "the", "mat"]);
  const [selectedQuery, setSelectedQuery] = useState(0);
  const [animationPhase, setAnimationPhase] = useState<"idle" | "query" | "key" | "score" | "softmax" | "value">("idle");
  const [isPlaying, setIsPlaying] = useState(false);

  // Simulated attention scores
  const attentionScores = [
    [0.35, 0.15, 0.20, 0.10, 0.12, 0.08],  // The
    [0.10, 0.45, 0.15, 0.12, 0.10, 0.08],  // cat
    [0.08, 0.12, 0.50, 0.15, 0.08, 0.07],  // sat
    [0.05, 0.08, 0.12, 0.55, 0.10, 0.10],  // on
    [0.15, 0.10, 0.08, 0.12, 0.35, 0.20],  // the
    [0.08, 0.05, 0.05, 0.10, 0.22, 0.50],  // mat
  ];

  const startAnimation = () => {
    setIsPlaying(true);
    setAnimationPhase("query");
    const phases: Array<"idle" | "query" | "key" | "score" | "softmax" | "value"> = ["query", "key", "score", "softmax", "value"];
    let currentPhase = 0;
    
    const interval = setInterval(() => {
      currentPhase++;
      if (currentPhase >= phases.length) {
        clearInterval(interval);
        setIsPlaying(false);
        setAnimationPhase("idle");
      } else {
        setAnimationPhase(phases[currentPhase]);
      }
    }, 1500);
  };

  const getCellColor = (score: number): string => {
    if (score > 0.4) return "bg-emerald-500";
    if (score > 0.25) return "bg-emerald-400";
    if (score > 0.15) return "bg-emerald-300";
    return "bg-emerald-200";
  };

  return (
    <section id="attention" className="py-32 relative">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 text-amber-300 text-sm font-medium mb-4">
            <Eye className="w-4 h-4" />
            Core Mechanism
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Self-<span className="text-amber-400">Attention</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            The heart of the transformer: allowing each token to attend to all other tokens
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-5 gap-8">
          {/* Attention Matrix Visualization */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="lg:col-span-3 bg-slate-900/50 backdrop-blur rounded-3xl p-8 border border-slate-700/50"
          >
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold text-white">Attention Weights Matrix</h3>
              <button
                onClick={startAnimation}
                disabled={isPlaying}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-600 text-white text-sm font-medium hover:bg-amber-500 disabled:opacity-50 transition-colors"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isPlaying ? "Playing..." : "Watch Animation"}
              </button>
            </div>

            {/* Matrix */}
            <div className="overflow-x-auto">
              <div className="inline-block">
                {/* Header */}
                <div className="flex mb-2">
                  <div className="w-20" /> {/* Corner spacer */}
                  {inputTokens.map((token, i) => (
                    <div key={i} className="w-16 text-center text-xs text-slate-500 font-mono">
                      {token}
                    </div>
                  ))}
                </div>

                {/* Rows */}
                <div className="space-y-1">
                  {inputTokens.map((token, rowIdx) => (
                    <div key={rowIdx} className="flex items-center">
                      <button
                        onClick={() => setSelectedQuery(rowIdx)}
                        className={`w-20 text-right pr-3 text-xs font-mono py-3 rounded-l-lg transition-colors ${
                          selectedQuery === rowIdx
                            ? "bg-amber-500/20 text-amber-300"
                            : "text-slate-400 hover:text-white"
                        }`}
                      >
                        {token}
                      </button>
                      <div className="flex gap-1">
                        {attentionScores[rowIdx].map((score, colIdx) => (
                          <motion.div
                            key={colIdx}
                            initial={{ opacity: 0, scale: 0 }}
                            animate={{
                              opacity: selectedQuery === rowIdx ? 1 : 0.4,
                              scale: 1,
                            }}
                            transition={{ delay: (rowIdx * 6 + colIdx) * 0.01 }}
                            className={`w-16 h-12 rounded flex items-center justify-center text-xs font-mono font-bold transition-all duration-300 ${
                              selectedQuery === rowIdx
                                ? `${getCellColor(score)} text-slate-900`
                                : "bg-slate-800 text-slate-500"
                            }`}
                          >
                            {formatNumber(score)}
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Selected Row Detail */}
            <div className="mt-6 p-4 bg-slate-800/50 rounded-xl">
              <p className="text-sm text-slate-400 mb-3">
                <span className="text-amber-400 font-semibold">{inputTokens[selectedQuery]}</span> attends to:
              </p>
              <div className="flex flex-wrap gap-2">
                {inputTokens.map((token, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-700/50"
                  >
                    <span className="text-slate-300 text-sm">{token}</span>
                    <div
                      className={`w-2 h-2 rounded-full ${getCellColor(attentionScores[selectedQuery][i])}`}
                    />
                    <span className="text-xs font-mono text-slate-400">
                      {(attentionScores[selectedQuery][i] * 100).toFixed(0)}%
                    </span>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Formula & Explanation */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Attention Formula */}
            <div className="bg-gradient-to-br from-amber-900/20 to-orange-900/20 rounded-3xl p-6 border border-amber-500/20">
              <h4 className="text-lg font-bold text-white mb-4">Attention Formula</h4>
              <div className="p-4 bg-slate-950 rounded-xl font-mono text-lg text-center">
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
              
              <div className="mt-4 space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded bg-violet-500/20 text-violet-400 flex items-center justify-center font-mono font-bold">Q</span>
                  <span className="text-slate-400">Query matrix</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded bg-rose-500/20 text-rose-400 flex items-center justify-center font-mono font-bold">K</span>
                  <span className="text-slate-400">Key matrix</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-mono font-bold">V</span>
                  <span className="text-slate-400">Value matrix</span>
                </div>
              </div>
            </div>

            {/* Steps */}
            <div className="bg-slate-900/50 rounded-3xl p-6 border border-slate-700/50">
              <h4 className="text-lg font-bold text-white mb-4">How It Works</h4>
              <div className="space-y-3">
                {[
                  { step: 1, title: "Create Q, K, V", desc: "Linear projections from input", active: animationPhase === "query" || animationPhase === "idle" },
                  { step: 2, title: "Compute Scores", desc: "Q × K<sup>T</sup> for similarity", active: animationPhase === "score" },
                  { step: 3, title: "Scale & Softmax", desc: "Divide by √d<sub>k</sub>, then normalize", active: animationPhase === "softmax" },
                  { step: 4, title: "Weighted Sum", desc: "Multiply by V for output", active: animationPhase === "value" },
                ].map((item) => (
                  <motion.div
                    key={item.step}
                    animate={{
                      backgroundColor: item.active ? "rgba(245, 158, 11, 0.1)" : "rgba(30, 41, 59, 0.5)",
                      borderColor: item.active ? "rgba(245, 158, 11, 0.3)" : "transparent",
                    }}
                    className="flex items-start gap-3 p-3 rounded-xl border"
                  >
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                      item.active ? "bg-amber-500 text-white" : "bg-slate-700 text-slate-400"
                    }`}>
                      {item.step}
                    </div>
                    <div>
                      <h5 className={`font-semibold ${item.active ? "text-amber-400" : "text-white"}`}>
                        {item.title}
                      </h5>
                      <p className="text-slate-500 text-sm" dangerouslySetInnerHTML={{ __html: item.desc }} />
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

// ============================================
// MULTI-HEAD ATTENTION SECTION
// ============================================

const MultiHeadAttentionSection = () => {
  const [numHeads, setNumHeads] = useState(4);
  const [activeHead, setActiveHead] = useState(0);
  const [showConcat, setShowConcat] = useState(false);

  const heads = Array.from({ length: numHeads }, (_, i) => ({
    id: i,
    name: `Head ${i + 1}`,
    focus: ["Syntax", "Semantics", "Long-range", "Local"][i % 4],
    color: ["from-violet-500 to-purple-500", "from-amber-500 to-orange-500", "from-emerald-500 to-teal-500", "from-rose-500 to-pink-500"][i % 4],
  }));

  return (
    <section id="multihead" className="py-32 relative bg-slate-950/50">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 text-violet-300 text-sm font-medium mb-4">
            <Brain className="w-4 h-4" />
            Parallel Processing
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Multi-Head <span className="text-violet-400">Attention</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Multiple attention heads working in parallel, each focusing on different aspects
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Heads Visualization */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="lg:col-span-2 bg-slate-900/50 backdrop-blur rounded-3xl p-8 border border-slate-700/50"
          >
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold text-white">Attention Heads</h3>
              <div className="flex items-center gap-4">
                <label className="text-sm text-slate-400">Heads:</label>
                <div className="flex gap-2">
                  {[2, 4, 8].map((n) => (
                    <button
                      key={n}
                      onClick={() => { setNumHeads(n); setActiveHead(0); }}
                      className={`w-10 h-10 rounded-lg font-semibold transition-all ${
                        numHeads === n
                          ? "bg-violet-600 text-white"
                          : "bg-slate-700 text-slate-400 hover:bg-slate-600"
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Heads Grid */}
            <div className={`grid gap-4 ${numHeads <= 4 ? "grid-cols-2" : "grid-cols-4"}`}>
              {heads.map((head, i) => (
                <motion.button
                  key={head.id}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.05 }}
                  onClick={() => setActiveHead(i)}
                  className={`relative p-6 rounded-2xl transition-all duration-300 ${
                    activeHead === i
                      ? "ring-2 ring-violet-500 scale-105"
                      : "hover:scale-102"
                  }`}
                >
                  <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${head.color} opacity-20`} />
                  <div className="relative">
                    <div className={`w-12 h-12 mx-auto mb-3 rounded-xl bg-gradient-to-br ${head.color} flex items-center justify-center text-white text-xl`}>
                      <Brain className="w-6 h-6" />
                    </div>
                    <h4 className="text-white font-semibold text-center">{head.name}</h4>
                    <p className="text-slate-400 text-xs text-center mt-1">{head.focus}</p>
                  </div>
                  
                  {activeHead === i && (
                    <motion.div
                      layoutId="activeHead"
                      className={`absolute inset-0 rounded-2xl border-2 border-violet-500`}
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                  )}
                </motion.button>
              ))}
            </div>

            {/* Concatenation Visualization */}
            <div className="mt-8 p-6 bg-slate-800/50 rounded-2xl">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-white font-semibold">Output Concatenation</h4>
                <button
                  onClick={() => setShowConcat(!showConcat)}
                  className="text-sm text-violet-400 hover:text-violet-300"
                >
                  {showConcat ? "Hide" : "Show"} Process
                </button>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="flex gap-2">
                  {heads.map((head, i) => (
                    <motion.div
                      key={head.id}
                      animate={{
                        opacity: activeHead === i ? 1 : 0.5,
                        scale: activeHead === i ? 1.05 : 1,
                      }}
                      className={`w-8 h-24 rounded bg-gradient-to-b ${head.color}`}
                    />
                  ))}
                </div>
                
                <ArrowRight className="w-6 h-6 text-slate-500" />
                
                <motion.div
                  animate={{ width: showConcat ? 200 : 100 }}
                  className="h-24 rounded bg-gradient-to-r from-violet-600 to-indigo-600 flex items-center justify-center"
                >
                  <span className="text-white font-mono text-sm">Concat → Linear</span>
                </motion.div>
              </div>
              
              <AnimatePresence>
                {showConcat && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="mt-4 p-4 bg-slate-950 rounded-xl overflow-hidden"
                  >
                    <code className="text-sm text-slate-300">
                      <span className="text-violet-400">MultiHead</span>(Q, K, V) = <span className="text-emerald-400">Concat</span>(head₁, ..., headₙ)W<sup>O</sup>
                    </code>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>

          {/* Benefits */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="bg-gradient-to-br from-violet-900/20 to-purple-900/20 rounded-3xl p-6 border border-violet-500/20">
              <h3 className="text-xl font-bold text-white mb-6">Why Multiple Heads?</h3>
              
              <div className="space-y-4">
                {[
                  { icon: "🎯", title: "Different Focus", desc: "Each head learns different relationships" },
                  { icon: "🔍", title: "Multiple Subspaces", desc: "Attend to information from different representation spaces" },
                  { icon: "⚡", title: "Ensemble Effect", desc: "Combined heads are more robust than single attention" },
                  { icon: "🧠", title: "Richer Context", desc: "Capture various types of dependencies simultaneously" },
                ].map((item, i) => (
                  <motion.div
                    key={item.title}
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.1 }}
                    className="flex items-start gap-3"
                  >
                    <span className="text-2xl">{item.icon}</span>
                    <div>
                      <h4 className="text-white font-semibold">{item.title}</h4>
                      <p className="text-slate-400 text-sm">{item.desc}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            <div className="bg-slate-900/50 rounded-3xl p-6 border border-slate-700/50">
              <h4 className="text-lg font-bold text-white mb-4">Common Configurations</h4>
              <div className="space-y-3">
                {[
                  { model: "BERT-Base", heads: 12, dims: "64d per head" },
                  { model: "GPT-3", heads: 96, dims: "128d per head" },
                  { model: "LLaMA-2", heads: 32, dims: "128d per head" },
                ].map((config) => (
                  <div key={config.model} className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-white font-medium">{config.model}</span>
                    <div className="text-right">
                      <span className="text-violet-400 font-mono text-sm">{config.heads} heads</span>
                      <span className="text-slate-500 text-xs ml-2">{config.dims}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

// ============================================
// FEED-FORWARD NETWORK SECTION
// ============================================

const FeedForwardSection = () => {
  const [hiddenSize, setHiddenSize] = useState(4);
  const [activation, setActivation] = useState<"relu" | "gelu">("gelu");
  const [isAnimating, setIsAnimating] = useState(false);

  const activations = {
    relu: (x: number) => Math.max(0, x),
    gelu: (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))),
  };

  const sampleValues = [-2, -1, -0.5, 0, 0.5, 1, 2];

  return (
    <section id="ffn" className="py-32 relative">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 text-emerald-300 text-sm font-medium mb-4">
            <Zap className="w-4 h-4" />
            Position-wise Processing
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Feed-Forward <span className="text-emerald-400">Network</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            The transformation applied to each position independently
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Architecture Diagram */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="bg-slate-900/50 backdrop-blur rounded-3xl p-8 border border-slate-700/50"
          >
            <h3 className="text-xl font-semibold text-white mb-6">FFN Architecture</h3>

            {/* Visual Representation */}
            <div className="flex items-center justify-center gap-8 py-8">
              {/* Input */}
              <div className="flex flex-col items-center gap-2">
                <div className="w-16 h-32 rounded-xl bg-slate-700 flex flex-col items-center justify-center gap-1">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="w-10 h-6 rounded bg-slate-600" />
                  ))}
                </div>
                <span className="text-sm text-slate-400">Input (d<sub>model</sub>)</span>
              </div>

              <ArrowRight className="w-6 h-6 text-slate-500" />

              {/* Expand */}
              <div className="flex flex-col items-center gap-2">
                <div className={`rounded-xl bg-gradient-to-b from-emerald-600 to-teal-600 flex flex-col items-center justify-center gap-1 p-2 transition-all duration-500`}>
                  {Array.from({ length: Math.max(4, hiddenSize * 2) }).map((_, i) => (
                    <motion.div
                      key={i}
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      className="w-12 h-4 rounded bg-white/20"
                    />
                  ))}
                </div>
                <span className="text-sm text-slate-400">Expand ({hiddenSize}×)</span>
              </div>

              <ArrowRight className="w-6 h-6 text-slate-500" />

              {/* Contract */}
              <div className="flex flex-col items-center gap-2">
                <div className="w-16 h-32 rounded-xl bg-slate-700 flex flex-col items-center justify-center gap-1">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="w-10 h-6 rounded bg-slate-600" />
                  ))}
                </div>
                <span className="text-sm text-slate-400">Output (d<sub>model</sub>)</span>
              </div>
            </div>

            {/* Controls */}
            <div className="mt-6 p-4 bg-slate-800/50 rounded-xl">
              <div className="flex items-center justify-between">
                <label className="text-sm text-slate-400">Expansion Factor:</label>
                <div className="flex gap-2">
                  {[2, 4, 8].map((factor) => (
                    <button
                      key={factor}
                      onClick={() => setHiddenSize(factor)}
                      className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                        hiddenSize === factor
                          ? "bg-emerald-600 text-white"
                          : "bg-slate-700 text-slate-400 hover:bg-slate-600"
                      }`}
                    >
                      {factor}×
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Formula */}
            <div className="mt-6 p-4 bg-slate-950 rounded-xl">
              <code className="text-lg text-slate-300">
                FFN(x) = <span className="text-emerald-400">activation</span>(xW₁ + b₁)W₂ + b₂
              </code>
            </div>
          </motion.div>

          {/* Activation Visualization */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="bg-slate-900/50 backdrop-blur rounded-3xl p-8 border border-slate-700/50">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-semibold text-white">Activation Functions</h3>
                <div className="flex gap-2 p-1 bg-slate-800 rounded-lg">
                  {(["relu", "gelu"] as const).map((act) => (
                    <button
                      key={act}
                      onClick={() => setActivation(act)}
                      className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-all ${
                        activation === act
                          ? "bg-emerald-600 text-white"
                          : "text-slate-400 hover:text-white"
                      }`}
                    >
                      {act}
                    </button>
                  ))}
                </div>
              </div>

              {/* Activation Graph */}
              <div className="relative h-48 bg-slate-950 rounded-xl overflow-hidden">
                <svg className="absolute inset-0 w-full h-full">
                  {/* Grid */}
                  <line x1="50%" y1="0" x2="50%" y2="100%" stroke="#334155" strokeWidth="1" />
                  <line x1="0" y1="50%" x2="100%" y2="50%" stroke="#334155" strokeWidth="1" />
                  
                  {/* Function curve */}
                  <path
                    d={sampleValues
                      .map((x, i) => {
                        const y = activations[activation](x);
                        const px = 50 + (x / 2.5) * 40;
                        const py = 50 - (y / 2.5) * 40;
                        return `${i === 0 ? "M" : "L"} ${px}% ${py}%`;
                      })
                      .join(" ")}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="3"
                  />
                </svg>
                
                {/* Labels */}
                <div className="absolute bottom-2 left-2 text-xs text-slate-500">-2.5</div>
                <div className="absolute bottom-2 right-2 text-xs text-slate-500">2.5</div>
              </div>

              {/* Values Table */}
              <div className="mt-4 grid grid-cols-7 gap-2">
                {sampleValues.map((x) => (
                  <div key={x} className="text-center">
                    <div className="text-xs text-slate-500 mb-1">{x}</div>
                    <div className="text-xs font-mono text-emerald-400">
                      {formatNumber(activations[activation](x))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Info Cards */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-slate-800/50 rounded-xl">
                <h4 className="text-rose-400 font-semibold mb-2">ReLU</h4>
                <code className="text-sm text-slate-400">f(x) = max(0, x)</code>
                <p className="text-slate-500 text-xs mt-2">Simple, fast, but can cause "dying ReLU"</p>
              </div>
              <div className="p-4 bg-slate-800/50 rounded-xl">
                <h4 className="text-emerald-400 font-semibold mb-2">GELU</h4>
                <code className="text-sm text-slate-400">Smoother alternative</code>
                <p className="text-slate-500 text-xs mt-2">Used in BERT, GPT. Smoother gradients</p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

// ============================================
// CODE IMPLEMENTATION SECTION
// ============================================

const CodeSection = () => {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"pytorch" | "tensorflow" | "numpy">("pytorch");

  const codeExamples = {
    pytorch: `import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply final linear
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        return self.W_o(attn_output)

# Usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)
output = mha(x, x, x)  # Self-attention
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")`,

    tensorflow: `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model=512, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
    def build(self, input_shape):
        self.W_q = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="W_q"
        )
        self.W_k = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform", 
            trainable=True,
            name="W_k"
        )
        self.W_v = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="W_v"
        )
        self.W_o = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="W_o"
        )
        super().build(input_shape)
    
    def call(self, inputs, mask=None):
        query, key, value = inputs, inputs, inputs
        batch_size = tf.shape(query)[0]
        
        # Linear projections
        Q = tf.matmul(query, self.W_q)
        K = tf.matmul(key, self.W_k)
        V = tf.matmul(value, self.W_v)
        
        # Reshape for multi-head attention
        Q = tf.reshape(Q, [batch_size, -1, self.num_heads, self.d_k])
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.reshape(K, [batch_size, -1, self.num_heads, self.d_k])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.reshape(V, [batch_size, -1, self.num_heads, self.d_k])
        V = tf.transpose(V, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(float(self.d_k))
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_output = tf.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [batch_size, -1, self.d_model])
        
        return tf.matmul(attn_output, self.W_o)

# Usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = tf.random.normal((2, 100, 512))
output = mha(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")`,

    numpy: `import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch_size, num_heads, seq_len, d_k)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, V)
    return output, attn_weights

def multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads=8):
    """
    x: (batch_size, seq_len, d_model)
    W_*: (d_model, d_model) weight matrices
    """
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    # Linear projections
    Q = np.matmul(x, W_q)
    K = np.matmul(x, W_k)
    V = np.matmul(x, W_v)
    
    # Reshape for multi-head: (batch, heads, seq, d_k)
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Apply attention
    attn_output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    # Concatenate heads
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # Final linear projection
    return np.matmul(attn_output, W_o)

# Usage example
batch_size, seq_len, d_model = 2, 100, 512
num_heads = 8

# Initialize random weights
scale = np.sqrt(2.0 / d_model)
W_q = np.random.randn(d_model, d_model) * scale
W_k = np.random.randn(d_model, d_model) * scale
W_v = np.random.randn(d_model, d_model) * scale
W_o = np.random.randn(d_model, d_model) * scale

# Forward pass
x = np.random.randn(batch_size, seq_len, d_model)
output = multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")`,
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(codeExamples[activeTab]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="code" className="py-32 relative bg-slate-950/50">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 text-cyan-300 text-sm font-medium mb-4">
            <Terminal className="w-4 h-4" />
            Implementation
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Code <span className="text-cyan-400">Examples</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Production-ready implementations in popular deep learning frameworks
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-slate-900/50 backdrop-blur rounded-3xl border border-slate-700/50 overflow-hidden"
        >
          {/* Tabs */}
          <div className="flex items-center justify-between px-6 py-4 bg-slate-800/50 border-b border-slate-700/50">
            <div className="flex gap-2">
              {(["pytorch", "tensorflow", "numpy"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-all ${
                    activeTab === tab
                      ? "bg-cyan-600 text-white"
                      : "text-slate-400 hover:text-white hover:bg-slate-700"
                  }`}
                >
                  {tab === "pytorch" && "PyTorch"}
                  {tab === "tensorflow" && "TensorFlow"}
                  {tab === "numpy" && "NumPy"}
                </button>
              ))}
            </div>
            <button
              onClick={handleCopy}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-700 transition-all"
            >
              {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              {copied ? "Copied!" : "Copy"}
            </button>
          </div>

          {/* Code */}
          <div className="relative">
            <pre className="p-6 overflow-x-auto text-sm font-mono leading-relaxed">
              <code className="text-slate-300">
                {codeExamples[activeTab].split("\n").map((line, i) => (
                  <div key={i} className="table-row">
                    <span className="table-cell text-slate-600 select-none pr-4 text-right w-12">
                      {i + 1}
                    </span>
                    <span
                      className="table-cell"
                      dangerouslySetInnerHTML={{
                        __html: line
                          .replace(/(#.*$)/gm, '<span class="text-slate-500">$1</span>')
                          .replace(/\b(import|from|class|def|return|if|else|for|in|assert|super)\b/g, '<span class="text-violet-400">$1</span>')
                          .replace(/\b(self|None|True|False)\b/g, '<span class="text-amber-400">$1</span>')
                          .replace(/\b(nn|torch|tf|np|math)\b/g, '<span class="text-cyan-400">$1</span>')
                          .replace(/(".*?"|'.*?')/g, '<span class="text-emerald-400">$1</span>')
                          .replace(/\b(\d+)\b/g, '<span class="text-rose-400">$1</span>')
                      }}
                    />
                  </div>
                ))}
              </code>
            </pre>
          </div>
        </motion.div>

        {/* Additional Resources */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="mt-12 grid md:grid-cols-3 gap-6"
        >
          {[
            {
              title: "Original Paper",
              desc: "Attention Is All You Need (Vaswani et al., 2017)",
              link: "https://arxiv.org/abs/1706.03762",
              icon: <BookOpen className="w-5 h-5" />,
            },
            {
              title: "Hugging Face",
              desc: "Pre-trained models and transformers library",
              link: "https://huggingface.co/docs/transformers",
              icon: <Sparkles className="w-5 h-5" />,
            },
            {
              title: "The Illustrated Transformer",
              desc: "Jay Alammar's visual guide",
              link: "http://jalammar.github.io/illustrated-transformer/",
              icon: <Eye className="w-5 h-5" />,
            },
          ].map((resource) => (
            <a
              key={resource.title}
              href={resource.link}
              target="_blank"
              rel="noopener noreferrer"
              className="group p-6 bg-slate-800/50 rounded-2xl border border-slate-700/50 hover:border-cyan-500/50 transition-all"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="p-3 rounded-xl bg-cyan-500/10 text-cyan-400 group-hover:bg-cyan-500/20 transition-colors">
                  {resource.icon}
                </div>
                <ArrowRight className="w-5 h-5 text-slate-500 group-hover:text-cyan-400 group-hover:translate-x-1 transition-all" />
              </div>
              <h3 className="text-white font-semibold mb-1">{resource.title}</h3>
              <p className="text-slate-400 text-sm">{resource.desc}</p>
            </a>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

// ============================================
// INTERACTIVE PLAYGROUND SECTION
// ============================================

const PlaygroundSection = () => {
  const [inputText, setInputText] = useState("The cat sat on the mat");
  const [tokens, setTokens] = useState<string[]>(["The", "cat", "sat", "on", "the", "mat"]);
  const [selectedToken, setSelectedToken] = useState(0);
  const [numHeads, setNumHeads] = useState(4);
  const [temperature, setTemperature] = useState(1.0);
  const [isComputing, setIsComputing] = useState(false);

  // Generate synthetic attention weights based on position and randomness
  const generateAttentionWeights = (queryIdx: number, headIdx: number): number[] => {
    const weights = tokens.map((_, i) => {
      // Base similarity based on position proximity
      const proximity = Math.exp(-Math.abs(i - queryIdx) / 2);
      // Add head-specific bias
      const headBias = Math.sin((i + headIdx) * 0.5) * 0.3;
      // Add temperature-based randomness
      const randomness = (Math.random() - 0.5) * temperature;
      return Math.max(0.01, proximity + headBias * 0.5 + randomness * 0.2);
    });
    // Normalize
    const sum = weights.reduce((a, b) => a + b, 0);
    return weights.map((w) => w / sum);
  };

  const handleTokenize = () => {
    setIsComputing(true);
    setTimeout(() => {
      // Simple tokenization (split by space)
      const newTokens = inputText.trim().split(/\s+/).filter(Boolean);
      setTokens(newTokens.length > 0 ? newTokens : ["<empty>"]);
      setSelectedToken(0);
      setIsComputing(false);
    }, 500);
  };

  const attentionData = Array.from({ length: numHeads }, (_, headIdx) =>
    Array.from({ length: tokens.length }, (_, queryIdx) =>
      generateAttentionWeights(queryIdx, headIdx)
    )
  );

  const headColors = [
    "from-violet-500 to-purple-500",
    "from-amber-500 to-orange-500",
    "from-emerald-500 to-teal-500",
    "from-rose-500 to-pink-500",
    "from-cyan-500 to-blue-500",
    "from-fuchsia-500 to-violet-500",
    "from-lime-500 to-green-500",
    "from-red-500 to-rose-500",
  ];

  return (
    <section id="playground" className="py-32 relative">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-violet-500/20 to-pink-500/20 text-violet-300 text-sm font-medium mb-4">
            <Sparkles className="w-4 h-4" />
            Interactive Demo
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Attention <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-pink-400">Playground</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Experiment with attention patterns in real-time
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Controls Panel */}
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="bg-slate-900/50 backdrop-blur rounded-3xl p-6 border border-slate-700/50">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-violet-400" />
                Configuration
              </h3>

              {/* Input Text */}
              <div className="mb-6">
                <label className="block text-sm text-slate-400 mb-2">Input Text</label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="w-full p-3 bg-slate-800 rounded-xl text-white text-sm resize-none focus:outline-none focus:ring-2 focus:ring-violet-500"
                  rows={3}
                />
                <button
                  onClick={handleTokenize}
                  disabled={isComputing}
                  className="mt-2 w-full py-2 bg-violet-600 text-white rounded-lg text-sm font-medium hover:bg-violet-500 disabled:opacity-50 transition-colors"
                >
                  {isComputing ? "Tokenizing..." : "Tokenize"}
                </button>
              </div>

              {/* Number of Heads */}
              <div className="mb-6">
                <label className="block text-sm text-slate-400 mb-2">
                  Attention Heads: <span className="text-white">{numHeads}</span>
                </label>
                <input
                  type="range"
                  min={1}
                  max={8}
                  value={numHeads}
                  onChange={(e) => setNumHeads(parseInt(e.target.value))}
                  className="w-full accent-violet-500"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>1</span>
                  <span>8</span>
                </div>
              </div>

              {/* Temperature */}
              <div className="mb-4">
                <label className="block text-sm text-slate-400 mb-2">
                  Randomness: <span className="text-white">{temperature.toFixed(1)}</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-violet-500"
                />
              </div>
            </div>

            {/* Token Selector */}
            <div className="bg-slate-900/50 backdrop-blur rounded-3xl p-6 border border-slate-700/50">
              <h3 className="text-lg font-semibold text-white mb-4">Select Query Token</h3>
              <div className="flex flex-wrap gap-2">
                {tokens.map((token, i) => (
                  <button
                    key={i}
                    onClick={() => setSelectedToken(i)}
                    className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                      selectedToken === i
                        ? "bg-violet-600 text-white"
                        : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                    }`}
                  >
                    {token}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Attention Visualization */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="lg:col-span-2 bg-slate-900/50 backdrop-blur rounded-3xl p-6 border border-slate-700/50"
          >
            <h3 className="text-lg font-semibold text-white mb-6">
              Attention Patterns for <span className="text-violet-400">"{tokens[selectedToken]}"</span>
            </h3>

            <div className={`grid gap-4 ${numHeads <= 4 ? "grid-cols-2" : "grid-cols-4"}`}>
              {Array.from({ length: numHeads }, (_, headIdx) => (
                <motion.div
                  key={headIdx}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: headIdx * 0.05 }}
                  className="p-4 bg-slate-800/50 rounded-2xl"
                >
                  <div className="flex items-center gap-2 mb-3">
                    <div className={`w-3 h-3 rounded-full bg-gradient-to-r ${headColors[headIdx % headColors.length]}`} />
                    <span className="text-sm text-slate-300">Head {headIdx + 1}</span>
                  </div>

                  {/* Attention bars */}
                  <div className="space-y-2">
                    {tokens.map((token, tokenIdx) => {
                      const weight = attentionData[headIdx][selectedToken][tokenIdx];
                      return (
                        <div key={tokenIdx} className="flex items-center gap-2">
                          <span className="w-12 text-xs text-slate-500 truncate">{token}</span>
                          <div className="flex-1 h-6 bg-slate-700/50 rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${weight * 100}%` }}
                              transition={{ duration: 0.5, delay: tokenIdx * 0.05 }}
                              className={`h-full bg-gradient-to-r ${headColors[headIdx % headColors.length]}`}
                            />
                          </div>
                          <span className="w-10 text-xs text-slate-400 text-right">
                            {(weight * 100).toFixed(0)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Attention Matrix */}
            <div className="mt-6 p-4 bg-slate-800/30 rounded-2xl">
              <h4 className="text-sm text-slate-400 mb-4">Combined Attention (Head {Math.min(selectedToken + 1, numHeads)})</h4>
              <div className="overflow-x-auto">
                <div className="inline-block">
                  <div className="flex mb-2">
                    <div className="w-16" />
                    {tokens.map((t, i) => (
                      <div key={i} className="w-12 text-center text-[10px] text-slate-500">{t}</div>
                    ))}
                  </div>
                  {tokens.map((_, rowIdx) => (
                    <div key={rowIdx} className="flex items-center">
                      <div className="w-16 text-right pr-2 text-[10px] text-slate-500">{tokens[rowIdx]}</div>
                      <div className="flex gap-0.5">
                        {tokens.map((_, colIdx) => {
                          const headIdx = Math.min(selectedToken, numHeads - 1);
                          const weight = attentionData[headIdx][rowIdx][colIdx];
                          return (
                            <motion.div
                              key={colIdx}
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              className="w-12 h-8 rounded flex items-center justify-center text-[9px] font-mono"
                              style={{
                                backgroundColor: `rgba(139, 92, 246, ${weight})`,
                                color: weight > 0.5 ? "white" : "#94a3b8",
                              }}
                            >
                              {weight > 0.15 ? (weight * 100).toFixed(0) : ""}
                            </motion.div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

// ============================================
// MAIN PAGE COMPONENT
// ============================================

export default function TransformerPage() {
  const [activeSection, setActiveSection] = useState("intro");
  const [showNav, setShowNav] = useState(false);

  // Track scroll position for navigation
  useEffect(() => {
    const handleScroll = () => {
      const sections = SECTIONS.map((s) => ({
        id: s.id,
        element: document.getElementById(s.id),
      })).filter((s) => s.element);

      const scrollPosition = window.scrollY + window.innerHeight / 3;

      for (let i = sections.length - 1; i >= 0; i--) {
        const section = sections[i];
        if (section.element) {
          const offsetTop = section.element.offsetTop;
          if (scrollPosition >= offsetTop) {
            setActiveSection(section.id);
            break;
          }
        }
      }

      setShowNav(window.scrollY > 500);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white overflow-x-hidden">
      {/* Navigation */}
      <AnimatePresence>
        {showNav && (
          <NavigationSidebar
            activeSection={activeSection}
            onSectionClick={scrollToSection}
          />
        )}
      </AnimatePresence>

      {/* Mobile Navigation Toggle */}
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: showNav ? 1 : 0 }}
        onClick={() => setShowNav(!showNav)}
        className="fixed bottom-6 right-6 z-50 lg:hidden w-14 h-14 rounded-full bg-violet-600 text-white shadow-lg shadow-violet-500/25 flex items-center justify-center"
      >
        {showNav ? <ChevronLeft className="w-6 h-6" /> : <BookOpen className="w-6 h-6" />}
      </motion.button>

      {/* Back to Home */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="fixed top-6 left-6 z-50"
      >
        <Link
          href="/"
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900/80 backdrop-blur border border-slate-700/50 text-slate-300 hover:text-white hover:border-violet-500/50 transition-all"
        >
          <ChevronLeft className="w-4 h-4" />
          <span className="hidden sm:inline">Back to Home</span>
        </Link>
      </motion.div>

      {/* Content Sections */}
      <HeroSection />
      <ArchitectureSection />
      <EmbeddingsSection />
      <AttentionSection />
      <MultiHeadAttentionSection />
      <FeedForwardSection />
      <CodeSection />
      <PlaygroundSection />

      {/* Footer */}
      <footer className="py-20 border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h3 className="text-2xl font-bold text-white mb-4">
              Ready to Build Your Own Transformer?
            </h3>
            <p className="text-slate-400 mb-8 max-w-2xl mx-auto">
              Now that you understand how transformers work, try implementing one yourself 
              or experiment with pre-trained models in our playground.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                href="/models"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold hover:shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
              >
                <Sparkles className="w-5 h-5" />
                Create Model
              </Link>
              <Link
                href="/"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-slate-800 text-white font-semibold hover:bg-slate-700 transition-all duration-300"
              >
                <Brain className="w-5 h-5" />
                Back to Platform
              </Link>
            </div>
          </motion.div>

          <div className="mt-16 pt-8 border-t border-slate-800/50">
            <p className="text-slate-500 text-sm">
              Built for learning. Powered by curiosity.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
