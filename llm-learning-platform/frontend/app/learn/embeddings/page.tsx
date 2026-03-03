'use client';

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Layers, Settings, Play, Search,
  Loader2, BarChart3, Sparkles, Box, Grid3X3,
  Maximize2, Zap, Eye, EyeOff, RotateCcw
} from 'lucide-react';
import Link from 'next/link';
import * as d3 from 'd3';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import { api } from '@/utils/api';
import ModuleNavBar from '@/components/ModuleNavBar';

/* ==========================================================================
   3D SCENE COMPONENTS
   ========================================================================== */

// Token color by character type
function getTokenColor(token: string): string {
  if (token.match(/^[a-z]$/)) return '#3b82f6';
  if (token.match(/^[A-Z]$/)) return '#8b5cf6';
  if (token.match(/^[0-9]$/)) return '#f59e0b';
  if (token.match(/^[.,;:!?]$/)) return '#ef4444';
  if (token === '␣') return '#6b7280';
  return '#10b981';
}

// Single point sphere in 3D
function TokenPoint({
  position, color, token, index, isSelected, isHighlighted, isSearchMatch,
  onSelect, showLabels,
}: {
  position: [number, number, number];
  color: string;
  token: string;
  index: number;
  isSelected: boolean;
  isHighlighted: boolean;
  isSearchMatch: boolean;
  onSelect: (i: number) => void;
  showLabels: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const [hovered, setHovered] = useState(false);
  const baseSize = isSelected ? 0.12 : 0.07;
  const size = hovered ? baseSize * 1.5 : baseSize;

  const opacity = isSearchMatch || isSelected || isHighlighted ? 1 : (isSearchMatch === false ? 0.15 : 0.8);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.scale.lerp(new THREE.Vector3(size, size, size), 0.15);
    }
  });

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={(e) => { e.stopPropagation(); onSelect(index); }}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        scale={[baseSize, baseSize, baseSize]}
      >
        <sphereGeometry args={[1, 16, 16]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={opacity}
          emissive={isSelected || hovered ? color : '#000000'}
          emissiveIntensity={isSelected ? 0.8 : hovered ? 0.5 : 0}
        />
      </mesh>
      {(showLabels || hovered || isSelected) && (
        <Billboard follow lockX={false} lockY={false} lockZ={false}>
          <Text
            position={[0, 0.15, 0]}
            fontSize={0.1}
            color={isSelected ? '#ffffff' : '#94a3b8'}
            anchorX="center"
            anchorY="bottom"
            outlineWidth={0.01}
            outlineColor="#0f172a"
          >
            {token}
          </Text>
        </Billboard>
      )}
    </group>
  );
}

// Similarity lines connecting high-similarity pairs
function SimilarityLines({
  pairs, projections3d, visible,
}: {
  pairs: any[];
  projections3d: number[][];
  visible: boolean;
}) {
  if (!visible || !pairs || pairs.length === 0) return null;

  const lineGeometries = useMemo(() => {
    return pairs.slice(0, 30).map((pair: any) => {
      const a = projections3d[pair.idx_a];
      const b = projections3d[pair.idx_b];
      if (!a || !b) return null;
      const points = [new THREE.Vector3(a[0], a[1], a[2]), new THREE.Vector3(b[0], b[1], b[2])];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      return { geometry, similarity: pair.similarity };
    }).filter(Boolean);
  }, [pairs, projections3d]);

  return (
    <group>
      {lineGeometries.map((item: any, i: number) => {
        const Line3D: any = 'line';
        const LineMat3D: any = 'lineBasicMaterial';
        return (
          <Line3D key={i} geometry={item.geometry}>
            <LineMat3D
              color="#6366f1"
              transparent
              opacity={Math.max(0.1, item.similarity - 0.5)}
              linewidth={1}
            />
          </Line3D>
        );
      })}
    </group>
  );
}

// Axis helper with labels
function AxisLabels({ variance }: { variance: number[] }) {
  return (
    <group>
      {/* X axis */}
      <Billboard position={[2.5, 0, 0]}><Text fontSize={0.08} color="#64748b">PC1 ({variance[0]?.toFixed(1)}%)</Text></Billboard>
      {/* Y axis */}
      <Billboard position={[0, 2.5, 0]}><Text fontSize={0.08} color="#64748b">PC2 ({variance[1]?.toFixed(1)}%)</Text></Billboard>
      {/* Z axis */}
      <Billboard position={[0, 0, 2.5]}><Text fontSize={0.08} color="#64748b">PC3 ({variance[2]?.toFixed(1)}%)</Text></Billboard>
    </group>
  );
}

// Auto-rotate toggle
function AutoRotate({ enabled }: { enabled: boolean }) {
  const { camera } = useThree();
  useFrame((_, delta) => {
    if (enabled) {
      camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), delta * 0.15);
      camera.lookAt(0, 0, 0);
    }
  });
  return null;
}

/* ==========================================================================
   MAIN PAGE COMPONENT
   ========================================================================== */

export default function EmbeddingsPage() {
  const svgRef = useRef<SVGSVGElement>(null);

  // Config state
  const [vocabSize, setVocabSize] = useState(128);
  const [embeddingDim, setEmbeddingDim] = useState(64);
  const [inputText, setInputText] = useState('');

  // Data state from backend
  const [embeddingData, setEmbeddingData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Interaction state
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<'explore' | 'text' | 'arithmetic'>('explore');
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('3d');

  // 3D controls
  const [showLabels, setShowLabels] = useState(false);
  const [showSimilarityLines, setShowSimilarityLines] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  // Arithmetic state
  const [arithA, setArithA] = useState('A');
  const [arithB, setArithB] = useState('a');
  const [arithC, setArithC] = useState('B');
  const [arithResult, setArithResult] = useState<any>(null);

  // Fetch embeddings from backend
  const fetchEmbeddings = useCallback(async (text?: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.computeEmbeddings({
        text: text || '',
        vocab_size: vocabSize,
        embedding_dim: embeddingDim,
        seed: 42,
      });
      setEmbeddingData(data);
      setSelectedToken(null);
    } catch (err: any) {
      setError(err.message || 'Failed to compute embeddings');
    } finally {
      setLoading(false);
    }
  }, [vocabSize, embeddingDim]);

  // Load on mount and config change
  useEffect(() => {
    if (activeTab === 'explore') fetchEmbeddings();
  }, [vocabSize, embeddingDim, activeTab]);

  const handleTextSubmit = () => {
    if (inputText.trim()) {
      fetchEmbeddings(inputText);
      setActiveTab('text');
    }
  };

  // Arithmetic
  const computeArithmetic = async () => {
    try {
      const resp = await fetch(
        `http://localhost:8000/api/compute/embeddings/arithmetic?token_a=${encodeURIComponent(arithA)}&token_b=${encodeURIComponent(arithB)}&token_c=${encodeURIComponent(arithC)}&embedding_dim=${embeddingDim}&seed=42&vocab_size=${vocabSize}`,
        { method: 'POST' }
      );
      if (resp.ok) setArithResult(await resp.json());
    } catch (err) {
      console.error('Arithmetic error:', err);
    }
  };

  // Search filter
  const searchMatchIndices = useMemo(() => {
    if (!searchQuery || !embeddingData?.tokens) return null;
    const q = searchQuery.toLowerCase();
    const matches = new Set<number>();
    embeddingData.tokens.forEach((t: string, i: number) => {
      if (t.toLowerCase().includes(q)) matches.add(i);
    });
    return matches;
  }, [searchQuery, embeddingData]);

  // Nearest neighbors for selected token
  const nearestNeighbors = useMemo(() => {
    if (selectedToken === null || !embeddingData?.sample_vectors) return [];
    // Use norms & projections to compute rough neighbors from 3D data
    if (!embeddingData.projections_3d) return [];
    const target = embeddingData.projections_3d[selectedToken];
    if (!target) return [];
    const distances: { idx: number; token: string; dist: number }[] = [];
    embeddingData.projections_3d.forEach((p: number[], i: number) => {
      if (i === selectedToken) return;
      const dist = Math.sqrt((p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2 + (p[2] - target[2]) ** 2);
      distances.push({ idx: i, token: embeddingData.tokens[i], dist });
    });
    distances.sort((a, b) => a.dist - b.dist);
    return distances.slice(0, 10);
  }, [selectedToken, embeddingData]);

  // D3 2D Visualization
  useEffect(() => {
    if (viewMode !== '2d' || !embeddingData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const width = svgRef.current.clientWidth;
    const height = 500;
    const projections = embeddingData.projections;
    const tokens = embeddingData.tokens;
    if (!projections || projections.length === 0) return;

    const xExtent = d3.extent(projections, (d: any) => d[0]) as unknown as [number, number];
    const yExtent = d3.extent(projections, (d: any) => d[1]) as unknown as [number, number];
    const xPad = (xExtent[1] - xExtent[0]) * 0.1 || 1;
    const yPad = (yExtent[1] - yExtent[0]) * 0.1 || 1;

    const xScale = d3.scaleLinear().domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([50, width - 50]);
    const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([height - 50, 50]);

    const g = svg.append('g');
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 10])
      .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);

    // Grid
    g.append('line').attr('x1', xScale(0)).attr('y1', 50).attr('x2', xScale(0)).attr('y2', height - 50)
      .attr('stroke', '#e2e8f0').attr('stroke-dasharray', '4,4');
    g.append('line').attr('x1', 50).attr('y1', yScale(0)).attr('x2', width - 50).attr('y2', yScale(0))
      .attr('stroke', '#e2e8f0').attr('stroke-dasharray', '4,4');

    // Points
    const points = g.selectAll('g.point').data(projections).join('g')
      .attr('class', 'point')
      .attr('transform', (d: any) => `translate(${xScale(d[0])},${yScale(d[1])})`)
      .style('cursor', 'pointer');

    points.append('circle')
      .attr('r', (d: any, i: number) => i === selectedToken ? 8 : 5)
      .attr('fill', (d: any, i: number) => getTokenColor(tokens[i]))
      .attr('fill-opacity', (d: any, i: number) => {
        if (searchMatchIndices && !searchMatchIndices.has(i) && i !== selectedToken) return 0.1;
        return 0.8;
      })
      .attr('stroke', (d: any, i: number) => i === selectedToken ? '#1e293b' : 'white')
      .attr('stroke-width', (d: any, i: number) => i === selectedToken ? 2 : 1)
      .on('mouseover', function () { d3.select(this).transition().duration(150).attr('r', 10).attr('fill-opacity', 1); })
      .on('mouseout', function (_, d: any) {
        const i = projections.indexOf(d);
        d3.select(this).transition().duration(150).attr('r', 5)
          .attr('fill-opacity', searchMatchIndices && !searchMatchIndices.has(i) ? 0.1 : 0.8);
      })
      .on('click', (_: any, d: any) => {
        const idx = projections.indexOf(d);
        setSelectedToken(idx >= 0 ? idx : null);
      });

    const maxLabels = embeddingData.mode === 'text' ? tokens.length : Math.min(tokens.length, 80);
    points.filter((_: any, i: number) => i < maxLabels)
      .append('text').attr('x', 8).attr('y', 4)
      .text((_: any, i: number) => tokens[i])
      .attr('font-size', '11px').attr('fill', '#475569').attr('font-weight', '500');

    // Axis labels
    svg.append('text').attr('x', width / 2).attr('y', height - 10)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8').attr('font-size', '12px').text('PCA Component 1');
    svg.append('text').attr('x', 15).attr('y', height / 2)
      .attr('text-anchor', 'middle').attr('transform', `rotate(-90, 15, ${height / 2})`)
      .attr('fill', '#94a3b8').attr('font-size', '12px').text('PCA Component 2');
  }, [embeddingData, selectedToken, viewMode, searchMatchIndices]);

  // Normalize 3D projections for good scene scale
  const normalized3d = useMemo(() => {
    if (!embeddingData?.projections_3d) return [];
    const pts = embeddingData.projections_3d;
    let maxAbs = 0;
    pts.forEach((p: number[]) => p.forEach((v: number) => { if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v); }));
    const scale = maxAbs > 0 ? 2.0 / maxAbs : 1;
    return pts.map((p: number[]) => [p[0] * scale, p[1] * scale, p[2] * scale] as [number, number, number]);
  }, [embeddingData]);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950">
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
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                  <Layers className="h-4 w-4 text-white" />
                </div>
                <h1 className="text-lg font-semibold text-white">Embedding Explorer</h1>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Link href="/learn/tokenization" className="text-sm text-slate-500 hover:text-slate-300">← Tokenization</Link>
              <span className="text-slate-600">|</span>
              <Link href="/learn/attention" className="text-sm text-slate-500 hover:text-slate-300">Attention →</Link>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Theory Section */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-8 mb-8"
        >
          <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
            What are Embeddings?
          </h2>
          <p className="text-slate-300 leading-relaxed mb-4">
            Embeddings convert discrete tokens into dense numerical vectors in a high-dimensional space.
            Tokens with similar meanings cluster together — <strong className="text-white">distance encodes meaning</strong>.
            Toggle between <strong className="text-blue-400">2D</strong> and <strong className="text-purple-400">3D</strong> views to explore the embedding space.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { title: 'Dense Vectors', desc: 'Each token → vector of real numbers', icon: '🔢' },
              { title: 'Semantic Proximity', desc: 'Similar tokens cluster together', icon: '🧲' },
              { title: 'Learned Mapping', desc: 'Trained to encode meaning', icon: '🧠' },
              { title: '3D Projection', desc: 'PCA reduces N-dim → 3D for visualization', icon: '🌐' },
            ].map((item) => (
              <div key={item.title} className="bg-white/5 rounded-xl p-4 ring-1 ring-white/5">
                <div className="text-lg mb-1">{item.icon}</div>
                <h4 className="font-semibold text-white text-sm mb-1">{item.title}</h4>
                <p className="text-slate-400 text-xs">{item.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {[
            { id: 'explore' as const, label: 'Vocabulary Explorer', icon: Search },
            { id: 'text' as const, label: 'Text Embeddings', icon: BarChart3 },
            { id: 'arithmetic' as const, label: 'Embedding Arithmetic', icon: Sparkles },
          ].map((tab) => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${activeTab === tab.id
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                : 'bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white ring-1 ring-white/10'
                }`}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
        </div>

        <div className="grid gap-8 lg:grid-cols-4">
          {/* Left Panel */}
          <div className="space-y-5">
            {/* Config */}
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
              className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-5"
            >
              <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <Settings className="h-4 w-4 text-blue-400" /> Configuration
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1">
                    <label className="text-xs font-medium text-slate-400">Vocabulary Size</label>
                    <span className="text-xs text-blue-400 font-mono">{vocabSize}</span>
                  </div>
                  <input type="range" min={32} max={256} step={32} value={vocabSize}
                    onChange={(e) => setVocabSize(parseInt(e.target.value))} className="w-full accent-blue-500" />
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <label className="text-xs font-medium text-slate-400">Embedding Dim</label>
                    <span className="text-xs text-purple-400 font-mono">{embeddingDim}</span>
                  </div>
                  <input type="range" min={16} max={256} step={16} value={embeddingDim}
                    onChange={(e) => setEmbeddingDim(parseInt(e.target.value))} className="w-full accent-purple-500" />
                </div>
              </div>
            </motion.div>

            {/* View Mode Toggle */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-5">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Maximize2 className="h-4 w-4 text-purple-400" /> View Mode
              </h3>
              <div className="grid grid-cols-2 gap-2 mb-4">
                <button onClick={() => setViewMode('2d')}
                  className={`flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all ${viewMode === '2d' ? 'bg-blue-600 text-white' : 'bg-white/5 text-slate-400 hover:bg-white/10'}`}
                >
                  <Grid3X3 className="h-3.5 w-3.5" /> 2D PCA
                </button>
                <button onClick={() => setViewMode('3d')}
                  className={`flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all ${viewMode === '3d' ? 'bg-purple-600 text-white' : 'bg-white/5 text-slate-400 hover:bg-white/10'}`}
                >
                  <Box className="h-3.5 w-3.5" /> 3D PCA
                </button>
              </div>
              {viewMode === '3d' && (
                <div className="space-y-2">
                  <button onClick={() => setShowLabels(!showLabels)}
                    className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:bg-white/5 transition-all"
                  >
                    {showLabels ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
                    {showLabels ? 'Hide' : 'Show'} Labels
                  </button>
                  <button onClick={() => setShowSimilarityLines(!showSimilarityLines)}
                    className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:bg-white/5 transition-all"
                  >
                    <Zap className="h-3.5 w-3.5" />
                    {showSimilarityLines ? 'Hide' : 'Show'} Similarity Lines
                  </button>
                  <button onClick={() => setAutoRotate(!autoRotate)}
                    className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:bg-white/5 transition-all"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Auto-Rotate: {autoRotate ? 'On' : 'Off'}
                  </button>
                </div>
              )}
            </div>

            {/* Token Search */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-5">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Search className="h-4 w-4 text-emerald-400" /> Token Search
              </h3>
              <input type="text" value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Type to highlight tokens..."
                className="w-full px-3 py-2 text-sm bg-white/5 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              {searchMatchIndices && (
                <p className="text-xs text-slate-500 mt-2">{searchMatchIndices.size} matches</p>
              )}
            </div>

            {/* Text Input */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-5">
              <h3 className="text-sm font-semibold text-white mb-3">Explore Text</h3>
              <textarea value={inputText} onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text to see its embeddings..."
                className="w-full px-3 py-2 text-sm bg-white/5 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-blue-500 resize-none"
                rows={3} />
              <button onClick={handleTextSubmit} disabled={!inputText.trim() || loading}
                className="w-full mt-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-sm font-medium rounded-lg hover:from-blue-500 hover:to-purple-500 disabled:opacity-50 transition-all flex items-center justify-center gap-2 shadow-lg shadow-blue-600/20"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Compute Embeddings
              </button>
            </div>

            {/* Nearest Neighbors */}
            {selectedToken !== null && nearestNeighbors.length > 0 && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-2xl p-5 ring-1 ring-blue-500/20"
              >
                <h3 className="text-sm font-semibold text-white mb-1">
                  Nearest Neighbors of &quot;{embeddingData?.tokens?.[selectedToken]}&quot;
                </h3>
                <p className="text-xs text-slate-500 mb-3">By 3D Euclidean distance</p>
                <div className="space-y-1.5">
                  {nearestNeighbors.map((n, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="text-xs text-slate-500 w-4">{i + 1}</span>
                      <button onClick={() => setSelectedToken(n.idx)}
                        className="text-sm font-mono text-blue-400 hover:text-blue-300 w-6 text-center"
                      >
                        {n.token}
                      </button>
                      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                          style={{ width: `${Math.max(5, 100 - n.dist * 50)}%` }} />
                      </div>
                      <span className="text-xs text-slate-500 font-mono w-10 text-right">{n.dist.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Legend */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-5">
              <h3 className="text-sm font-semibold text-white mb-3">Color Legend</h3>
              <div className="space-y-1.5">
                {[
                  { color: '#3b82f6', label: 'Lowercase (a-z)' },
                  { color: '#8b5cf6', label: 'Uppercase (A-Z)' },
                  { color: '#f59e0b', label: 'Digits (0-9)' },
                  { color: '#ef4444', label: 'Punctuation' },
                  { color: '#10b981', label: 'Other symbols' },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="text-xs text-slate-400">{item.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Main Visualization Area */}
          <div className="lg:col-span-3 space-y-6">
            {/* Visualization */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 overflow-hidden"
            >
              <div className="p-4 border-b border-white/5 flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-white">
                    {embeddingData?.mode === 'text' ? 'Text Token Embeddings' : 'Vocabulary Embeddings'} — {viewMode === '3d' ? '3D' : '2D'} PCA
                  </h3>
                  <p className="text-sm text-slate-500">
                    {embeddingData ? `${embeddingData.tokens?.length || 0} tokens • ${embeddingData.embedding_dim || embeddingDim}D → ${viewMode === '3d' ? '3' : '2'}D` : 'Loading...'}
                    {embeddingData?.variance_explained && viewMode === '3d' && (
                      <span className="ml-2 text-blue-400">
                        ({embeddingData.variance_explained.map((v: number) => `${v.toFixed(1)}%`).join(' + ')} variance)
                      </span>
                    )}
                  </p>
                </div>
                {loading && <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />}
              </div>
              <div className="relative" style={{ height: 550 }}>
                {error && (
                  <div className="absolute inset-0 flex items-center justify-center bg-red-900/50 z-10">
                    <p className="text-red-300 text-sm">{error}</p>
                  </div>
                )}
                {viewMode === '3d' && embeddingData?.projections_3d ? (
                  <Canvas camera={{ position: [4, 3, 4], fov: 50 }}>
                    <color attach="background" args={['#0f172a']} />
                    <ambientLight intensity={0.4} />
                    <pointLight position={[10, 10, 10]} intensity={1} />
                    <pointLight position={[-10, -10, -10]} intensity={0.3} color="#818cf8" />
                    <fog attach="fog" args={['#0f172a', 8, 20]} />

                    {/* Grid & Axes */}
                    <gridHelper args={[6, 12, '#1e293b', '#1e293b']} position={[0, -2.5, 0]} />
                    <axesHelper args={[2.5]} />
                    <AxisLabels variance={embeddingData.variance_explained || [0, 0, 0]} />

                    {/* Token points */}
                    {normalized3d.map((pos: [number, number, number], i: number) => (
                      <TokenPoint
                        key={i}
                        position={pos}
                        color={getTokenColor(embeddingData.tokens[i])}
                        token={embeddingData.tokens[i]}
                        index={i}
                        isSelected={i === selectedToken}
                        isHighlighted={nearestNeighbors.some(n => n.idx === i)}
                        isSearchMatch={searchMatchIndices ? searchMatchIndices.has(i) : true}
                        onSelect={setSelectedToken}
                        showLabels={showLabels}
                      />
                    ))}

                    {/* Similarity lines */}
                    <SimilarityLines
                      pairs={embeddingData.top_similarity_pairs || embeddingData.similarities || []}
                      projections3d={normalized3d}
                      visible={showSimilarityLines}
                    />

                    <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
                    <AutoRotate enabled={autoRotate} />
                  </Canvas>
                ) : (
                  <svg ref={svgRef} width="100%" height={550} className="bg-slate-900/50" />
                )}
              </div>
            </motion.div>

            {/* Stats Row */}
            {embeddingData?.embedding_stats && activeTab === 'explore' && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="grid grid-cols-2 sm:grid-cols-4 gap-4"
              >
                {[
                  { label: 'Mean', value: embeddingData.embedding_stats.mean.toFixed(4), color: 'text-blue-400' },
                  { label: 'Std Dev', value: embeddingData.embedding_stats.std.toFixed(4), color: 'text-purple-400' },
                  { label: 'Min', value: embeddingData.embedding_stats.min.toFixed(4), color: 'text-red-400' },
                  { label: 'Max', value: embeddingData.embedding_stats.max.toFixed(4), color: 'text-green-400' },
                ].map((stat) => (
                  <div key={stat.label} className="bg-white/5 backdrop-blur-sm rounded-xl ring-1 ring-white/10 p-4 text-center">
                    <p className="text-xs text-slate-500 mb-1">{stat.label}</p>
                    <p className={`text-lg font-bold font-mono ${stat.color}`}>{stat.value}</p>
                  </div>
                ))}
              </motion.div>
            )}

            {/* Similarities */}
            {activeTab === 'text' && embeddingData?.similarities && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-6"
              >
                <h3 className="font-semibold text-white mb-4">Cosine Similarities (Top Pairs)</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {embeddingData.similarities.slice(0, 12).map((sim: any, idx: number) => (
                    <div key={idx} className="flex items-center justify-between bg-white/5 rounded-lg px-3 py-2">
                      <span className="text-sm font-mono text-slate-300">
                        &quot;{sim.token_a}&quot; ↔ &quot;{sim.token_b}&quot;
                      </span>
                      <span className={`text-sm font-bold ${sim.similarity > 0.5 ? 'text-green-400' : sim.similarity > 0 ? 'text-amber-400' : 'text-red-400'}`}>
                        {sim.similarity.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Arithmetic Tab */}
            {activeTab === 'arithmetic' && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 backdrop-blur-sm rounded-2xl ring-1 ring-white/10 p-8"
              >
                <h3 className="text-xl font-bold text-white mb-2">Embedding Arithmetic</h3>
                <p className="text-sm text-slate-400 mb-6">
                  Explore vector relationships: compute <strong className="text-white">A − B + C</strong> and find the closest token.
                </p>
                <div className="flex flex-wrap items-center gap-3 mb-6">
                  <input type="text" value={arithA} onChange={(e) => setArithA(e.target.value)} maxLength={1}
                    className="w-16 text-center px-3 py-3 bg-blue-500/10 border-2 border-blue-500/30 rounded-xl text-2xl font-bold text-blue-400 focus:ring-2 focus:ring-blue-500" />
                  <span className="text-2xl font-bold text-slate-500">−</span>
                  <input type="text" value={arithB} onChange={(e) => setArithB(e.target.value)} maxLength={1}
                    className="w-16 text-center px-3 py-3 bg-red-500/10 border-2 border-red-500/30 rounded-xl text-2xl font-bold text-red-400 focus:ring-2 focus:ring-red-500" />
                  <span className="text-2xl font-bold text-slate-500">+</span>
                  <input type="text" value={arithC} onChange={(e) => setArithC(e.target.value)} maxLength={1}
                    className="w-16 text-center px-3 py-3 bg-green-500/10 border-2 border-green-500/30 rounded-xl text-2xl font-bold text-green-400 focus:ring-2 focus:ring-green-500" />
                  <span className="text-2xl font-bold text-slate-500">=</span>
                  <span className="text-2xl font-bold text-purple-400">{arithResult?.top_matches?.[0]?.token || '?'}</span>
                  <button onClick={computeArithmetic}
                    className="ml-4 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium rounded-xl hover:from-blue-500 hover:to-purple-500 transition-all shadow-lg shadow-blue-600/20"
                  >
                    Compute
                  </button>
                </div>
                {arithResult && (
                  <div>
                    <h4 className="text-sm font-semibold text-slate-300 mb-3">Top 10 Closest Tokens:</h4>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                      {arithResult.top_matches?.slice(0, 10).map((match: any, i: number) => (
                        <div key={i} className={`text-center p-3 rounded-xl ${i === 0 ? 'bg-purple-500/20 ring-2 ring-purple-400/40' : 'bg-white/5'}`}>
                          <div className="text-lg font-bold text-white">{match.token}</div>
                          <div className="text-xs text-slate-500">sim: {match.similarity}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </div>
        </div>
      </div>
      <ModuleNavBar />
    </main>
  );
}
