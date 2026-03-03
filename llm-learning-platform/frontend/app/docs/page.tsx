'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Book, Code2, BookOpen, Cpu, Settings, Zap, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import Link from 'next/link';

interface DocSection {
    title: string;
    icon: any;
    content: React.ReactNode;
}

const CodeBlock = ({ children, title }: { children: string; title?: string }) => (
    <div className="my-4">
        {title && <p className="text-xs font-medium text-slate-500 mb-1">{title}</p>}
        <pre className="bg-slate-900 text-slate-100 p-4 rounded-xl text-sm font-mono overflow-x-auto leading-relaxed">
            <code>{children}</code>
        </pre>
    </div>
);

export default function DocsPage() {
    const [expandedSection, setExpandedSection] = useState<number | null>(0);

    const sections: DocSection[] = [
        {
            title: 'Getting Started',
            icon: Book,
            content: (
                <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
                    <p>
                        The LLM Learning Platform is an interactive educational tool that lets you explore how
                        Large Language Models work — from the ground up. Unlike other tutorials, this platform
                        uses a <strong>custom-built, from-scratch deep learning framework</strong> in Python,
                        so you can see every computation happening inside the model.
                    </p>
                    <h4 className="font-semibold text-slate-800">Prerequisites</h4>
                    <ul className="list-disc list-inside space-y-1 text-slate-500">
                        <li>Python 3.8+ with pip</li>
                        <li>Node.js 18+ with npm</li>
                        <li>Basic understanding of linear algebra and calculus (helpful but not required)</li>
                    </ul>
                    <h4 className="font-semibold text-slate-800">Quick Start</h4>
                    <CodeBlock title="Backend Setup">{`# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or .\\venv\\Scripts\\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --reload`}</CodeBlock>
                    <CodeBlock title="Frontend Setup">{`# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev`}</CodeBlock>
                    <p className="text-slate-500">
                        The platform runs at <code className="bg-slate-100 px-1.5 py-0.5 rounded text-xs">http://localhost:3000</code> (frontend)
                        and <code className="bg-slate-100 px-1.5 py-0.5 rounded text-xs">http://localhost:8000</code> (backend API).
                    </p>
                </div>
            ),
        },
        {
            title: 'Architecture Overview',
            icon: Cpu,
            content: (
                <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
                    <p>The platform consists of two main components:</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-blue-50 rounded-xl p-4">
                            <h4 className="font-semibold text-blue-800 mb-2">Frontend (Next.js)</h4>
                            <ul className="text-xs text-blue-700 space-y-1">
                                <li>• React 18 with App Router</li>
                                <li>• Framer Motion for animations</li>
                                <li>• D3.js for data visualization</li>
                                <li>• Recharts for charts</li>
                                <li>• Tailwind CSS for styling</li>
                                <li>• Zustand for state management</li>
                            </ul>
                        </div>
                        <div className="bg-green-50 rounded-xl p-4">
                            <h4 className="font-semibold text-green-800 mb-2">Backend (FastAPI)</h4>
                            <ul className="text-xs text-green-700 space-y-1">
                                <li>• Custom autograd engine (Tensor class)</li>
                                <li>• Full GPT implementation from scratch</li>
                                <li>• Training engine with LR scheduling</li>
                                <li>• Adam/AdamW/SGD optimizers</li>
                                <li>• WebSocket for real-time metrics</li>
                                <li>• RESTful API endpoints</li>
                            </ul>
                        </div>
                    </div>
                    <h4 className="font-semibold text-slate-800 mt-4">Custom Framework</h4>
                    <p>
                        The backend uses a <strong>completely custom deep learning framework</strong> — no PyTorch,
                        no TensorFlow. This includes:
                    </p>
                    <ul className="list-disc list-inside space-y-1 text-slate-500">
                        <li><strong>Tensor</strong> — N-dimensional array with automatic differentiation</li>
                        <li><strong>Module</strong> — Base class for neural network layers (Linear, Embedding, Dropout)</li>
                        <li><strong>GPT Model</strong> — Transformer with Multi-Head Attention, RMSNorm, MLP</li>
                        <li><strong>Training Engine</strong> — Full training loop with checkpointing and metrics</li>
                        <li><strong>Optimizers</strong> — SGD, Adam, AdamW with learning rate schedulers</li>
                    </ul>
                </div>
            ),
        },
        {
            title: 'Learning Modules',
            icon: BookOpen,
            content: (
                <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
                    <p>The platform offers six interactive learning modules, each focusing on a different aspect of LLMs:</p>
                    {[
                        { title: '1. Tokenization Lab', href: '/learn/tokenization', desc: 'Explore character-level, word-level, and BPE tokenization. See how text is broken into tokens and mapped to IDs.' },
                        { title: '2. Embedding Explorer', href: '/learn/embeddings', desc: 'Visualize token embeddings in 2D using PCA projection. Explore cosine similarities and try embedding arithmetic.' },
                        { title: '3. Attention Visualizer', href: '/learn/attention', desc: 'Watch real attention computation with configurable d_model, num_heads, and layers. Interactive heatmaps and Q/K/V vector display.' },
                        { title: '4. Transformer Block', href: '/learn/transformer', desc: 'Step through the components of a transformer: RMSNorm, Multi-Head Attention, Residual Connections, and MLP.' },
                        { title: '5. Training Fundamentals', href: '/learn/training', desc: 'Train a real model on built-in or custom datasets. Watch loss curves and perplexity update in real-time via WebSocket.' },
                        { title: '6. Inference Playground', href: '/learn/inference', desc: 'Generate text with a trained model. Visualize how temperature, Top-K, and Top-P affect the probability distribution.' },
                    ].map((mod) => (
                        <Link key={mod.href} href={mod.href} className="block bg-slate-50 rounded-xl p-4 hover:bg-slate-100 transition-colors group">
                            <div className="flex items-center justify-between">
                                <h4 className="font-semibold text-slate-800">{mod.title}</h4>
                                <ExternalLink className="h-3.5 w-3.5 text-slate-400 group-hover:text-blue-500 transition-colors" />
                            </div>
                            <p className="text-xs text-slate-500 mt-1">{mod.desc}</p>
                        </Link>
                    ))}
                </div>
            ),
        },
        {
            title: 'API Reference',
            icon: Code2,
            content: (
                <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
                    <p>The backend provides RESTful endpoints organized into three groups:</p>

                    <h4 className="font-semibold text-slate-800">Model Management</h4>
                    <div className="space-y-2">
                        {[
                            { method: 'POST', path: '/api/models/create', desc: 'Create a new GPT model' },
                            { method: 'GET', path: '/api/models', desc: 'List all models' },
                            { method: 'GET', path: '/api/models/{id}', desc: 'Get model details' },
                            { method: 'DELETE', path: '/api/models/{id}', desc: 'Delete a model' },
                        ].map((ep) => (
                            <div key={ep.path} className="flex items-center gap-3 text-xs font-mono bg-slate-50 px-3 py-2 rounded-lg">
                                <span className={`px-2 py-0.5 rounded text-white text-[10px] font-bold ${ep.method === 'GET' ? 'bg-green-500' : ep.method === 'POST' ? 'bg-blue-500' : 'bg-red-500'
                                    }`}>{ep.method}</span>
                                <span className="text-slate-700">{ep.path}</span>
                                <span className="text-slate-400 ml-auto font-sans">{ep.desc}</span>
                            </div>
                        ))}
                    </div>

                    <h4 className="font-semibold text-slate-800 mt-4">Compute (Standalone)</h4>
                    <div className="space-y-2">
                        {[
                            { method: 'POST', path: '/api/compute/attention', desc: 'Compute real attention weights' },
                            { method: 'POST', path: '/api/compute/embeddings', desc: 'Generate embeddings with PCA' },
                            { method: 'POST', path: '/api/compute/sampling', desc: 'Token probability distribution' },
                            { method: 'POST', path: '/api/compute/forward-step', desc: 'Step-by-step forward pass' },
                            { method: 'GET', path: '/api/compute/datasets', desc: 'List available datasets' },
                            { method: 'POST', path: '/api/compute/datasets/upload', desc: 'Upload custom dataset' },
                        ].map((ep) => (
                            <div key={ep.path} className="flex items-center gap-3 text-xs font-mono bg-slate-50 px-3 py-2 rounded-lg">
                                <span className={`px-2 py-0.5 rounded text-white text-[10px] font-bold ${ep.method === 'GET' ? 'bg-green-500' : 'bg-blue-500'
                                    }`}>{ep.method}</span>
                                <span className="text-slate-700">{ep.path}</span>
                                <span className="text-slate-400 ml-auto font-sans">{ep.desc}</span>
                            </div>
                        ))}
                    </div>

                    <h4 className="font-semibold text-slate-800 mt-4">Training & Inference</h4>
                    <div className="space-y-2">
                        {[
                            { method: 'POST', path: '/api/training/start', desc: 'Start training session' },
                            { method: 'POST', path: '/api/training/stop/{id}', desc: 'Stop training session' },
                            { method: 'POST', path: '/api/inference/generate', desc: 'Generate text' },
                            { method: 'POST', path: '/api/tokenize', desc: 'Tokenize text' },
                        ].map((ep) => (
                            <div key={ep.path} className="flex items-center gap-3 text-xs font-mono bg-slate-50 px-3 py-2 rounded-lg">
                                <span className="px-2 py-0.5 rounded text-white text-[10px] font-bold bg-blue-500">{ep.method}</span>
                                <span className="text-slate-700">{ep.path}</span>
                                <span className="text-slate-400 ml-auto font-sans">{ep.desc}</span>
                            </div>
                        ))}
                    </div>
                </div>
            ),
        },
        {
            title: 'Configuration',
            icon: Settings,
            content: (
                <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
                    <h4 className="font-semibold text-slate-800">Model Configuration (GPTConfig)</h4>
                    <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                            <thead>
                                <tr className="border-b border-slate-200">
                                    <th className="py-2 px-3 text-left text-slate-600">Parameter</th>
                                    <th className="py-2 px-3 text-left text-slate-600">Default</th>
                                    <th className="py-2 px-3 text-left text-slate-600">Description</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {[
                                    ['vocab_size', '256', 'Size of the token vocabulary'],
                                    ['d_model', '64', 'Embedding dimension / model width'],
                                    ['n_heads', '4', 'Number of attention heads'],
                                    ['n_layers', '2', 'Number of transformer layers'],
                                    ['d_ff', '256', 'Feedforward network hidden dimension'],
                                    ['max_seq_len', '128', 'Maximum sequence length'],
                                    ['dropout', '0.1', 'Dropout probability'],
                                ].map(([param, def, desc]) => (
                                    <tr key={param}>
                                        <td className="py-2 px-3 font-mono text-slate-800">{param}</td>
                                        <td className="py-2 px-3 font-mono text-blue-600">{def}</td>
                                        <td className="py-2 px-3 text-slate-500">{desc}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <h4 className="font-semibold text-slate-800 mt-4">Model Presets</h4>
                    <div className="grid grid-cols-2 gap-3">
                        {[
                            { name: 'Nano', params: '~5K', desc: 'd_model=32, 2 heads, 2 layers' },
                            { name: 'Micro', params: '~20K', desc: 'd_model=64, 4 heads, 4 layers' },
                            { name: 'Mini', params: '~100K', desc: 'd_model=128, 4 heads, 4 layers' },
                            { name: 'Small', params: '~500K', desc: 'd_model=256, 8 heads, 6 layers' },
                        ].map((preset) => (
                            <div key={preset.name} className="bg-slate-50 rounded-lg p-3">
                                <div className="font-semibold text-slate-800 text-xs">{preset.name} <span className="text-slate-400 font-normal">({preset.params})</span></div>
                                <div className="text-xs text-slate-500 mt-0.5">{preset.desc}</div>
                            </div>
                        ))}
                    </div>
                </div>
            ),
        },
        {
            title: 'Key Concepts',
            icon: Zap,
            content: (
                <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
                    <div className="space-y-4">
                        {[
                            {
                                term: 'Autograd (Automatic Differentiation)',
                                desc: 'Our Tensor class tracks all operations in a computation graph. When you call .backward(), it automatically computes gradients for all parameters using the chain rule — no manual calculus needed.'
                            },
                            {
                                term: 'Self-Attention',
                                desc: 'A mechanism where each token in a sequence attends to every other token to determine relevance. Computed as softmax(QK^T/√d_k)V, where Q, K, V are learned linear projections of the input.'
                            },
                            {
                                term: 'Multi-Head Attention',
                                desc: 'Instead of one attention function, the model uses multiple "heads" that attend to different parts of the input simultaneously. Each head can learn different patterns (syntax, semantics, position).'
                            },
                            {
                                term: 'Cross-Entropy Loss',
                                desc: 'Measures how far the model\'s predicted probability distribution is from the actual next token. Lower loss = better predictions. Related to perplexity: PPL = e^loss.'
                            },
                            {
                                term: 'Causal Masking',
                                desc: 'In autoregressive generation, each token can only attend to previous tokens (not future ones). This is enforced by setting future attention scores to -infinity before softmax.'
                            },
                            {
                                term: 'Temperature Sampling',
                                desc: 'Controls randomness during generation. Low temperature (0.1) = deterministic/repetitive. High temperature (2.0) = creative/chaotic. Applied by dividing logits before softmax.'
                            },
                        ].map((item) => (
                            <div key={item.term} className="bg-slate-50 rounded-xl p-4">
                                <h4 className="font-semibold text-slate-800 mb-1">{item.term}</h4>
                                <p className="text-xs text-slate-500">{item.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            ),
        },
    ];

    return (
        <main className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/20 to-violet-50/20">
            <header className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-50">
                <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
                    <div className="flex h-16 items-center gap-4">
                        <Link href="/" className="flex items-center gap-2 text-slate-600 hover:text-slate-900">
                            <FileText className="h-5 w-5" />
                        </Link>
                        <div className="h-6 w-px bg-slate-300" />
                        <h1 className="text-lg font-semibold text-slate-900">Documentation</h1>
                    </div>
                </div>
            </header>

            <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-10"
                >
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-3">
                        LLM Learning Platform Documentation
                    </h1>
                    <p className="text-slate-500 text-lg">
                        Everything you need to understand, set up, and use the interactive LLM learning platform.
                    </p>
                </motion.div>

                <div className="space-y-4">
                    {sections.map((section, idx) => {
                        const Icon = section.icon;
                        const isExpanded = expandedSection === idx;

                        return (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: idx * 0.05 }}
                                className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 overflow-hidden"
                            >
                                <button
                                    onClick={() => setExpandedSection(isExpanded ? null : idx)}
                                    className="w-full flex items-center gap-4 p-6 text-left hover:bg-slate-50 transition-colors"
                                >
                                    <div className={`h-10 w-10 rounded-xl flex items-center justify-center ${isExpanded ? 'bg-gradient-to-br from-blue-600 to-purple-600' : 'bg-slate-100'
                                        }`}>
                                        <Icon className={`h-5 w-5 ${isExpanded ? 'text-white' : 'text-slate-500'}`} />
                                    </div>
                                    <h2 className="text-lg font-semibold text-slate-900 flex-1">{section.title}</h2>
                                    {isExpanded ? (
                                        <ChevronDown className="h-5 w-5 text-slate-400" />
                                    ) : (
                                        <ChevronRight className="h-5 w-5 text-slate-400" />
                                    )}
                                </button>
                                <AnimatePresence>
                                    {isExpanded && (
                                        <motion.div
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: 'auto', opacity: 1 }}
                                            exit={{ height: 0, opacity: 0 }}
                                            transition={{ duration: 0.2 }}
                                            className="overflow-hidden"
                                        >
                                            <div className="px-6 pb-6 pt-0">
                                                {section.content}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </motion.div>
                        );
                    })}
                </div>
            </div>
        </main>
    );
}
