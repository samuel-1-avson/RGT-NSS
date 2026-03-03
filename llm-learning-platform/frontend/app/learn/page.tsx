'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  ArrowRight, 
  CheckCircle2, 
  Lock,
  Sparkles
} from 'lucide-react';
import Link from 'next/link';

const learningPaths = [
  {
    id: 'beginner',
    name: 'Beginner Path',
    description: 'New to LLMs? Start here to build a solid foundation.',
    modules: [
      { id: 'tokenization', name: '1. Tokenization', status: 'available', icon: '🔤' },
      { id: 'embeddings', name: '2. Embeddings', status: 'locked', icon: '🎯' },
      { id: 'attention', name: '3. Attention Mechanisms', status: 'locked', icon: '🔍' },
      { id: 'transformer', name: '4. Transformer Architecture', status: 'locked', icon: '🏗️' },
      { id: 'training', name: '5. Training Fundamentals', status: 'locked', icon: '🚂' },
    ]
  },
  {
    id: 'intermediate',
    name: 'Intermediate Path',
    description: 'Have ML experience? Dive deeper into implementation details.',
    modules: [
      { id: 'attention', name: '1. Attention Deep Dive', status: 'available', icon: '🔍' },
      { id: 'transformer', name: '2. Transformer Blocks', status: 'available', icon: '🏗️' },
      { id: 'training', name: '3. Training & Optimization', status: 'locked', icon: '🚂' },
      { id: 'inference', name: '4. Inference & Generation', status: 'locked', icon: '💬' },
    ]
  },
  {
    id: 'expert',
    name: 'Expert Path',
    description: 'Ready for advanced topics? Explore cutting-edge techniques.',
    modules: [
      { id: 'advanced-attention', name: '1. Advanced Attention', status: 'available', icon: '⚡' },
      { id: 'optimization', name: '2. Training Optimization', status: 'available', icon: '🚀' },
      { id: 'efficiency', name: '3. Model Efficiency', status: 'available', icon: '🔧' },
      { id: 'research', name: '4. Research Frontiers', status: 'available', icon: '🔬' },
    ]
  }
];

const allModules = [
  { id: 'tokenization', name: 'Tokenization Lab', icon: '🔤', color: 'bg-blue-500', description: 'Understand how text is converted to tokens' },
  { id: 'embeddings', name: 'Embedding Explorer', icon: '🎯', color: 'bg-green-500', description: 'Explore vector representations of tokens' },
  { id: 'attention', name: 'Attention Visualizer', icon: '🔍', color: 'bg-purple-500', description: 'See how attention mechanisms work' },
  { id: 'transformer', name: 'Transformer Block', icon: '🏗️', color: 'bg-orange-500', description: 'Understand the complete transformer' },
  { id: 'training', name: 'Training Dashboard', icon: '🚂', color: 'bg-red-500', description: 'Train models and visualize progress' },
  { id: 'inference', name: 'Inference Playground', icon: '💬', color: 'bg-pink-500', description: 'Generate text and experiment' },
];

export default function LearnPage() {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  return (
    <main className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="h-8 w-8 text-primary-600" />
              <span className="text-xl font-semibold text-slate-900">
                LLM Learning Platform
              </span>
            </div>
            <nav className="flex gap-4">
              <Link
                href="/"
                className="text-sm font-medium text-slate-600 hover:text-slate-900"
              >
                Home
              </Link>
              <Link
                href="/train"
                className="text-sm font-medium text-slate-600 hover:text-slate-900"
              >
                Train
              </Link>
              <Link
                href="/docs"
                className="text-sm font-medium text-slate-600 hover:text-slate-900"
              >
                Docs
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero */}
      <div className="bg-white py-16">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h1 className="text-4xl font-bold tracking-tight text-slate-900">
                Choose Your Learning Path
              </h1>
              <p className="mt-4 text-lg text-slate-600 max-w-2xl mx-auto">
                Select a guided path based on your experience level, or explore 
                individual modules at your own pace.
              </p>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Learning Paths */}
      <div className="py-12">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid gap-6 lg:grid-cols-3">
            {learningPaths.map((path, index) => (
              <motion.div
                key={path.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`rounded-xl border-2 p-6 cursor-pointer transition-all ${
                  selectedPath === path.id
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-slate-200 bg-white hover:border-primary-300'
                }`}
                onClick={() => setSelectedPath(path.id)}
              >
                <h3 className="text-lg font-semibold text-slate-900">{path.name}</h3>
                <p className="mt-2 text-sm text-slate-600">{path.description}</p>
                <div className="mt-4 flex items-center text-primary-600">
                  <span className="text-sm font-medium">
                    {path.modules.length} modules
                  </span>
                  <ArrowRight className="ml-1 h-4 w-4" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Selected Path Modules */}
      {selectedPath && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-white border-t border-slate-200 py-12"
        >
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <h2 className="text-xl font-semibold text-slate-900 mb-6">
              {learningPaths.find(p => p.id === selectedPath)?.name} Modules
            </h2>
            <div className="space-y-4">
              {learningPaths
                .find(p => p.id === selectedPath)
                ?.modules.map((module, index) => (
                  <Link
                    key={module.id}
                    href={module.status === 'locked' ? '#' : `/learn/${module.id}`}
                    className={`flex items-center justify-between rounded-lg border p-4 transition-all ${
                      module.status === 'locked'
                        ? 'border-slate-200 bg-slate-50 opacity-60 cursor-not-allowed'
                        : 'border-slate-200 bg-white hover:border-primary-300 hover:shadow-sm'
                    }`}
                  >
                    <div className="flex items-center gap-4">
                      <span className="text-2xl">{module.icon}</span>
                      <div>
                        <h4 className="font-medium text-slate-900">{module.name}</h4>
                        <span className="text-sm text-slate-500">
                          {module.status === 'available' ? 'Ready to start' : 'Complete previous modules'}
                        </span>
                      </div>
                    </div>
                    {module.status === 'available' ? (
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                    ) : (
                      <Lock className="h-5 w-5 text-slate-400" />
                    )}
                  </Link>
                ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* All Modules Grid */}
      <div className="py-16">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold text-slate-900">All Modules</h2>
            <Sparkles className="h-5 w-5 text-primary-500" />
          </div>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {allModules.map((module, index) => (
              <motion.div
                key={module.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
              >
                <Link
                  href={`/learn/${module.id}`}
                  className="group block rounded-xl bg-white shadow-sm ring-1 ring-slate-200 hover:shadow-md transition-all overflow-hidden"
                >
                  <div className={`h-2 ${module.color}`} />
                  <div className="p-6">
                    <div className="flex items-start justify-between">
                      <span className="text-3xl">{module.icon}</span>
                      <ArrowRight className="h-5 w-5 text-slate-400 group-hover:text-primary-500 transition-colors" />
                    </div>
                    <h3 className="mt-4 text-lg font-semibold text-slate-900 group-hover:text-primary-600 transition-colors">
                      {module.name}
                    </h3>
                    <p className="mt-2 text-sm text-slate-600">
                      {module.description}
                    </p>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
