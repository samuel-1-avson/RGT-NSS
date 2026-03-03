'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  Brain, Layers, Activity, Code, Play, BookOpen,
  ArrowRight, Sparkles, Eye, Box, GraduationCap, Wand2
} from 'lucide-react';
import Link from 'next/link';

const features = [
  {
    icon: Layers,
    title: 'Interactive Modules',
    description: 'Explore tokenization, embeddings, attention, and transformers through hands-on visualizations with real data',
    gradient: 'from-blue-500 to-cyan-400',
  },
  {
    icon: Code,
    title: 'From Scratch Implementation',
    description: 'Learn with a custom-built deep learning framework — no PyTorch, no TensorFlow, full transparency',
    gradient: 'from-violet-500 to-purple-400',
  },
  {
    icon: Activity,
    title: 'Real-time Training',
    description: 'Watch your model train live with WebSocket-powered charts tracking loss, perplexity, and more',
    gradient: 'from-emerald-500 to-green-400',
  },
  {
    icon: Play,
    title: 'Inference Playground',
    description: 'Generate text with trained models and visualize how temperature, Top-K, and Top-P affect output',
    gradient: 'from-orange-500 to-amber-400',
  },
];

const modules = [
  { id: 'tokenization', name: 'Tokenization Lab', desc: 'Break text into tokens', icon: Code, gradient: 'from-blue-600 to-blue-400' },
  { id: 'embeddings', name: 'Embedding Explorer', desc: 'Visualize vector spaces', icon: Layers, gradient: 'from-purple-600 to-violet-400' },
  { id: 'attention', name: 'Attention Visualizer', desc: 'See what tokens attend to', icon: Eye, gradient: 'from-orange-600 to-amber-400' },
  { id: 'transformer', name: 'Transformer Block', desc: 'Explore the architecture', icon: Box, gradient: 'from-pink-600 to-rose-400' },
  { id: 'training', name: 'Training Fundamentals', desc: 'Train a real model', icon: GraduationCap, gradient: 'from-green-600 to-emerald-400' },
  { id: 'inference', name: 'Inference Playground', desc: 'Generate text', icon: Wand2, gradient: 'from-violet-600 to-fuchsia-400' },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-50">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-blue-950 to-purple-950" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:60px_60px]" />
        <div className="absolute top-20 left-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-10 right-1/4 w-72 h-72 bg-purple-600/20 rounded-full blur-[100px]" />

        <div className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="mb-8 inline-flex items-center gap-2 rounded-full bg-white/10 backdrop-blur-sm px-4 py-2 text-sm text-blue-200 ring-1 ring-white/20"
            >
              <Sparkles className="h-4 w-4" />
              Built with a custom deep learning framework
            </motion.div>

            {/* Logo */}
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1, type: 'spring', stiffness: 200 }}
              className="mb-8 flex justify-center"
            >
              <div className="h-20 w-20 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-2xl shadow-blue-500/30 animate-float">
                <Brain className="h-10 w-10 text-white" />
              </div>
            </motion.div>

            <h1 className="text-5xl sm:text-7xl font-bold tracking-tight text-white mb-6">
              Interactive{' '}
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                LLM
              </span>
              <br />
              Learning Platform
            </h1>
            <p className="mx-auto max-w-2xl text-lg sm:text-xl text-slate-300 leading-relaxed mb-10">
              Master Large Language Models through hands-on experimentation.
              Build a GPT from scratch, visualize every computation, and understand
              transformers at a fundamental level.
            </p>

            <div className="flex items-center justify-center gap-4 flex-wrap">
              <Link
                href="/learn"
                className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-500 hover:to-purple-500 transition-all shadow-xl shadow-blue-500/25 flex items-center gap-2"
              >
                Start Learning
                <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                href="/docs"
                className="px-8 py-4 bg-white/10 backdrop-blur-sm text-white font-semibold rounded-xl ring-1 ring-white/20 hover:bg-white/20 transition-all flex items-center gap-2"
              >
                <BookOpen className="h-5 w-5" />
                Documentation
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-white">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center mb-16">
            <h2 className="text-sm font-semibold text-blue-600 uppercase tracking-wider mb-3">
              Learn by Doing
            </h2>
            <p className="text-3xl sm:text-4xl font-bold text-slate-900">
              Everything you need to understand LLMs
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="group relative bg-white rounded-2xl p-8 ring-1 ring-slate-200 hover:shadow-xl hover:shadow-slate-100 transition-all"
              >
                <div className={`h-12 w-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-5 group-hover:scale-110 transition-transform`}>
                  <feature.icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-900 mb-2">{feature.title}</h3>
                <p className="text-slate-500 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Modules Section */}
      <section className="py-24 bg-slate-50">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-slate-900 mb-4">
              Learning Modules
            </h2>
            <p className="text-lg text-slate-500">
              Progress through each module to build your understanding from the ground up
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {modules.map((mod, index) => {
              const Icon = mod.icon;
              return (
                <motion.div
                  key={mod.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <Link
                    href={`/learn/${mod.id}`}
                    className="group relative flex flex-col overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 hover:shadow-lg hover:ring-slate-300 transition-all"
                  >
                    <div className={`h-1.5 bg-gradient-to-r ${mod.gradient}`} />
                    <div className="p-6">
                      <div className="flex items-center gap-4 mb-3">
                        <div className={`h-10 w-10 rounded-xl bg-gradient-to-br ${mod.gradient} flex items-center justify-center shadow-lg`}>
                          <Icon className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                            {mod.name}
                          </h3>
                          <p className="text-sm text-slate-400">{mod.desc}</p>
                        </div>
                      </div>
                      <div className="flex items-center text-sm text-slate-500 group-hover:text-blue-500 transition-colors">
                        <span>Explore module</span>
                        <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                      </div>
                    </div>
                  </Link>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-blue-950 to-purple-950 py-24">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px]" />
        <div className="relative mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <Sparkles className="mx-auto h-12 w-12 text-blue-400 mb-6" />
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
                Ready to train your own GPT?
              </h2>
              <p className="text-lg text-slate-300 mb-10">
                Choose a dataset, configure your model, and watch it learn in real-time
                with live loss curves and metrics.
              </p>
              <div className="flex items-center justify-center gap-4 flex-wrap">
                <Link
                  href="/train"
                  className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-500 hover:to-purple-500 transition-all shadow-xl shadow-blue-600/25 flex items-center gap-2"
                >
                  <Play className="h-5 w-5" />
                  Open Training Dashboard
                </Link>
                <Link
                  href="/docs"
                  className="px-8 py-4 bg-white/10 backdrop-blur-sm text-white font-semibold rounded-xl ring-1 ring-white/20 hover:bg-white/20 transition-all flex items-center gap-2"
                >
                  <BookOpen className="h-5 w-5" />
                  Documentation
                </Link>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white py-12 border-t border-slate-200">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="flex flex-col items-center justify-between gap-6 sm:flex-row">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center">
                <Brain className="h-4 w-4 text-white" />
              </div>
              <span className="text-lg font-bold text-slate-900">
                LLM Learning Platform
              </span>
            </div>
            <p className="text-sm text-slate-500">
              Built for education. Custom deep learning framework with zero external dependencies.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
