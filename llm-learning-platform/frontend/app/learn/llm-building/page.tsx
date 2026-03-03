"use client";

import React from "react";
import { motion } from "framer-motion";
import { ArrowLeft, Code, BookOpen, ArrowRight, Layers, Sparkles, CheckCircle } from "lucide-react";
import Link from "next/link";

const beginnerSteps = [
  {
    title: "Understanding LLMs",
    description: "Learn what Large Language Models are and how they work at a high level.",
    duration: "15 min",
    completed: true
  },
  {
    title: "Tokenization Fundamentals",
    description: "How text is converted into tokens that models can process.",
    duration: "20 min",
    completed: true
  },
  {
    title: "The Transformer Architecture",
    description: "Understanding the breakthrough architecture behind modern AI.",
    duration: "30 min",
    completed: false
  },
  {
    title: "Your First Model",
    description: "Build a simple character-level language model from scratch.",
    duration: "45 min",
    completed: false
  }
];

const intermediateSteps = [
  {
    title: "Attention Mechanisms",
    description: "Deep dive into self-attention and multi-head attention.",
    duration: "40 min"
  },
  {
    title: "Training at Scale",
    description: "Understanding loss functions, optimization, and gradient descent.",
    duration: "50 min"
  },
  {
    title: "Building GPT-2",
    description: "Implement a GPT-2 style transformer model step by step.",
    duration: "60 min"
  }
];

export default function LLMBuildingPage() {
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
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-rose-500/10 text-rose-300 text-sm font-medium mb-6">
              <Code className="w-4 h-4" />
              <span>LLM Building</span>
            </div>
            <h1 className="text-5xl font-bold mb-6">
              Build Your Own <span className="text-rose-400">Language Model</span>
            </h1>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto">
              From zero to hero. Learn to build language models from scratch with hands-on projects.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Beginner Track */}
      <section className="py-20 border-y border-slate-800/50">
        <div className="max-w-4xl mx-auto px-6">
          <div className="flex items-center gap-3 mb-8">
            <BookOpen className="w-6 h-6 text-emerald-400" />
            <h2 className="text-2xl font-bold">Beginner Track</h2>
          </div>

          <div className="space-y-4">
            {beginnerSteps.map((step, i) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="flex items-center gap-4 p-4 bg-slate-900/50 rounded-xl border border-slate-700/50"
              >
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  step.completed 
                    ? "bg-emerald-500/20 text-emerald-400" 
                    : "bg-slate-700 text-slate-400"
                }`}>
                  {step.completed ? <CheckCircle className="w-5 h-5" /> : <span>{i + 1}</span>}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-white">{step.title}</h3>
                  <p className="text-sm text-slate-400">{step.description}</p>
                </div>
                <span className="text-xs text-slate-500">{step.duration}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Intermediate Track */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-6">
          <div className="flex items-center gap-3 mb-8">
            <Layers className="w-6 h-6 text-violet-400" />
            <h2 className="text-2xl font-bold">Intermediate Track</h2>
          </div>

          <div className="space-y-4">
            {intermediateSteps.map((step, i) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="flex items-center gap-4 p-4 bg-slate-900/50 rounded-xl border border-slate-700/50"
              >
                <div className="w-10 h-10 rounded-full bg-slate-700 text-slate-400 flex items-center justify-center">
                  {i + 1}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-white">{step.title}</h3>
                  <p className="text-sm text-slate-400">{step.description}</p>
                </div>
                <span className="text-xs text-slate-500">{step.duration}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 border-t border-slate-800/50">
        <div className="max-w-4xl mx-auto px-6">
          <div className="p-8 bg-gradient-to-br from-rose-900/20 to-pink-900/20 rounded-3xl border border-rose-500/20 text-center">
            <Sparkles className="w-12 h-12 text-rose-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-white mb-2">Ready to Start Building?</h2>
            <p className="text-slate-400 mb-6">
              Jump into the transformer tutorial and start coding your first model.
            </p>
            <Link
              href="/learn/transformer/"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-rose-600 to-pink-600 text-white font-semibold hover:shadow-lg hover:shadow-rose-500/25 transition-all"
            >
              <Code className="w-5 h-5" />
              Start Building
              <ArrowRight className="w-5 h-5" />
            </Link>
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
            href="/learn/tokenization/"
            className="flex items-center gap-2 text-rose-400 hover:text-rose-300 transition-colors"
          >
            <span>Start: Tokenization</span>
            <ArrowLeft className="w-5 h-5 rotate-180" />
          </Link>
        </div>
      </section>
    </main>
  );
}
