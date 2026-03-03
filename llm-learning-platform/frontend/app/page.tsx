"use client";

import React from "react";
import { motion } from "framer-motion";
import { Brain, BookOpen, Sparkles, ArrowRight, Layers, Zap, Code } from "lucide-react";
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-slate-950 text-white">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(124,58,237,0.15),transparent_50%)]" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,rgba(99,102,241,0.15),transparent_50%)]" />
          
          <motion.div
            className="absolute top-20 left-10 w-64 h-64 bg-violet-500/10 rounded-full blur-3xl"
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 8, repeat: Infinity }}
          />
          <motion.div
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
            <span>Master AI & Machine Learning</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="text-6xl md:text-8xl font-bold mb-6"
          >
            <span className="bg-gradient-to-r from-white via-violet-200 to-indigo-200 bg-clip-text text-transparent">
              LLM Learning
            </span>
            <br />
            <span className="bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
              Platform
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-xl md:text-2xl text-slate-400 max-w-3xl mx-auto mb-12"
          >
            Build, train, and understand Large Language Models from scratch. 
            Interactive tutorials, visualizations, and hands-on experiments.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="flex flex-wrap justify-center gap-4"
          >
            <Link
              href="/learn/"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold hover:shadow-lg hover:shadow-violet-500/25 transition-all duration-300 hover:scale-105"
            >
              <BookOpen className="w-5 h-5" />
              Start Learning
            </Link>
            <Link
              href="/models/"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-slate-800/80 border border-slate-700 text-white font-semibold hover:bg-slate-700/80 transition-all duration-300"
            >
              <Brain className="w-5 h-5" />
              Build Models
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-32 relative">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Everything You Need to <span className="text-violet-400">Master AI</span>
            </h2>
            <p className="text-slate-400 text-lg max-w-2xl mx-auto">
              From foundational concepts to cutting-edge implementations
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: <BookOpen className="w-6 h-6" />,
                title: "Interactive Tutorials",
                desc: "Learn transformers, attention mechanisms, and more with visual guides",
                link: "/learn/transformer/",
                color: "from-violet-500 to-purple-500",
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Model Builder",
                desc: "Create and configure your own language models with our visual interface",
                link: "/models/",
                color: "from-emerald-500 to-teal-500",
              },
              {
                icon: <Layers className="w-6 h-6" />,
                title: "Training Dashboard",
                desc: "Monitor training progress with real-time metrics and visualizations",
                link: "/train/",
                color: "from-amber-500 to-orange-500",
              },
              {
                icon: <Code className="w-6 h-6" />,
                title: "Code Examples",
                desc: "Production-ready implementations in PyTorch, TensorFlow, and NumPy",
                link: "/learn/transformer/#code",
                color: "from-cyan-500 to-blue-500",
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "Interactive Playground",
                desc: "Experiment with attention patterns and see results in real-time",
                link: "/learn/transformer/#playground",
                color: "from-rose-500 to-pink-500",
              },
              {
                icon: <Sparkles className="w-6 h-6" />,
                title: "Inference API",
                desc: "Test your trained models with our easy-to-use API",
                link: "/inference/",
                color: "from-fuchsia-500 to-violet-500",
              },
            ].map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
              >
                <Link
                  href={feature.link}
                  className="group block p-8 bg-slate-900/50 rounded-3xl border border-slate-700/50 hover:border-violet-500/50 transition-all duration-300"
                >
                  <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center text-white mb-6 group-hover:scale-110 transition-transform`}>
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2 group-hover:text-violet-400 transition-colors">
                    {feature.title}
                  </h3>
                  <p className="text-slate-400 mb-4">{feature.desc}</p>
                  <div className="flex items-center gap-2 text-violet-400 text-sm font-medium">
                    <span>Explore</span>
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 relative">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="p-12 bg-gradient-to-br from-violet-900/20 to-indigo-900/20 rounded-3xl border border-violet-500/20"
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Start Your AI Journey Today
            </h2>
            <p className="text-slate-400 text-lg mb-8 max-w-2xl mx-auto">
              Dive into our comprehensive transformer tutorial and understand 
              the architecture powering modern AI.
            </p>
            <Link
              href="/learn/"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold hover:shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
            >
              <BookOpen className="w-5 h-5" />
              Start Learning
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center gap-2">
              <Brain className="w-6 h-6 text-violet-400" />
              <span className="text-xl font-bold text-white">LLM Learning Platform</span>
            </div>
            <p className="text-slate-500 text-sm">
              Built for learning. Powered by curiosity.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
