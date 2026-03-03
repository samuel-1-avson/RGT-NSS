"use client";

import React from "react";
import { motion } from "framer-motion";
import { 
  BookOpen, 
  ArrowRight, 
  Brain, 
  Layers, 
  Eye, 
  Sparkles,
  Code,
  Terminal
} from "lucide-react";
import Link from "next/link";

const learningPaths = [
  {
    id: "beginner",
    title: "Beginner Track",
    description: "Start your journey into LLMs. Learn the fundamentals of transformers, tokenization, and attention mechanisms.",
    icon: <BookOpen className="w-6 h-6" />,
    color: "from-emerald-500 to-teal-500",
    modules: [
      { title: "What are LLMs?", path: "/learn/llm-building/" },
      { title: "Tokenization", path: "/learn/tokenization/" },
      { title: "Transformer Architecture", path: "/learn/transformer/" },
    ]
  },
  {
    id: "intermediate",
    title: "Intermediate Track",
    description: "Deep dive into attention mechanisms, multi-head attention, and building your first language model.",
    icon: <Layers className="w-6 h-6" />,
    color: "from-violet-500 to-purple-500",
    modules: [
      { title: "Attention Mechanisms", path: "/learn/attention/" },
      { title: "Multi-Head Attention", path: "/learn/transformer/#multihead" },
      { title: "Training Fundamentals", path: "/train/" },
    ]
  },
  {
    id: "advanced",
    title: "Advanced Track",
    description: "Build production-ready models. Explore optimization, distributed training, and deployment.",
    icon: <Sparkles className="w-6 h-6" />,
    color: "from-amber-500 to-orange-500",
    modules: [
      { title: "Model Optimization", path: "/models/" },
      { title: "Distributed Training", path: "/train/" },
      { title: "Inference API", path: "/inference/" },
    ]
  }
];

const featuredModules = [
  {
    title: "Transformer Architecture",
    description: "The breakthrough architecture that powers GPT, BERT, and modern AI.",
    icon: <Brain className="w-8 h-8" />,
    path: "/learn/transformer/",
    color: "from-violet-500 to-indigo-500",
    tag: "Popular"
  },
  {
    title: "Attention Visualizer",
    description: "Interactive exploration of how attention mechanisms work.",
    icon: <Eye className="w-8 h-8" />,
    path: "/learn/attention/",
    color: "from-amber-500 to-orange-500",
    tag: "Interactive"
  },
  {
    title: "Tokenizer Playground",
    description: "Learn how text becomes tokens with hands-on examples.",
    icon: <Terminal className="w-8 h-8" />,
    path: "/learn/tokenization/",
    color: "from-emerald-500 to-teal-500",
    tag: "Hands-on"
  },
  {
    title: "Build Your LLM",
    description: "Step-by-step guide to building a language model from scratch.",
    icon: <Code className="w-8 h-8" />,
    path: "/learn/llm-building/",
    color: "from-rose-500 to-pink-500",
    tag: "Project"
  }
];

export default function LearnPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-white">
      {/* Hero Section */}
      <section className="relative py-32 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(124,58,237,0.15),transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,rgba(99,102,241,0.15),transparent_50%)]" />
        
        <div className="relative z-10 max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center max-w-3xl mx-auto"
          >
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 text-violet-300 text-sm font-medium mb-6">
              <BookOpen className="w-4 h-4" />
              Learning Center
            </span>
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              Master <span className="text-violet-400">Language Models</span>
            </h1>
            <p className="text-xl text-slate-400 leading-relaxed">
              From beginner fundamentals to advanced architecture. 
              Learn at your own pace with interactive tutorials and hands-on projects.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Featured Modules */}
      <section className="py-20 border-y border-slate-800/50">
        <div className="max-w-7xl mx-auto px-6">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-3xl font-bold text-center mb-12"
          >
            Featured Learning Modules
          </motion.h2>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {featuredModules.map((module, i) => (
              <motion.div
                key={module.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
              >
                <Link
                  href={module.path}
                  className="group block h-full p-6 bg-slate-900/50 rounded-2xl border border-slate-700/50 hover:border-violet-500/50 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-br ${module.color} text-white`}>
                      {module.icon}
                    </div>
                    <span className="px-2 py-1 rounded-full bg-slate-800 text-slate-400 text-xs font-medium">
                      {module.tag}
                    </span>
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2 group-hover:text-violet-400 transition-colors">
                    {module.title}
                  </h3>
                  <p className="text-slate-400 text-sm mb-4">{module.description}</p>
                  <div className="flex items-center gap-1 text-violet-400 text-sm font-medium">
                    <span>Start Learning</span>
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Learning Paths */}
      <section className="py-32">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl font-bold mb-4">Choose Your Learning Path</h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Structured tracks designed to take you from complete beginner to advanced practitioner.
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-8">
            {learningPaths.map((path, i) => (
              <motion.div
                key={path.id}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="p-8 bg-slate-900/50 rounded-3xl border border-slate-700/50"
              >
                <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${path.color} flex items-center justify-center text-white mb-6`}>
                  {path.icon}
                </div>
                <h3 className="text-2xl font-bold text-white mb-3">{path.title}</h3>
                <p className="text-slate-400 mb-6">{path.description}</p>
                
                <div className="space-y-3">
                  {path.modules.map((module, j) => (
                    <Link
                      key={module.title}
                      href={module.path}
                      className="flex items-center gap-3 p-3 rounded-xl bg-slate-800/50 hover:bg-slate-800 transition-colors group"
                    >
                      <span className="w-6 h-6 rounded-full bg-slate-700 flex items-center justify-center text-xs text-slate-400 font-medium">
                        {j + 1}
                      </span>
                      <span className="text-slate-300 group-hover:text-white transition-colors">
                        {module.title}
                      </span>
                      <ArrowRight className="w-4 h-4 text-slate-500 ml-auto group-hover:text-violet-400 group-hover:translate-x-1 transition-all" />
                    </Link>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 border-t border-slate-800/50">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="p-12 bg-gradient-to-br from-violet-900/20 to-indigo-900/20 rounded-3xl border border-violet-500/20"
          >
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Build Your Own Model?
            </h2>
            <p className="text-slate-400 mb-8 max-w-xl mx-auto">
              Put your knowledge into practice. Create, train, and deploy your own language model.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                href="/models/"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold hover:shadow-lg hover:shadow-violet-500/25 transition-all"
              >
                <Brain className="w-5 h-5" />
                Create Model
              </Link>
              <Link
                href="/learn/transformer/"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-slate-800 text-white font-semibold hover:bg-slate-700 transition-all"
              >
                <BookOpen className="w-5 h-5" />
                Start Learning
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </main>
  );
}
