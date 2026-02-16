"use client";

import { Navbar } from "@/components/Navbar";
import { motion } from "framer-motion";
import { 
  ShieldCheck, 
  Cpu, 
  Database, 
  BarChart2, 
  Info, 
  History, 
  ExternalLink,
  Table as TableIcon,
  CircleDot
} from "lucide-react";

export default function ModelCardsPage() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const item = {
    hidden: { y: 20, opacity: 0 },
    show: { y: 0, opacity: 1 }
  };

  const metrics = [
    { label: "Accuracy", value: "80.4%", description: "Overall prediction correctness" },
    { label: "ROC-AUC", value: "0.83", description: "Classifier discriminative power" },
    { label: "Recall", value: "54.0%", description: "Capture rate of actual churners" },
    { label: "Precision", value: "68.0%", description: "Reliability of churn alerts" },
  ];

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 pt-40 pb-20 px-6">
      <Navbar />

      <main className="max-w-5xl mx-auto">
        {/* Header */}
        <motion.header 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-16"
        >
          <div className="flex items-center gap-3 mb-6">
            <ShieldCheck className="w-8 h-8 text-emerald-500" />
            <span className="text-sm font-black text-slate-400 uppercase tracking-[0.4em]">Official Specification</span>
          </div>
          <h1 className="text-5xl font-black tracking-tighter text-slate-900 mb-6">
            Model Card: <br />
            <span className="text-emerald-500">Churn Predictor v1.0</span>
          </h1>
          <p className="text-lg text-slate-500 font-medium leading-relaxed max-w-2xl">
            Technical documentation for the Telco Customer Churn Prediction model, following the Model Cards for Model Reporting framework.
          </p>
        </motion.header>

        {/* Technical Overview Grid */}
        <motion.div 
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-20"
        >
          {metrics.map((m, i) => (
            <motion.div key={i} variants={item} className="glass-card bg-white p-8">
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">{m.label}</p>
              <h3 className="text-3xl font-black text-emerald-500 mb-2">{m.value}</h3>
              <p className="text-[10px] font-bold text-slate-400 leading-tight">{m.description}</p>
            </motion.div>
          ))}
        </motion.div>

        <section className="space-y-12">
          {/* Model Architecture */}
          <div className="glass-card bg-white p-12">
            <div className="flex items-center gap-4 mb-10">
              <div className="w-12 h-12 bg-slate-100 rounded-2xl flex items-center justify-center">
                <Cpu className="w-6 h-6 text-slate-900" />
              </div>
              <h2 className="text-2xl font-black text-slate-900">Architecture & Config</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
              <div className="space-y-6">
                <div>
                  <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Model Type</h4>
                  <p className="text-sm font-bold text-slate-700">Random Forest Classifier (Ensemble)</p>
                </div>
                <div>
                  <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Hyperparameters</h4>
                  <ul className="space-y-2">
                    {[
                      "n_estimators: 200",
                      "max_depth: 20",
                      "min_samples_split: 2"
                    ].map((h, i) => (
                      <li key={i} className="flex items-center gap-3 text-xs font-bold text-slate-500">
                        <CircleDot className="w-2 h-2 text-emerald-500" />
                        {h}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="bg-slate-50 rounded-3xl p-8 border border-slate-100">
                <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                  <TableIcon className="w-3 h-3" />
                  Preprocessing Pipeline
                </h4>
                <div className="space-y-4">
                  <div className="p-4 bg-white rounded-xl border border-slate-200">
                    <p className="text-xs font-black mb-1">StandardScaler</p>
                    <p className="text-[10px] text-slate-400 font-bold">Applied to numeric features (tenure, charges)</p>
                  </div>
                  <div className="p-4 bg-white rounded-xl border border-slate-200">
                    <p className="text-xs font-black mb-1">One-Hot Encoding</p>
                    <p className="text-[10px] text-slate-400 font-bold">Categorical expansion with unknown handling</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Training Data */}
          <div className="glass-card bg-white p-12">
            <div className="flex items-center gap-4 mb-10">
              <div className="w-12 h-12 bg-slate-100 rounded-2xl flex items-center justify-center">
                <Database className="w-6 h-6 text-slate-900" />
              </div>
              <h2 className="text-2xl font-black text-slate-900">Dataset & Lineage</h2>
            </div>
            
            <div className="flex flex-col md:flex-row gap-8 items-start justify-between">
              <div className="max-w-sm">
                <p className="text-sm text-slate-500 font-medium leading-relaxed mb-6">
                  Trained on the **IBM Telco Customer Churn** dataset (Kaggle), consisting of 7,043 customer records with 21 distinct features.
                </p>
                <button className="flex items-center gap-2 text-xs font-black text-emerald-600 hover:text-emerald-700 transition-colors uppercase tracking-widest">
                  View Dataset Source
                  <ExternalLink className="w-3 h-3" />
                </button>
              </div>
              <div className="w-full md:w-auto grid grid-cols-2 gap-4">
                <div className="px-6 py-4 bg-emerald-50 rounded-2xl border border-emerald-100 text-center">
                  <p className="text-[10px] font-black text-emerald-700 uppercase mb-1">Samples</p>
                  <p className="text-lg font-black text-slate-900">7,043</p>
                </div>
                <div className="px-6 py-4 bg-emerald-50 rounded-2xl border border-emerald-100 text-center">
                  <p className="text-[10px] font-black text-emerald-700 uppercase mb-1">Features</p>
                  <p className="text-lg font-black text-slate-900">19</p>
                </div>
              </div>
            </div>
          </div>

          {/* Ethical Considerations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
            <div className="glass-card bg-slate-900 p-10 text-white">
              <div className="flex items-center gap-3 mb-8">
                <Info className="w-5 h-5 text-emerald-400" />
                <h3 className="text-xl font-bold">Ethical Oversight</h3>
              </div>
              <p className="text-sm text-slate-400 font-medium leading-relaxed mb-8">
                No sensitive demographic attributes (race, specific age, location) are used as primary drivers. Fairness auditing reveals a ±2% variance in recall across gender segments.
              </p>
              <div className="px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-xl">
                <p className="text-[10px] font-black text-emerald-400 uppercase tracking-widest">Status: Compliant</p>
              </div>
            </div>

            <div className="glass-card bg-white p-10">
              <div className="flex items-center gap-3 mb-8">
                <History className="w-5 h-5 text-slate-400" />
                <h3 className="text-xl font-black text-slate-900">Version History</h3>
              </div>
              <div className="space-y-6">
                {[
                  { v: "1.0.0", date: "Jan 2024", desc: "Initial production release" },
                  { v: "0.9.5-beta", date: "Nov 2023", desc: "Extended service feature testing" }
                ].map((v, i) => (
                  <div key={i} className="flex justify-between items-center pb-4 border-b border-slate-50 last:border-0 last:pb-0">
                    <div>
                      <p className="text-sm font-black text-slate-900">{v.v}</p>
                      <p className="text-[10px] font-bold text-slate-400">{v.desc}</p>
                    </div>
                    <span className="text-[10px] font-black text-slate-400 uppercase">{v.date}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Footer Link */}
        <div className="mt-20 text-center">
          <p className="text-xs font-medium text-slate-400 mb-6">Want deeper technical logs?</p>
          <button className="px-8 py-3 bg-slate-50 border border-slate-200 text-slate-900 rounded-2xl font-black text-xs hover:bg-slate-100 transition-all uppercase tracking-widest">
            Access System API Docs
          </button>
        </div>
      </main>
    </div>
  );
}
