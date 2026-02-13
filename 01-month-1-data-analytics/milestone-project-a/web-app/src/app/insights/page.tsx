"use client";

import { Navbar } from "@/components/Navbar";
import { motion } from "framer-motion";
import { 
  Lightbulb, 
  ShieldAlert, 
  Target, 
  BrainCircuit, 
  Search, 
  ChevronRight,
  TrendingUp,
  Stethoscope,
  Zap,
  ArrowRight
} from "lucide-react";

export default function InsightsPage() {
  const riskProfiles = [
    { 
      title: "Critical Cohort", 
      desc: "Age > 55, Chol > 240, BP > 140", 
      prob: "82%", 
      color: "bg-red-500",
      lightColor: "bg-red-50",
      textColor: "text-red-700",
      icon: <ShieldAlert className="w-5 h-5 text-red-500" />
    },
    { 
      title: "Exercise Impact", 
      desc: "Positive Exercise Angina + High ST Depression", 
      prob: "74%", 
      color: "bg-amber-500",
      lightColor: "bg-amber-50",
      textColor: "text-amber-700",
      icon: <Zap className="w-5 h-5 text-amber-500" />
    },
    { 
      title: "Asymptomatic Risk", 
      desc: "Silent chest pain signals with high heart rate", 
      prob: "61%", 
      color: "bg-blue-500",
      lightColor: "bg-blue-50",
      textColor: "text-blue-700",
      icon: <Search className="w-5 h-5 text-blue-500" />
    },
  ];

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans pb-20">
      <Navbar />

      <main className="max-w-7xl mx-auto px-6 py-12 md:py-16">
        <motion.header 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-16"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 bg-amber-100 rounded-2xl shadow-sm">
              <Lightbulb className="w-7 h-7 text-amber-600" />
            </div>
            <h1 className="text-4xl font-black tracking-tight text-slate-900">Clinical Insights & Risk Modeling</h1>
          </div>
          <p className="text-lg text-slate-500 max-w-3xl font-medium leading-relaxed">
            Advanced risk stratification and behavioral patterns identified through SQL modelling and Exploratory Data Analysis of the heart disease registry.
          </p>
        </motion.header>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-12">
          {/* Main Insights Content */}
          <div className="xl:col-span-8 space-y-12">
            
            {/* Predictive Risk Strata - Redesigned for width */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-card bg-white p-6 md:p-10 rounded-[3rem] border border-slate-100 shadow-[0_8px_30px_rgb(0,0,0,0.02)] relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 w-96 h-96 bg-blue-50 rounded-full -mr-32 -mt-32 blur-3xl opacity-50" />
              
              <div className="relative">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-10">
                  <h2 className="text-2xl font-black flex items-center gap-3 text-slate-800">
                    <Target className="w-6 h-6 text-red-600" />
                    Predictive Risk Strata
                  </h2>
                  <div className="flex items-center gap-2 text-xs font-bold text-slate-400 uppercase tracking-widest bg-slate-50 px-4 py-2 rounded-xl">
                    <span>Model Confidence: High</span>
                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                  {riskProfiles.map((profile, i) => (
                    <motion.div 
                      key={i} 
                      whileHover={{ y: -5 }}
                      className="p-8 bg-white border border-slate-100 rounded-[2rem] shadow-sm hover:shadow-xl hover:border-blue-100 transition-all group flex flex-col justify-between"
                    >
                      <div>
                        <div className={`${profile.lightColor} w-12 h-12 rounded-2xl flex items-center justify-center mb-6`}>
                          {profile.icon}
                        </div>
                        <h3 className="text-lg font-black text-slate-800 mb-3 tracking-tight">{profile.title}</h3>
                        <p className="text-sm text-slate-500 font-semibold leading-relaxed mb-8">{profile.desc}</p>
                      </div>
                      
                      <div className="pt-6 border-t border-slate-50">
                        <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Observation Prob.</p>
                        <p className={`text-4xl font-black ${profile.textColor}`}>{profile.prob}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </motion.div>

            {/* Pattern vs Strategy Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <section className="glass-card bg-[#0f172a] p-10 rounded-[3rem] text-white relative overflow-hidden group shadow-2xl">
                <div className="absolute top-0 right-0 w-48 h-48 bg-blue-600/10 rounded-full blur-3xl group-hover:bg-blue-600/20 transition-all" />
                <h2 className="text-xl font-extrabold mb-8 flex items-center gap-3">
                  <BrainCircuit className="w-6 h-6 text-blue-400" />
                  Key Pattern Identification
                </h2>
                <ul className="space-y-6">
                  {[
                    "Exercise-induced angina is a 2x clearer predictor than age alone.",
                    "The combination of low heart rate and high oldpeak represents clinical danger.",
                    "Metabolic markers show non-linear correlation with target states."
                  ].map((text, i) => (
                    <li key={i} className="flex gap-4 items-start">
                      <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center shrink-0 mt-0.5 group-hover:bg-blue-600/20 transition-colors">
                        <span className="text-blue-400 text-xs font-bold leading-none">{i + 1}</span>
                      </div>
                      <p className="text-sm text-slate-300 font-medium leading-relaxed">{text}</p>
                    </li>
                  ))}
                </ul>
              </section>

              <section className="glass-card bg-white p-10 rounded-[3rem] border border-slate-100 shadow-sm hover:shadow-md transition-all">
                <h2 className="text-xl font-extrabold text-slate-800 mb-8 flex items-center gap-3">
                  <TrendingUp className="w-6 h-6 text-blue-600" />
                  Strategy Optimization
                </h2>
                <div className="space-y-6">
                  <div className="p-6 bg-emerald-50 rounded-[2rem] border border-emerald-100 group">
                    <div className="flex items-center gap-2 mb-3">
                      <Zap className="w-4 h-4 text-emerald-600" />
                      <p className="text-xs font-bold text-emerald-700 uppercase tracking-widest">Resource Allocation</p>
                    </div>
                    <p className="text-sm font-bold text-emerald-900 leading-snug">Focus preventative budget on the 55+ demographic with comorbid risk factors.</p>
                  </div>
                  <div className="p-6 bg-blue-50 rounded-[2rem] border border-blue-100">
                    <div className="flex items-center gap-2 mb-3">
                      <Stethoscope className="w-4 h-4 text-blue-600" />
                      <p className="text-xs font-bold text-blue-700 uppercase tracking-widest">Screening Protocol</p>
                    </div>
                    <p className="text-sm font-bold text-blue-900 leading-snug">Implement mandatory stress tests for patients showing asymptomatic chest pain patterns.</p>
                  </div>
                </div>
              </section>
            </div>
          </div>

          {/* Sidebar Area - Improved for readability */}
          <div className="xl:col-span-4 space-y-10">
            <div className="bg-gradient-to-br from-[#1e40af] to-[#312e81] p-10 rounded-[3rem] text-white shadow-2xl shadow-blue-900/10 relative overflow-hidden group">
              <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-white/10 rounded-full blur-2xl group-hover:scale-125 transition-transform" />
              <Stethoscope className="w-12 h-12 text-blue-200 mb-8" />
              <h3 className="text-2xl font-black mb-4 tracking-tight leading-none">Clinical Toolkit</h3>
              <p className="text-blue-100/80 text-base font-medium leading-relaxed mb-10">
                Download the complete derived dataset and high-fidelity slide deck for hospital admin review.
              </p>
              <button className="w-full py-5 bg-white text-blue-700 rounded-2xl text-sm font-black uppercase tracking-widest flex items-center justify-center gap-3 hover:bg-blue-50 transition-all shadow-xl active:scale-95 group">
                Download Full PPT
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>

            <div className="glass-card bg-white p-10 rounded-[3rem] border border-slate-100 shadow-sm relative group overflow-hidden">
              <div className="absolute top-0 right-0 w-2 h-full bg-amber-400 group-hover:w-3 transition-all" />
              <div className="flex items-center gap-3 mb-8">
                <div className="w-10 h-10 bg-amber-100 rounded-xl flex items-center justify-center">
                  <Zap className="w-5 h-5 text-amber-600" />
                </div>
                <h4 className="text-sm font-bold text-slate-800 uppercase tracking-widest">System Status</h4>
              </div>
              <p className="text-xl font-black text-slate-800 mb-6 leading-tight tracking-tight">Insight generation is 100% automated.</p>
              <div className="space-y-4">
                <div className="flex items-center gap-3 text-[11px] font-bold text-slate-500 tracking-tight">
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]" />
                  Models synced with SQLite
                </div>
                <div className="flex items-center gap-3 text-[11px] font-bold text-slate-500 tracking-tight">
                  <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                  EDA Phase: Complete
                </div>
              </div>
            </div>

            <div className="p-8 bg-blue-50/50 rounded-[2.5rem] border border-blue-100 text-center">
              <p className="text-xs font-bold text-blue-600 uppercase tracking-widest mb-2">Need Custom Analysis?</p>
              <p className="text-sm text-slate-500 font-medium mb-6 leading-relaxed">Our data scientists are available for specific cohort research.</p>
              <button className="text-blue-700 font-black text-sm hover:underline flex items-center gap-2 mx-auto">
                Request Custom API <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
