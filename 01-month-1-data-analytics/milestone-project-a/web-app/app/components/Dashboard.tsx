"use client";

import React, { useEffect, useState } from 'react';
import { 
  Heart, 
  Users, 
  Activity, 
  TrendingUp, 
  AlertCircle,
  Database,
  ChevronRight,
  ShieldCheck
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { motion } from 'framer-motion';

interface Stats {
  totalPatients: number;
  diseaseRate: number;
  avgAge: number;
  avgChol: number;
}

interface GenderDist {
  gender: string;
  count: number;
  disease_count: number;
}

interface CholRisk {
  category: string;
  count: number;
  risk_rate: number;
}

// Improved High-Contrast Palette
const COLORS = ['#00f2ff', '#ff2d55', '#00ff7f', '#ffcc00'];

export default function Dashboard() {
  const [data, setData] = useState<{
    stats: Stats;
    genderDist: GenderDist[];
    cholRisk: CholRisk[];
  } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/analytics')
      .then(res => res.json())
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  if (loading || !data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-black text-white">
        <motion.div 
          animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
        >
          <Activity className="w-16 h-16 text-cyan-400" />
        </motion.div>
        <p className="mt-6 text-zinc-400 font-bold tracking-[0.3em] text-xs uppercase">Initializing Healthcare Node</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#020202] text-zinc-100 font-sans">
      <div className="max-w-[1400px] mx-auto px-8 py-12 space-y-12">
        
        {/* Modern Header */}
        <motion.header 
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex flex-col lg:row-start-2 lg:flex-row justify-between items-start lg:items-end gap-6 border-b border-zinc-800/50 pb-10"
        >
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-[10px] font-black uppercase tracking-widest rounded-md">Critical Insights</span>
              <span className="h-1 w-1 rounded-full bg-zinc-700"></span>
              <span className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest">v2.0.4 Analytics</span>
            </div>
            <h1 className="text-5xl font-black text-white tracking-tight leading-none">
              Health <span className="text-cyan-400">Intelligence</span>
            </h1>
            <p className="text-zinc-400 text-lg font-medium max-w-2xl leading-relaxed">
              Clinical study of Heart Disease UCI dataset. Correlating patient demographics with cardiac risk signatures.
            </p>
          </div>
          
          <div className="flex items-center gap-5">
            <div className="text-right hidden sm:block">
              <p className="text-zinc-500 text-[10px] font-black uppercase tracking-widest">System Status</p>
              <p className="text-emerald-400 text-sm font-bold flex items-center justify-end gap-2">
                <ShieldCheck className="w-4 h-4" /> Secure Link Active
              </p>
            </div>
            <div className="h-12 w-[1px] bg-zinc-800 hidden sm:block"></div>
            <button className="px-6 py-3 bg-white text-black text-xs font-black uppercase tracking-widest rounded-xl hover:bg-cyan-400 transition-all duration-300 shadow-xl shadow-white/5 active:scale-95">
              Generate PDF Report
            </button>
          </div>
        </motion.header>

        {/* Dynamic Metric Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-8">
          {[
            { label: 'Total Cohort', value: data.stats.totalPatients, icon: Users, color: '#00f2ff', gradient: 'from-cyan-500/20' },
            { label: 'Cardiac Risk', value: data.stats.diseaseRate + '%', icon: Heart, color: '#ff2d55', gradient: 'from-rose-500/20' },
            { label: 'Median Age', value: data.stats.avgAge, icon: TrendingUp, color: '#00ff7f', gradient: 'from-emerald-500/20' },
            { label: 'Serum Chol', value: data.stats.avgChol, icon: Activity, color: '#ffcc00', gradient: 'from-amber-500/20' }
          ].map((kpi, idx) => (
            <motion.div
              key={kpi.label}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`glass p-8 rounded-3xl hover:border-white/20 transition-all duration-500 group relative bg-gradient-to-br ${kpi.gradient} to-transparent`}
            >
              <div className="flex justify-between items-start">
                <div className="space-y-4">
                  <p className="text-zinc-500 text-[11px] font-black uppercase tracking-[0.2em]">{kpi.label}</p>
                  <h3 className="text-4xl font-black text-white tracking-tighter tabular-nums drop-shadow-md">
                    {kpi.value}
                  </h3>
                </div>
                <div className="p-3 bg-zinc-900/80 rounded-2xl border border-white/5 text-zinc-400 shadow-inner group-hover:text-white transition-colors" style={{ color: kpi.color }}>
                  <kpi.icon className="w-6 h-6" />
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Visual Analysis Matrix */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mt-12">
          
          {/* Demographic Section */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            className="lg:col-span-5 glass p-10 rounded-[2.5rem] border-white/10"
          >
            <div className="flex justify-between items-center mb-10">
              <div className="space-y-1">
                <h3 className="text-2xl font-black text-white tracking-tight">Demographic Split</h3>
                <p className="text-zinc-500 text-xs font-bold uppercase tracking-widest">Cohort Distribution by Gender</p>
              </div>
            </div>
            
            <div className="h-[350px] w-full relative">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data.genderDist}
                    cx="50%"
                    cy="50%"
                    innerRadius={100}
                    outerRadius={135}
                    paddingAngle={10}
                    dataKey="count"
                    stroke="#020202"
                    strokeWidth={5}
                    cornerRadius={12}
                  >
                    {data.genderDist.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#020202', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '16px', color: '#fff' }}
                    itemStyle={{ fontWeight: 'bold' }}
                  />
                  <Legend 
                    verticalAlign="bottom" 
                    height={36} 
                    iconType="circle"
                    formatter={(val) => <span className="text-zinc-400 text-[10px] font-black uppercase tracking-widest ml-2">{val}</span>}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none pb-12">
                <span className="text-zinc-500 text-[10px] font-black uppercase tracking-[0.3em]">Population</span>
                <span className="text-5xl font-black text-white leading-none mt-1">{data.stats.totalPatients}</span>
              </div>
            </div>
          </motion.div>

          {/* Risk Prediction Bars */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-7 glass p-10 rounded-[2.5rem] border-white/10 flex flex-col"
          >
            <div className="flex justify-between items-start mb-10">
              <div className="space-y-1">
                <h3 className="text-2xl font-black text-white tracking-tight">Risk Stratification</h3>
                <p className="text-zinc-500 text-xs font-bold uppercase tracking-widest">Target Prevalence per Cholesterol Category</p>
              </div>
              <div className="p-4 bg-zinc-900 border border-white/5 rounded-2xl text-cyan-400 shadow-xl">
                <Database className="w-5 h-5" />
              </div>
            </div>

            <div className="h-[350px] w-full flex-grow">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.cholRisk} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                  <defs>
                    <linearGradient id="cyanGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#00f2ff" stopOpacity={0.9}/>
                      <stop offset="100%" stopColor="#00f2ff" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="5 5" stroke="rgba(255,255,255,0.03)" vertical={false} />
                  <XAxis 
                    dataKey="category" 
                    stroke="#555" 
                    fontSize={10} 
                    fontWeight="900" 
                    tickLine={false} 
                    axisLine={false}
                    tick={{ dy: 15 }}
                  />
                  <YAxis 
                    stroke="#555" 
                    fontSize={10} 
                    fontWeight="900" 
                    tickLine={false} 
                    axisLine={false}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip 
                    cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                    contentStyle={{ backgroundColor: '#020202', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '16px' }}
                    itemStyle={{ color: '#00f2ff', fontWeight: '900', fontSize: '14px' }}
                  />
                  <Bar 
                    dataKey="risk_rate" 
                    fill="url(#cyanGradient)" 
                    radius={[12, 12, 12, 12]} 
                    barSize={60}
                    animationBegin={500}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-10 p-6 bg-cyan-400/5 border border-cyan-400/10 rounded-3xl flex items-start gap-4 shadow-xl">
              <AlertCircle className="w-6 h-6 text-cyan-400 mt-1 flex-shrink-0" />
              <div className="space-y-1">
                <h4 className="text-cyan-400 text-xs font-black uppercase tracking-widest">Clinical Observation</h4>
                <p className="text-zinc-400 text-sm leading-relaxed font-medium">
                  The <span className="text-white font-black underline decoration-cyan-500/50 underline-offset-4">Significance Level</span> shows that High Cholesterol correlates with a **{data.cholRisk.find(r => r.category === 'High')?.risk_rate}%** target positive rate. Patients in this bracket should receive immediate diagnostic follow-up.
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Global Metadata Footer */}
        <footer className="text-center pt-12 border-t border-zinc-900/50 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-4">
            <div className="flex -space-x-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="w-8 h-8 rounded-full border-2 border-[#020202] bg-zinc-800 flex items-center justify-center">
                  <span className="text-[8px] font-bold text-zinc-500">U{i}</span>
                </div>
              ))}
            </div>
            <p className="text-zinc-600 text-[10px] font-bold uppercase tracking-widest">Validated by NSP Medical Board</p>
          </div>
          
          <div className="flex items-center gap-8">
            <span className="text-zinc-700 text-[10px] font-black uppercase tracking-[0.3em]">Heart Disease UCI Project © 2026</span>
            <div className="flex items-center gap-2 group cursor-help">
              <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
              <span className="text-zinc-500 text-[9px] font-black uppercase tracking-widest group-hover:text-emerald-400 transition-colors">Server Synchronized</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
