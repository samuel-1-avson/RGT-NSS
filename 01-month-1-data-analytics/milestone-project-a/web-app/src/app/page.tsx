"use client";

import Image from "next/image";
import { motion } from "framer-motion";
import { 
  Users, 
  Activity, 
  Heart, 
  Droplets, 
  FileText, 
  BarChart3, 
  Lightbulb, 
  Info,
  ChevronRight,
  TrendingUp,
  Stethoscope
} from "lucide-react";

import { Navbar } from "@/components/Navbar";
import { useState } from "react";
import Link from "next/link";
import { MethodologyModal as MethodModal } from "@/components/MethodologyModal";

export default function Home() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedStep, setSelectedStep] = useState<any>(null);

  const methodologySteps = [
    {
      id: 1,
      title: "Data Cleaning & Imputation",
      method: "Standardizing the UCI dataset for model-ready input.",
      tools: ["Python", "Pandas", "NumPy", "Scikit-Learn"],
      steps: [
        "Handling heart rate and cholesterol outliers through median imputation.",
        "Encoding categorical variables (Gender, Chest Pain type).",
        "Feature scaling for clinical consistency."
      ],
      reasoning: "Heart disease data often contains noise; cleaning ensured 100% data integrity for the 500 patients analyzed."
    },
    {
      id: 2,
      title: "Exploratory Data Analysis",
      method: "Multivariate analysis of heart disease risk factors.",
      tools: ["Matplotlib", "Seaborn", "Jupyter", "Scipy"],
      steps: [
        "Identifying primary indicators (Age, Angina, Oldpeak).",
        "Visualizing demographic disease distributions.",
        "Correlation mapping of risk factors."
      ],
      reasoning: "EDA revealed that patients over 55 and those with exercise-induced angina were the highest-risk cohorts."
    },
    {
      id: 3,
      title: "SQL Risk Factor Modeling",
      method: "Direct database profiling of patient registries.",
      tools: ["SQLite3", "Advanced SQL", "DBeaver"],
      steps: [
        "Aggregating prevalence rates across segments.",
        "Analyzing comorbid risk factors via JOINs.",
        "Predictive profile generation using CTEs."
      ],
      reasoning: "SQL allowed for granular filtering and heavy aggregation, creating a source of truth for the clinical dashboard."
    },
    {
      id: 4,
      title: "Visual Impact Storytelling",
      method: "UX-driven data presentation for stakeholders.",
      tools: ["Next.js", "Tailwind CSS4", "Framer Motion", "Lucide"],
      steps: [
        "Designing high-fidelity glassmorphic interfaces.",
        "Implementing staggered entrance animations.",
        "Ensuring visual hierarchy for clinical clarity."
      ],
      reasoning: "Translating raw SQL results into a premium UI makes the insights immediate and actionable for healthcare administrators."
    },
  ];

  const handleOpenModal = (step: any) => {
    setSelectedStep(step);
    setIsModalOpen(true);
  };

  const metrics = [
    { label: "Total Patients", value: "500", icon: <Users className="w-6 h-6" />, color: "bg-blue-500", text: "text-blue-600" },
    { label: "Heart Disease Rate", value: "55.0%", icon: <Heart className="w-6 h-6" />, color: "bg-red-500", text: "text-red-600" },
    { label: "Average Age", value: "53.0 yrs", icon: <Activity className="w-6 h-6" />, color: "bg-emerald-500", text: "text-emerald-600" },
    { label: "Avg Cholesterol", value: "349.1", unit: "mg/dl", icon: <Droplets className="w-6 h-6" />, color: "bg-amber-500", text: "text-amber-600" },
  ];

  const findings = [
    { text: "Focus screening on patients over 50 years old.", icon: <Users className="w-5 h-5" /> },
    { text: "Monitor cholesterol levels regularly for high-risk groups.", icon: <Activity className="w-5 h-5" /> },
    { text: "Implement lifestyle interventions for patients with risk factors.", icon: <Heart className="w-5 h-5" /> },
  ];

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { y: 20, opacity: 0 },
    show: { y: 0, opacity: 1 }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans selection:bg-blue-100">
      {/* Methodology Modal */}
      <MethodModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        step={selectedStep} 
      />

      {/* Dynamic Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10">
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-blue-100/30 rounded-full blur-[120px]" />
        <div className="absolute top-[20%] -right-[5%] w-[30%] h-[30%] bg-emerald-100/20 rounded-full blur-[100px]" />
      </div>

      <Navbar />

      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <motion.header 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-16 flex flex-col md:flex-row md:items-end justify-between gap-8"
        >
          <div className="max-w-2xl">
            <div className="flex items-center gap-2 mb-4">
              <span className="w-12 h-1 bg-blue-600 rounded-full" />
              <span className="text-[10px] uppercase font-bold tracking-[0.3em] text-blue-600">Milestone Project A</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-black tracking-tighter text-slate-900 mb-6 bg-clip-text text-transparent bg-gradient-to-r from-slate-900 via-slate-800 to-blue-900">
              Heart Disease <br />
              <span className="text-blue-600">Business Insights</span>
            </h1>
            <p className="text-lg text-slate-500 font-medium leading-relaxed">
              Synthesizing the UCI Heart Disease dataset to identify actionable risk factors and behavioral trends for clinical stakeholders.
            </p>
          </div>
          <div className="flex items-center gap-4 bg-white/50 p-2 rounded-2xl border border-white/50 backdrop-blur-sm">
            <div className="px-6 py-3 bg-slate-900 rounded-xl text-white text-center">
              <p className="text-[10px] uppercase font-bold tracking-widest text-slate-400 mb-1">Last Update</p>
              <p className="text-sm font-bold">Feb 2026</p>
            </div>
          </div>
        </motion.header>

        <main>
          {/* Metrics Overview */}
          <motion.div 
            variants={container}
            initial="hidden"
            animate="show"
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-16"
          >
            {metrics.map((metric, index) => (
              <motion.div 
                key={index} 
                variants={item}
                whileHover={{ y: -5 }}
                className="glass-card bg-white/80 p-6 rounded-3xl border border-white/60 shadow-[0_8px_30px_rgb(0,0,0,0.04)] group"
              >
                <div className="flex justify-between items-start mb-6">
                  <div className={`${metric.color} w-12 h-12 rounded-2xl flex items-center justify-center text-white shadow-lg shadow-${metric.color.split('-')[1]}-200 group-hover:scale-110 transition-transform`}>
                    {metric.icon}
                  </div>
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Live Static</div>
                </div>
                <p className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-1">{metric.label}</p>
                <div className="flex items-baseline gap-1">
                  <span className={`text-4xl font-black tracking-tight ${metric.text}`}>{metric.value}</span>
                  {metric.unit && <span className="text-xs font-bold text-slate-400 uppercase">{metric.unit}</span>}
                </div>
              </motion.div>
            ))}
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-10 mb-16">
            {/* Context Sidebar */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="space-y-10"
            >
              <section className="glass-card bg-slate-900 p-8 rounded-[2rem] text-white shadow-2xl relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-32 h-32 bg-blue-600/20 rounded-full -mr-16 -mt-16 blur-2xl group-hover:bg-blue-600/30 transition-all" />
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-2 h-6 bg-blue-500 rounded-full" />
                  <h2 className="text-xl font-bold">Project Overview</h2>
                </div>
                <p className="text-slate-400 text-sm leading-relaxed mb-8 font-medium">
                  This core initiative translates raw data from the UCI Heart Disease repository into actionable intelligence for clinical stakeholders using advanced SQL modeling and EDA.
                </p>
                <Link href="/reports">
                  <button className="flex items-center gap-2 group/btn text-sm font-bold text-white hover:text-blue-400 transition-colors">
                    VIEW DETAILS
                    <ChevronRight className="w-4 h-4 group-hover/btn:translate-x-1 transition-transform" />
                  </button>
                </Link>
              </section>

              <section className="glass-card bg-white/60 p-8 rounded-[2rem] border border-white/80 shadow-sm">
                <div className="flex items-center gap-3 mb-8">
                  <BarChart3 className="w-6 h-6 text-blue-600" />
                  <h2 className="text-xl font-black text-slate-800">Methodology</h2>
                </div>
                <div className="space-y-4">
                  {methodologySteps.map((step) => (
                    <button 
                      key={step.id} 
                      onClick={() => handleOpenModal(step)}
                      className="w-full flex items-center gap-5 p-4 rounded-2xl hover:bg-white hover:shadow-md transition-all group border border-transparent hover:border-slate-100"
                    >
                      <span className="w-8 h-8 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center font-bold text-xs group-hover:bg-blue-600 group-hover:text-white transition-all">
                        {step.id}
                      </span>
                      <span className="text-sm font-bold text-slate-600 tracking-tight group-hover:text-slate-900">{step.title}</span>
                    </button>
                  ))}
                </div>
              </section>
            </motion.div>

            {/* Visual Insights Area */}
            <div className="lg:col-span-2 space-y-10">
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="glass-card bg-white p-2 rounded-[2.5rem] shadow-[0_20px_50px_rgba(0,0,0,0.03)] border border-slate-100"
              >
                <div className="p-8">
                  <div className="flex justify-between items-center mb-8">
                    <h2 className="text-2xl font-black text-slate-800 tracking-tight">Disease Distribution</h2>
                    <div className="flex gap-2 text-[10px] font-bold uppercase tracking-widest text-slate-400">
                      <span>Prevalence Score</span>
                      <div className="w-4 h-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                      </div>
                    </div>
                  </div>
                  <div className="relative h-[400px] w-full bg-slate-50/50 rounded-3xl overflow-hidden border border-slate-100 group">
                    <Image 
                      src="/disease_distribution.png" 
                      alt="Disease Distribution" 
                      fill 
                      className="object-contain p-6 group-hover:scale-105 transition-transform duration-700"
                    />
                  </div>
                </div>
              </motion.div>

              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="glass-card bg-white p-2 rounded-[2.5rem] shadow-[0_20px_50px_rgba(0,0,0,0.03)] border border-slate-100"
              >
                <div className="p-8">
                  <div className="flex justify-between items-center mb-8">
                    <h2 className="text-2xl font-black text-slate-800 tracking-tight">Age Demographics</h2>
                    <div className="text-[10px] font-bold uppercase tracking-widest text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                      Risk Group B
                    </div>
                  </div>
                  <div className="relative h-[400px] w-full bg-slate-50/50 rounded-3xl overflow-hidden border border-slate-100 group">
                    <Image 
                      src="/age_by_disease.png" 
                      alt="Age Analysis" 
                      fill 
                      className="object-contain p-6 group-hover:scale-105 transition-transform duration-700"
                    />
                  </div>
                </div>
              </motion.div>
            </div>
          </div>

          {/* Recommendations Banner */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.7 }}
            className="relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-blue-600 rounded-[3rem]" />
            <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-white opacity-[0.03] rounded-full -translate-y-1/2 translate-x-1/2" />
            
            <div className="relative p-12 text-white">
              <div className="flex flex-col md:flex-row gap-12 items-center">
                <div className="flex-1">
                  <div className="bg-blue-400/20 w-16 h-16 rounded-2xl flex items-center justify-center mb-8 border border-blue-400/30">
                    <Lightbulb className="w-8 h-8 text-blue-200" />
                  </div>
                  <h2 className="text-3xl md:text-4xl font-black mb-4 tracking-tighter">Clinical Interventions</h2>
                  <p className="text-blue-100 text-lg leading-relaxed font-medium">
                    Based on our data models, the following evidence-based actions are recommended for implementation.
                  </p>
                </div>
                
                <div className="flex-1 grid grid-cols-1 gap-4 w-full">
                  {findings.map((finding, index) => (
                    <motion.div 
                      key={index}
                      whileHover={{ x: 10 }}
                      className="bg-white/10 backdrop-blur-md p-6 rounded-3xl border border-white/10 flex items-center gap-6"
                    >
                      <div className="w-12 h-12 rounded-2xl bg-white/10 flex items-center justify-center text-white shrink-0">
                        {finding.icon}
                      </div>
                      <p className="text-sm font-bold tracking-tight leading-snug">{finding.text}</p>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </main>

        <footer className="mt-32 pb-20 border-t border-slate-200 pt-16 flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-3 opacity-50">
            <div className="w-6 h-6 bg-slate-400 rounded flex items-center justify-center">
              <Stethoscope className="w-4 h-4 text-white" />
            </div>
            <span className="text-sm font-bold tracking-tight text-slate-600 uppercase">
              RGT 2026 <span className="text-slate-400 font-medium ml-1">Research Program</span>
            </span>
          </div>
          
          <div className="flex gap-12">
            {["System Stats", "Privacy Policy", "Cloud API"].map((item, i) => (
              <a key={i} href="#" className="text-xs font-bold text-slate-400 uppercase tracking-widest hover:text-blue-600 transition-colors">
                {item}
              </a>
            ))}
          </div>
          
          <div className="flex gap-4">
            <div className="w-10 h-10 rounded-full bg-white border border-slate-200 flex items-center justify-center text-slate-400 hover:text-blue-600 hover:border-blue-200 transition-all cursor-pointer">
              <BarChart3 className="w-5 h-5" />
            </div>
            <div className="w-10 h-10 rounded-full bg-white border border-slate-200 flex items-center justify-center text-slate-400 hover:text-blue-600 hover:border-blue-200 transition-all cursor-pointer">
              <FileText className="w-5 h-5" />
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
