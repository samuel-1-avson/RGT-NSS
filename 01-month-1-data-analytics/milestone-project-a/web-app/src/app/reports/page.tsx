"use client";

import { Navbar } from "@/components/Navbar";
import { motion } from "framer-motion";
import { 
  Users, 
  Venus, 
  Mars, 
  PieChart, 
  BarChart, 
  ArrowUpRight, 
  Calendar,
  Layers,
  Activity
} from "lucide-react";

export default function ReportsPage() {
  const genderData = [
    { label: "Female", rate: "56.0%", icon: <Venus className="w-5 h-5" />, color: "bg-pink-500", trend: "+1.2%" },
    { label: "Male", rate: "53.8%", icon: <Mars className="w-5 h-5" />, color: "bg-blue-500", trend: "-0.5%" },
  ];

  const ageData = [
    { group: "Under 50", rate: "59.3%", status: "High Risk" },
    { group: "50-60", rate: "47.0%", status: "Moderate" },
    { group: "Over 60", rate: "56.3%", status: "Significant" },
  ];

  const clinicalMetrics = [
    { label: "Avg Blood Pressure", value: "145.8", unit: "mmHg", icon: <Activity className="w-5 h-5" /> },
    { label: "Avg ST Depression", value: "1.1", unit: "unit", icon: <BarChart className="w-5 h-5" /> },
    { label: "Sample Coverage", value: "100%", unit: "valid", icon: <Layers className="w-5 h-5" /> },
  ];

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans">
      <Navbar />

      <main className="max-w-7xl mx-auto px-6 py-12">
        <motion.header 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-100 rounded-lg">
              <PieChart className="w-6 h-6 text-blue-600" />
            </div>
            <h1 className="text-3xl font-black tracking-tight">Statistical Population Report</h1>
          </div>
          <p className="text-slate-500 max-w-2xl font-medium">
            Detailed categorical breakdown of heart disease prevalence across different demographic segments within the 500-patient dataset.
          </p>
        </motion.header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          {/* Gender Analysis */}
          <motion.section 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-1"
          >
            <div className="glass-card bg-white p-8 rounded-[2.5rem] border border-slate-100 shadow-sm h-full">
              <h2 className="text-xl font-bold mb-8 flex items-center gap-2">
                <Users className="w-5 h-5 text-blue-600" />
                Gender Distribution
              </h2>
              <div className="space-y-6">
                {genderData.map((data, i) => (
                  <div key={i} className="p-6 bg-slate-50 rounded-3xl border border-slate-100 group hover:border-blue-200 transition-colors">
                    <div className="flex justify-between items-center mb-4">
                      <div className={`${data.color} w-10 h-10 rounded-xl flex items-center justify-center text-white`}>
                        {data.icon}
                      </div>
                      <span className="text-[10px] font-bold text-emerald-600 bg-emerald-50 px-2 py-1 rounded-md">
                        {data.trend}
                      </span>
                    </div>
                    <div>
                      <p className="text-sm font-bold text-slate-400 uppercase tracking-widest">{data.label}</p>
                      <p className="text-4xl font-black text-slate-800 tracking-tighter">{data.rate}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.section>

          {/* Age Group Analysis */}
          <motion.section 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="lg:col-span-2 space-y-10"
          >
            <div className="glass-card bg-white p-8 rounded-[2.5rem] border border-slate-100 shadow-sm">
              <div className="flex justify-between items-center mb-8">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-blue-600" />
                  Age-Based Risk Levels
                </h2>
                <div className="flex items-center gap-2 text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                  <span>Last Updated: Feb 2026</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {ageData.map((item, i) => (
                  <div key={i} className="relative overflow-hidden p-6 rounded-3xl border border-slate-100 bg-white hover:shadow-lg transition-all">
                    <div className="absolute top-0 right-0 p-4">
                      <ArrowUpRight className="w-4 h-4 text-slate-300" />
                    </div>
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">{item.group}</p>
                    <p className="text-3xl font-black text-slate-800 mb-6">{item.rate}</p>
                    <div className={`inline-block px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest ${
                      item.status === 'High Risk' ? 'bg-red-50 text-red-600' : 
                      item.status === 'Significant' ? 'bg-orange-50 text-orange-600' : 
                      'bg-blue-50 text-blue-600'
                    }`}>
                      {item.status}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Clinical Highlights Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {clinicalMetrics.map((metric, i) => (
                <div key={i} className="glass-card bg-white p-6 rounded-[2rem] border border-slate-100 shadow-sm flex items-center gap-5">
                  <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center">
                    {metric.icon}
                  </div>
                  <div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{metric.label}</p>
                    <div className="flex items-baseline gap-1">
                      <span className="text-xl font-black text-slate-800">{metric.value}</span>
                      <span className="text-[10px] font-bold text-slate-500 uppercase">{metric.unit}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.section>
        </div>

        {/* Footer info message */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-16 p-8 bg-blue-600 rounded-[2.5rem] text-white flex flex-col md:flex-row justify-between items-center gap-6"
        >
          <div>
            <h3 className="text-xl font-bold mb-1">Looking for deeper analysis?</h3>
            <p className="text-blue-100 text-sm font-medium">Switch to the Insights tab to see clinical profiles and risk modeling.</p>
          </div>
          <button className="bg-white text-blue-600 px-8 py-3 rounded-2xl text-sm font-bold hover:bg-blue-50 transition-colors shrink-0">
            Go to Insights
          </button>
        </motion.div>
      </main>
    </div>
  );
}
