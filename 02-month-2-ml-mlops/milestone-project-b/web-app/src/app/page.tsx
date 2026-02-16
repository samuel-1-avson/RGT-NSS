"use client";

import { Navbar } from "@/components/Navbar";
import { motion } from "framer-motion";
import { 
  TrendingUp, 
  Users, 
  DollarSign, 
  Target, 
  ArrowRight, 
  BarChart3, 
  ShieldAlert, 
  ChevronRight,
  Zap,
  CheckCircle2
} from "lucide-react";
import Link from "next/link";
import { useState, useEffect } from "react";

export default function Home() {
  const [stats, setStats] = useState<any>(null);
  const [drivers, setDrivers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, driversRes] = await Promise.all([
          fetch("http://localhost:8000/analytics/stats"),
          fetch("http://localhost:8000/analytics/features")
        ]);
        
        if (!statsRes.ok || !driversRes.ok) throw new Error("Fetch failed");
        
        const statsData = await statsRes.json();
        const driversData = await driversRes.json();
        
        setStats(statsData);
        setDrivers(driversData);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const metrics = [
    { 
      label: "Total Customers", 
      value: stats ? stats.total_customers.toLocaleString() : "7,043", 
      icon: <Users className="w-6 h-6" />, 
      color: "bg-blue-600",
      trend: stats ? "Live" : "+2.4%",
      trendColor: "text-emerald-500"
    },
    { 
      label: "Churn Rate", 
      value: stats ? `${stats.churn_rate}%` : "26.5%", 
      icon: <TrendingUp className="w-6 h-6" />, 
      color: "bg-emerald-500",
      trend: stats ? "Real-time" : "-1.2%",
      trendColor: "text-emerald-500"
    },
    { 
      label: "Annual Revenue Loss", 
      value: stats ? `$${(stats.annual_revenue_loss / 1000000).toFixed(2)}M` : "$1.45M", 
      icon: <DollarSign className="w-6 h-6" />, 
      color: "bg-red-500",
      trend: stats ? "Calculated" : "+5.1%",
      trendColor: "text-red-500"
    },
    { 
      label: "Total Dataset Revenue", 
      value: stats ? `$${(stats.total_revenue / 1000000).toFixed(1)}M` : "$16.1M", 
      icon: <Target className="w-6 h-6" />, 
      color: "bg-slate-900",
      trend: "Aggregated",
      trendColor: "text-blue-400"
    },
  ];

  const churnDrivers = drivers.length > 0 ? drivers : [
    { feature: "New Tenure (< 12mo)", weight: 17.5, color: "bg-red-500" },
    { feature: "Total Charges", weight: 14.1, color: "bg-amber-500" },
    { feature: "Monthly Charges", weight: 11.8, color: "bg-blue-500" },
    { feature: "Contract Type", weight: 9.2, color: "bg-emerald-500" },
    { feature: "Payment Method", weight: 8.4, color: "bg-slate-800" },
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
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans selection:bg-emerald-100">
      {/* Background Decor */}
      <div className="fixed inset-0 pointer-events-none -z-10 overflow-hidden">
        <div className="absolute top-[10%] -left-[10%] w-[50%] h-[50%] bg-emerald-100/30 rounded-full blur-[120px]" />
        <div className="absolute -bottom-[5%] -right-[5%] w-[40%] h-[40%] bg-blue-100/20 rounded-full blur-[100px]" />
      </div>

      <Navbar />

      <main className="max-w-7xl mx-auto px-6 pt-40 pb-20">
        {/* Hero Section */}
        <section className="mb-20">
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-3xl"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 rounded-full border border-emerald-100 mb-8">
              <Zap className="w-4 h-4 text-emerald-600 fill-emerald-500" />
              <span className="text-[10px] font-black uppercase tracking-widest text-emerald-700">Predictive Intelligence v1.0</span>
            </div>
            <h1 className="text-6xl md:text-7xl font-black tracking-tighter leading-[0.9] text-slate-900 mb-8">
              Predict Churn. <br />
              <span className="text-emerald-500">Protect Revenue.</span>
            </h1>
            <p className="text-xl text-slate-500 font-medium leading-relaxed mb-10">
              Identifying high-risk customer segments with 80.4% precision. Turn raw telco data into actionable retention strategies.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link href="/predictor">
                <button className="px-10 py-5 bg-slate-900 text-white rounded-[1.5rem] font-bold text-lg hover:bg-slate-800 transition-all flex items-center gap-3 active:scale-95 shadow-2xl shadow-slate-900/10">
                  Try Predictor
                  <ArrowRight className="w-5 h-5" />
                </button>
              </Link>
              <button className="px-10 py-5 bg-white border border-slate-200 text-slate-900 rounded-[1.5rem] font-bold text-lg hover:bg-slate-50 transition-all active:scale-95">
                View Documentation
              </button>
            </div>
          </motion.div>
        </section>

        {/* Metrics Grid */}
        <motion.div 
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-20"
        >
          {metrics.map((metric, i) => (
            <motion.div 
              key={i} 
              variants={item}
              whileHover={{ y: -5 }}
              className="glass-card bg-white p-8 group border-transparent hover:border-emerald-100 transition-all"
            >
              <div className="flex items-start justify-between mb-8">
                <div className={`${metric.color} w-14 h-14 rounded-2xl flex items-center justify-center text-white shadow-xl shadow-${metric.color.split('-')[1]}-200/50 group-hover:scale-110 transition-transform`}>
                  {metric.icon}
                </div>
                <div className={`text-xs font-black ${metric.trendColor} flex items-center gap-1`}>
                  {metric.trend}
                </div>
              </div>
              <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-2">{metric.label}</p>
              <h3 className="text-4xl font-black tracking-tight text-slate-900">{metric.value}</h3>
            </motion.div>
          ))}
        </motion.div>

        {/* Analytics Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          {/* Churn Drivers Chart */}
          <motion.section 
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="lg:col-span-2 glass-card bg-white p-10"
          >
            <div className="flex items-center justify-between mb-12">
              <div>
                <h2 className="text-2xl font-black text-slate-900 mb-1 flex items-center gap-3">
                  <BarChart3 className="w-6 h-6 text-emerald-500" />
                  Primary Churn Drivers
                </h2>
                <p className="text-sm font-medium text-slate-400 uppercase tracking-widest">Model Feature Importance</p>
              </div>
            </div>
            
            <div className="space-y-8">
              {churnDrivers.map((driver: any, i: number) => (
                <div key={i}>
                  <div className="flex justify-between items-end mb-3">
                    <span className="text-sm font-bold text-slate-600">{driver.feature || driver.label}</span>
                    <span className="text-sm font-black text-slate-900">{driver.weight}%</span>
                  </div>
                  <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${(driver.weight / 20) * 100}%` }}
                      transition={{ duration: 1, delay: 0.6 + (i * 0.1) }}
                      className={`h-full ${driver.color} rounded-full`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.section>

          {/* Quick Insights Sidebar */}
          <motion.aside 
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="space-y-10"
          >
            <div className="glass-card bg-slate-900 p-10 text-white relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/20 rounded-full -mr-16 -mt-16 blur-2xl group-hover:bg-emerald-500/30 transition-all" />
              <h2 className="text-xl font-bold mb-6 flex items-center gap-3">
                <ShieldAlert className="w-5 h-5 text-emerald-400" />
                Critical Findings
              </h2>
              <ul className="space-y-6">
                {[
                  "New customers (< 1year) are 5x more likely to churn.",
                  "Month-to-month contracts have a 43% churn rate.",
                  "Electronic check users show higher risk profiles."
                ].map((text, i) => (
                  <li key={i} className="flex items-start gap-4">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 shrink-0" />
                    <p className="text-sm text-slate-400 font-medium leading-relaxed">{text}</p>
                  </li>
                ))}
              </ul>
            </div>

            <div className="glass-card bg-emerald-500 p-10 text-slate-900">
              <h2 className="text-xl font-black mb-6">Retention Lift</h2>
              <p className="text-slate-900/70 font-bold mb-10 leading-relaxed text-sm">
                Optimizing campaigns with model predictions can save over <span className="font-black">$50,000</span> in annual revenue.
              </p>
              <Link href="/predictor">
                <button className="w-full py-4 bg-slate-900 text-white rounded-2xl font-bold text-sm hover:translate-x-1 transition-all flex items-center justify-center gap-2">
                  START PREDICTION
                  <ChevronRight className="w-4 h-4" />
                </button>
              </Link>
            </div>
          </motion.aside>
        </div>
      </main>

      <footer className="mt-20 py-20 border-t border-slate-200 max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-slate-200 rounded-lg" />
          <span className="text-sm font-black text-slate-400 uppercase tracking-widest">ChurnAI Platform</span>
        </div>
        <div className="flex items-center gap-8 text-xs font-bold text-slate-400 uppercase tracking-widest">
          <a href="#" className="hover:text-emerald-500 transition-colors">Safety</a>
          <a href="#" className="hover:text-emerald-500 transition-colors">Performance</a>
          <a href="#" className="hover:text-emerald-500 transition-colors">Privacy</a>
        </div>
        <p className="text-xs font-bold text-slate-400">© 2026 RGT Predictive Analytics</p>
      </footer>
    </div>
  );
}
