"use client";

import { useState, useEffect } from "react";
import { Navbar } from "@/components/Navbar";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Settings as SettingsIcon, 
  Cpu, 
  ShieldCheck, 
  Activity, 
  Save, 
  RotateCcw,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Sliders,
  Terminal,
  Database,
  Info
} from "lucide-react";

export default function SettingsPage() {
  const [settings, setSettings] = useState<any>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [settingsRes, infoRes] = await Promise.all([
        fetch("http://localhost:8000/settings"),
        fetch("http://localhost:8000/model/info")
      ]);

      if (!settingsRes.ok || !infoRes.ok) throw new Error("Could not connect to API");

      const settingsData = await settingsRes.json();
      const infoData = await infoRes.json();

      setSettings(settingsData);
      setModelInfo(infoData);
    } catch (err: any) {
      setError("Failed to load configuration. Ensure the FastAPI backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateSetting = (name: string, value: any) => {
    setSettings((prev: any) => ({ ...prev, [name]: value }));
    setSuccess(false);
  };

  const saveSettings = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("http://localhost:8000/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction_threshold: settings.prediction_threshold,
          log_level: settings.log_level,
          max_batch_size: settings.max_batch_size
        })
      });

      if (!res.ok) throw new Error("Update failed");

      const updated = await res.json();
      setSettings(updated);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err: any) {
      setError("Failed to update settings.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#f8fafc] flex items-center justify-center">
        <Loader2 className="w-10 h-10 text-emerald-500 animate-spin" />
      </div>
    );
  }

  const inputClasses = "w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all";
  const labelClasses = "block text-[10px] font-black uppercase tracking-widest text-slate-400 mb-2 ml-1";

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 pt-40 pb-20 px-6">
      <Navbar />

      <main className="max-w-5xl mx-auto">
        <header className="mb-12">
          <div className="flex items-center gap-3 mb-6">
            <SettingsIcon className="w-8 h-8 text-emerald-500" />
            <span className="text-sm font-black text-slate-400 uppercase tracking-[0.4em]">System Control</span>
          </div>
          <h1 className="text-5xl font-black tracking-tighter text-slate-900 mb-6 underline decoration-emerald-500 decoration-8 underline-offset-8">
            Model Governance
          </h1>
          <p className="text-lg text-slate-500 font-medium leading-relaxed max-w-2xl">
            Configure churn prediction sensitivity, system logging behavior, and monitor model health from a centralized control plane.
          </p>
        </header>

        {error && (
          <div className="mb-8 p-6 bg-red-50 border border-red-100 rounded-3xl flex items-center gap-4 text-red-700 font-bold">
            <AlertCircle className="w-6 h-6 shrink-0" />
            <p>{error}</p>
            <button onClick={fetchData} className="ml-auto underline">Try Reconnect</button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Controls */}
          <section className="lg:col-span-7 space-y-8">
            <div className="glass-card bg-white p-10 border border-white/60">
              <h2 className="text-xl font-black mb-10 flex items-center gap-4">
                <Sliders className="w-6 h-6 text-emerald-500" />
                 Prediction Sensitivity
              </h2>

              <div className="space-y-8">
                <div>
                  <div className="flex justify-between items-end mb-4">
                    <label className={labelClasses}>Probability Threshold</label>
                    <span className="text-2xl font-black text-slate-900">{(settings.prediction_threshold * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="1" 
                    step="0.01" 
                    value={settings.prediction_threshold}
                    onChange={(e) => handleUpdateSetting('prediction_threshold', parseFloat(e.target.value))}
                    className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                  <div className="flex justify-between mt-4">
                    <p className="text-[10px] font-bold text-slate-400">HIGHER PRECISION</p>
                    <p className="text-[10px] font-bold text-slate-400">HIGHER RECALL</p>
                  </div>
                  <div className="mt-6 p-4 bg-emerald-50 rounded-2xl border border-emerald-100">
                    <p className="text-xs text-emerald-700 font-medium leading-relaxed">
                      <Info className="w-3 h-3 inline mr-2" />
                      Adjusting this shifts the decision boundary for "Churn" vs "Retained". Lower thresholds are more aggressive at flagging risks.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-6 border-t border-slate-50">
                  <div>
                    <label className={labelClasses}>System Log Level</label>
                    <select 
                      value={settings.log_level}
                      onChange={(e) => handleUpdateSetting('log_level', e.target.value)}
                      className={inputClasses}
                    >
                      <option value="DEBUG">DEBUG (Verbose)</option>
                      <option value="INFO">INFO (Standard)</option>
                      <option value="WARNING">WARNING (Critical)</option>
                    </select>
                  </div>
                  <div>
                    <label className={labelClasses}>Max Batch Size</label>
                    <input 
                      type="number" 
                      value={settings.max_batch_size}
                      onChange={(e) => handleUpdateSetting('max_batch_size', parseInt(e.target.value))}
                      className={inputClasses}
                    />
                  </div>
                </div>
              </div>

              <div className="mt-12 flex gap-4">
                <button 
                  onClick={saveSettings}
                  disabled={saving}
                  className="flex-1 py-5 bg-slate-900 text-white rounded-[1.5rem] font-black text-sm uppercase tracking-widest hover:bg-slate-800 transition-all flex items-center justify-center gap-3 active:scale-95 disabled:opacity-50"
                >
                  {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                  Save Configuration
                </button>
                <button 
                  onClick={fetchData}
                  className="px-8 py-5 bg-white border border-slate-200 text-slate-400 rounded-[1.5rem] font-black text-sm uppercase tracking-widest hover:bg-slate-50 transition-all flex items-center gap-3 active:scale-95"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>

              <AnimatePresence>
                {success && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mt-6 p-4 bg-emerald-500 text-white rounded-2xl flex items-center gap-3 font-bold text-sm"
                  >
                    <CheckCircle2 className="w-5 h-5 text-slate-900" />
                    Configuration synchronized with API successfully.
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </section>

          {/* Monitoring */}
          <aside className="lg:col-span-5 space-y-8">
            <div className="glass-card bg-slate-900 p-10 text-white">
              <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
                <Cpu className="w-5 h-5 text-emerald-400" />
                Execution Context
              </h2>
              
              <div className="space-y-6">
                <div className="flex justify-between items-center py-4 border-b border-white/5">
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Model Loaded</span>
                  <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase ${modelInfo?.model_loaded ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                    {modelInfo?.model_loaded ? 'ACTIVE' : 'FAILED'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-4 border-b border-white/5">
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">API Version</span>
                  <span className="text-sm font-bold text-slate-300">{settings.app_version}</span>
                </div>
                <div className="flex justify-between items-center py-4">
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Model Version</span>
                  <span className="text-sm font-bold text-slate-300">{settings.model_version}</span>
                </div>
              </div>
            </div>

            <div className="glass-card bg-white p-10">
               <h2 className="text-xl font-black mb-8 flex items-center gap-3">
                <ShieldCheck className="w-5 h-5 text-emerald-500" />
                 Safety Audit
              </h2>
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  <p className="text-xs font-bold text-slate-600">Cross-Origin Policy: <span className="text-slate-900">Liberal (*)</span></p>
                </div>
                <div className="flex items-center gap-4">
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  <p className="text-xs font-bold text-slate-600">Auth Status: <span className="text-slate-900">Developer Mode</span></p>
                </div>
                <div className="flex items-center gap-4">
                   <div className="w-2 h-2 rounded-full bg-amber-500" />
                  <p className="text-xs font-bold text-slate-600">Metrics: <span className="text-slate-900">Internal Only</span></p>
                </div>
              </div>
            </div>

             <div className="glass-card bg-emerald-50 p-8 text-emerald-900 flex flex-col gap-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-white rounded-2xl flex items-center justify-center shadow-sm">
                  <Database className="w-6 h-6 text-emerald-500" />
                </div>
                <div>
                   <h4 className="text-sm font-black uppercase tracking-tight">Data Health</h4>
                   <p className="text-[10px] font-bold text-emerald-700/70">UCI Repository v4.1 Connected</p>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
