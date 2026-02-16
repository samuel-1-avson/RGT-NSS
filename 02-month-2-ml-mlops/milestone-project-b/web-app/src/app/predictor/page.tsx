"use client";

import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { motion, AnimatePresence } from "framer-motion";
import { 
  BrainCircuit, 
  Send, 
  RefreshCw, 
  ShieldCheck, 
  ShieldAlert, 
  UserPlus, 
  CreditCard, 
  Globe,
  Loader2,
  AlertCircle,
  CheckCircle2
} from "lucide-react";

export default function PredictorPage() {
  const [formData, setFormData] = useState({
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 1,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 70.0,
    TotalCharges: 70.0
  });

  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === "MonthlyCharges" || name === "TotalCharges" || name === "tenure" || name === "SeniorCitizen" 
        ? Number(value) 
        : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          customer_id: "WEB_" + Math.random().toString(36).substr(2, 9),
          data: formData
        })
      });

      if (!response.ok) throw new Error("Backend service is unavailable");
      
      const data = await response.json();
      setPrediction(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong. Please ensure the FastAPI backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const inputClasses = "w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-sm font-bold text-slate-700 focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all outline-none appearance-none";
  const labelClasses = "block text-[10px] font-black uppercase tracking-widest text-slate-400 mb-2 ml-1";

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 pt-40 pb-20 px-6">
      <Navbar />

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-12">
        {/* Form Column */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-7"
        >
          <div className="mb-10">
            <h1 className="text-4xl font-black tracking-tight text-slate-900 mb-4 flex items-center gap-4">
              <div className="w-12 h-12 bg-emerald-500 rounded-2xl flex items-center justify-center shadow-lg shadow-emerald-500/20">
                <BrainCircuit className="w-7 h-7 text-slate-900" />
              </div>
              Predictor Input
            </h1>
            <p className="text-slate-500 font-medium">Enter customer demographics and account details for real-time risk assessment.</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-10">
            {/* Demographics */}
            <section className="glass-card bg-white p-8 border border-white/60">
              <h2 className="text-sm font-black text-slate-900 uppercase tracking-widest mb-8 flex items-center gap-3">
                <UserPlus className="w-4 h-4 text-emerald-500" />
                Demographics
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div>
                  <label className={labelClasses}>Gender</label>
                  <select name="gender" value={formData.gender} onChange={handleInputChange} className={inputClasses}>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Senior Citizen</label>
                  <select name="SeniorCitizen" value={formData.SeniorCitizen} onChange={handleInputChange} className={inputClasses}>
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Partner</label>
                  <select name="Partner" value={formData.Partner} onChange={handleInputChange} className={inputClasses}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Dependents</label>
                  <select name="Dependents" value={formData.Dependents} onChange={handleInputChange} className={inputClasses}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>
              </div>
            </section>

            {/* Account & Service */}
            <section className="glass-card bg-white p-8 border border-white/60">
              <h2 className="text-sm font-black text-slate-900 uppercase tracking-widest mb-8 flex items-center gap-3">
                <CreditCard className="w-4 h-4 text-emerald-500" />
                Account & Billing
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div>
                  <label className={labelClasses}>Tenure (Months)</label>
                  <input type="number" name="tenure" value={formData.tenure} onChange={handleInputChange} className={inputClasses} min="0" max="72" />
                </div>
                <div>
                  <label className={labelClasses}>Contract</label>
                  <select name="Contract" value={formData.Contract} onChange={handleInputChange} className={inputClasses}>
                    <option value="Month-to-month">Month-to-month</option>
                    <option value="One year">One year</option>
                    <option value="Two year">Two year</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Payment Method</label>
                  <select name="PaymentMethod" value={formData.PaymentMethod} onChange={handleInputChange} className={inputClasses}>
                    <option value="Electronic check">Electronic check</option>
                    <option value="Mailed check">Mailed check</option>
                    <option value="Bank transfer (automatic)">Bank transfer</option>
                    <option value="Credit card (automatic)">Credit card</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Monthly Charges</label>
                  <input type="number" name="MonthlyCharges" value={formData.MonthlyCharges} onChange={handleInputChange} className={inputClasses} step="0.01" />
                </div>
                <div>
                  <label className={labelClasses}>Total Charges</label>
                  <input type="number" name="TotalCharges" value={formData.TotalCharges} onChange={handleInputChange} className={inputClasses} step="0.01" />
                </div>
                <div>
                  <label className={labelClasses}>Paperless Billing</label>
                  <select name="PaperlessBilling" value={formData.PaperlessBilling} onChange={handleInputChange} className={inputClasses}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>
              </div>
            </section>

            {/* Connectivity */}
            <section className="glass-card bg-white p-8 border border-white/60">
              <h2 className="text-sm font-black text-slate-900 uppercase tracking-widest mb-8 flex items-center gap-3">
                <Globe className="w-4 h-4 text-emerald-500" />
                Connectivity & Services
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div>
                  <label className={labelClasses}>Internet Service</label>
                  <select name="InternetService" value={formData.InternetService} onChange={handleInputChange} className={inputClasses}>
                    <option value="DSL">DSL</option>
                    <option value="Fiber optic">Fiber optic</option>
                    <option value="No">No Internet</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Phone Service</label>
                  <select name="PhoneService" value={formData.PhoneService} onChange={handleInputChange} className={inputClasses}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Multiple Lines</label>
                  <select name="MultipleLines" value={formData.MultipleLines} onChange={handleInputChange} className={inputClasses}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                    <option value="No phone service">No phone service</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Online Security</label>
                  <select name="OnlineSecurity" value={formData.OnlineSecurity} onChange={handleInputChange} className={inputClasses}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                    <option value="No internet service">No internet service</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Tech Support</label>
                  <select name="TechSupport" value={formData.TechSupport} onChange={handleInputChange} className={inputClasses}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                    <option value="No internet service">No internet service</option>
                  </select>
                </div>
                <div>
                  <label className={labelClasses}>Streaming Movies</label>
                  <select name="StreamingMovies" value={formData.StreamingMovies} onChange={handleInputChange} className={inputClasses}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                    <option value="No internet service">No internet service</option>
                  </select>
                </div>
              </div>
            </section>

            <button 
              type="submit" 
              disabled={loading}
              className="w-full py-6 bg-slate-900 text-white rounded-[2rem] font-black text-xl hover:bg-slate-800 transition-all shadow-2xl shadow-slate-900/10 flex items-center justify-center gap-4 disabled:opacity-50 active:scale-95"
            >
              {loading ? (
                <>
                  <Loader2 className="w-6 h-6 animate-spin" />
                  ANALYZING DATA...
                </>
              ) : (
                <>
                  <Send className="w-6 h-6" />
                  GET PREDICTION
                </>
              )}
            </button>
          </form>
        </motion.div>

        {/* Results Column */}
        <div className="lg:col-span-5">
          <div className="sticky top-40 space-y-8">
            <AnimatePresence mode="wait">
              {!prediction && !loading && !error && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="glass-card bg-white p-12 text-center border-dashed border-2 border-slate-200"
                >
                  <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mx-auto mb-6">
                    <BrainCircuit className="w-10 h-10 text-slate-300" />
                  </div>
                  <h3 className="text-xl font-black text-slate-800 mb-2">Ready for Analysis</h3>
                  <p className="text-sm text-slate-400 font-medium">Fill out the form and submit to receive a real-time churn prediction.</p>
                </motion.div>
              )}

              {error && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="glass-card bg-red-50 p-10 border-red-100 text-center"
                >
                  <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                  <h3 className="text-lg font-black text-red-900 mb-2">Connection Error</h3>
                  <p className="text-sm text-red-700 font-bold leading-relaxed">{error}</p>
                </motion.div>
              )}

              {prediction && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="space-y-8"
                >
                  {/* Score Card */}
                  <div className={`glass-card p-10 text-center text-white relative overflow-hidden ${prediction.churn_prediction ? 'bg-red-500 shadow-red-500/20' : 'bg-emerald-500 shadow-emerald-500/20'}`}>
                    <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16 blur-2xl" />
                    <p className="text-[10px] font-black uppercase tracking-[0.3em] mb-4 text-white/70">Prediction Result</p>
                    <h2 className="text-5xl font-black tracking-tighter mb-4">
                      {prediction.churn_prediction ? 'CHURN RISK' : 'RETAINED'}
                    </h2>
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-black/10 rounded-full text-xs font-black">
                      Probability: {(prediction.churn_probability * 100).toFixed(1)}%
                    </div>
                  </div >

                  {/* Confidence Levels */}
                  <div className="glass-card bg-white p-8">
                    <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-6 flex items-center gap-2">
                       {prediction.confidence === 'high' ? <ShieldCheck className="w-4 h-4 text-emerald-500" /> : <Loader2 className="w-4 h-4 text-amber-500" />}
                       Predictive Confidence: {prediction.confidence.toUpperCase()}
                    </h4>
                    <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${prediction.confidence === 'high' ? 95 : 60}%` }}
                        className={`h-full ${prediction.confidence === 'high' ? 'bg-emerald-500' : 'bg-amber-500'}`}
                      />
                    </div>
                  </div>

                  {/* Recommended Action */}
                  <div className="glass-card bg-slate-900 p-8 text-white">
                    <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-6 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-blue-400" />
                      Protocol Recommendation
                    </h4>
                    <p className="text-sm font-bold text-slate-300 leading-relaxed italic">
                      {prediction.churn_prediction 
                        ? "Immediate retention campaign suggested: Offer loyalty incentive and contract upgrade path."
                        : "Maintain standard engagement. Strategic upsell opportunity for Fiber Optic services."}
                    </p>
                  </div>

                  <button 
                    onClick={() => setPrediction(null)}
                    className="w-full py-4 text-slate-400 font-black text-[10px] uppercase tracking-[0.5em] hover:text-slate-900 transition-colors flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Reset Analysis
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
