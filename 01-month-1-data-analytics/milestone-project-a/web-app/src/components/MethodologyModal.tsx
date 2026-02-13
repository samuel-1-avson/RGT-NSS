"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, CheckCircle2, Wrench, Settings, BookOpen } from "lucide-react";

interface MethodologyStep {
  id: number;
  title: string;
  method: string;
  tools: string[];
  steps: string[];
  reasoning: string;
}

interface MethodologyModalProps {
  isOpen: boolean;
  onClose: () => void;
  step: MethodologyStep | null;
}

export function MethodologyModal({ isOpen, onClose, step }: MethodologyModalProps) {
  if (!step) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-900/40 backdrop-blur-md z-[60] flex items-center justify-center p-6"
          />

          {/* Modal Container */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl bg-white rounded-[2.5rem] shadow-2xl z-[70] overflow-hidden border border-slate-100"
          >
            {/* Header */}
            <div className="relative p-8 md:p-10 border-b border-slate-50">
              <button 
                onClick={onClose}
                className="absolute top-6 right-6 p-2 rounded-full hover:bg-slate-100 transition-colors"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
              
              <div className="flex items-center gap-4 mb-4">
                <div className="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center text-white font-bold text-xl">
                  {step.id}
                </div>
                <h2 className="text-3xl font-black text-slate-900 tracking-tight">{step.title}</h2>
              </div>
              <p className="text-slate-500 font-medium leading-relaxed">
                {step.method}
              </p>
            </div>

            {/* Content */}
            <div className="p-8 md:p-10 space-y-8 max-h-[60vh] overflow-y-auto custom-scrollbar">
              {/* Reasoning Section */}
              <section>
                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  Clinical Reasoning
                </h3>
                <div className="p-6 bg-blue-50/50 rounded-2xl border border-blue-100 text-blue-900 font-semibold leading-relaxed">
                  {step.reasoning}
                </div>
              </section>

              {/* Tools and Steps Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <section>
                  <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Wrench className="w-4 h-4" />
                    Project Stack
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {step.tools.map((tool, idx) => (
                      <span key={idx} className="px-3 py-1.5 bg-slate-100 rounded-lg text-xs font-bold text-slate-600">
                        {tool}
                      </span>
                    ))}
                  </div>
                </section>

                <section>
                  <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4" />
                    Key Deliverables
                  </h3>
                  <ul className="space-y-3">
                    {step.steps.map((item, idx) => (
                      <li key={idx} className="flex items-start gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-600 mt-1.5 shrink-0" />
                        <span className="text-sm text-slate-600 font-medium leading-tight">{item}</span>
                      </li>
                    ))}
                  </ul>
                </section>
              </div>
            </div>

            {/* Footer Action */}
            <div className="p-8 bg-slate-50 border-t border-slate-100 flex justify-end">
              <button 
                onClick={onClose}
                className="px-8 py-3 bg-slate-900 text-white rounded-2xl text-sm font-bold hover:bg-slate-800 transition-all shadow-lg active:scale-95"
              >
                Close Details
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
