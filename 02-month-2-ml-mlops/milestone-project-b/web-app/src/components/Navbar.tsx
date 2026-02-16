"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Shield, LayoutDashboard, BrainCircuit, Users, Settings } from "lucide-react";
import { motion } from "framer-motion";

export function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { name: "Dashboard", href: "/", icon: <LayoutDashboard className="w-4 h-4" /> },
    { name: "Predictor", href: "/predictor", icon: <BrainCircuit className="w-4 h-4" /> },
    { name: "Model Cards", href: "/model-cards", icon: <Shield className="w-4 h-4" /> },
    { name: "Settings", href: "/settings", icon: <Settings className="w-4 h-4" /> },
  ];

  return (
    <nav className="fixed top-6 left-1/2 -translate-x-1/2 z-50 w-[calc(100%-3rem)] max-w-5xl glass-card bg-slate-900/90 border-slate-700 shadow-2xl px-8 h-20 flex justify-between items-center transition-all">
      <div className="flex items-center gap-10">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center group-hover:rotate-12 transition-transform shadow-lg shadow-emerald-500/20">
            <Users className="w-6 h-6 text-slate-900" />
          </div>
          <div className="hidden sm:block">
            <h1 className="text-white font-black tracking-tight text-lg leading-none">CHURN<span className="text-emerald-500">AI</span></h1>
            <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mt-1">Predictive Ops</p>
          </div>
        </Link>

        <div className="hidden md:flex items-center gap-8">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 text-sm font-bold transition-all px-4 py-2 rounded-xl h-10 ${
                  isActive 
                    ? "bg-white/10 text-emerald-400" 
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
              >
                {item.icon}
                {item.name}
              </Link>
            );
          })}
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button className="hidden sm:flex items-center gap-2 px-6 py-2.5 bg-emerald-500 text-slate-900 rounded-xl text-xs font-black uppercase hover:bg-emerald-400 transition-all shadow-lg shadow-emerald-500/10 active:scale-95">
          New Analysis
        </button>
      </div>
    </nav>
  );
}
