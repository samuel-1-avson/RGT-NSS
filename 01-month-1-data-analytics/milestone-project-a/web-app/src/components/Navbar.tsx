"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity } from "lucide-react";
import { motion } from "framer-motion";

export function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { name: "Dashboard", href: "/" },
    { name: "Reports", href: "/reports" },
    { name: "Insights", href: "/insights" },
  ];

  return (
    <nav className="sticky top-0 z-50 glass-card bg-white/70 border-b border-slate-200/50 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-6 h-16 flex justify-between items-center">
        <Link href="/" className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <span className="text-lg font-bold tracking-tight text-slate-800">
            RGT <span className="text-blue-600 uppercase text-xs tracking-widest ml-1 font-extrabold">Analytics</span>
          </span>
        </Link>
        <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-500">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`relative py-5 transition-all hover:text-slate-800 ${
                  isActive ? "text-blue-600" : ""
                }`}
              >
                {item.name}
                {isActive && (
                  <motion.div
                    layoutId="nav-active"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600"
                  />
                )}
              </Link>
            );
          })}
        </div>
        <button className="bg-slate-900 text-white px-4 py-2 rounded-full text-xs font-semibold hover:bg-slate-800 transition-all shadow-sm">
          Contact Support
        </button>
      </div>
    </nav>
  );
}
