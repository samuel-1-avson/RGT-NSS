"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";
import { motion } from "framer-motion";
import type { ComponentType } from "react";
import {
  BookOpen,
  Brain,
  Sparkles,
  Layers,
  Zap,
  BarChart3,
  Home,
  Settings,
  Trophy,
  GraduationCap,
  FlaskConical,
  Microscope,
  Gauge,
  Search,
  Server,
  MessageSquare,
  Shield,
  ScrollText,
  Target,
} from "lucide-react";

type NavLink = {
  name: string;
  href: string;
  icon: ComponentType<{ className?: string }>;
};

type NavSection = {
  name: string;
  items: NavLink[];
};

const navigation: Array<NavLink | NavSection> = [
  { name: "Home", href: "/", icon: Home },
  { name: "Dashboard", href: "/dashboard", icon: BarChart3 },
  {
    name: "FOUNDATIONS",
    items: [
      { name: "Tokenization Lab", href: "/modules/tokenization", icon: BookOpen },
      { name: "Embedding Explorer", href: "/modules/embeddings", icon: Sparkles },
    ],
  },
  {
    name: "ARCHITECTURE",
    items: [
      { name: "Attention Visualizer", href: "/modules/attention", icon: Brain },
      { name: "Transformer Builder", href: "/modules/transformer", icon: Layers },
    ],
  },
  {
    name: "TRAINING",
    items: [
      { name: "Training Dashboard", href: "/modules/training", icon: Zap },
      { name: "Inference Playground", href: "/modules/inference", icon: FlaskConical },
    ],
  },
  {
    name: "ADVANCED",
    items: [
      { name: "RLHF Lab", href: "/modules/rlhf", icon: GraduationCap },
      { name: "LoRA Studio", href: "/modules/lora", icon: Microscope },
      { name: "Evaluation Suite", href: "/modules/evaluation", icon: Target },
    ],
  },
  {
    name: "FRONTIER",
    items: [
      { name: "Inference Optimization", href: "/modules/inference-opt", icon: Gauge },
      { name: "Interpretability", href: "/modules/interpretability", icon: Search },
      { name: "Distributed Training", href: "/modules/distributed", icon: Server },
      { name: "Prompt Engineering", href: "/modules/prompt-eng", icon: MessageSquare },
      { name: "AI Safety", href: "/modules/safety", icon: Shield },
      { name: "Long Context", href: "/modules/long-context", icon: ScrollText },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-border bg-card flex flex-col h-full">
      {/* Logo */}
      <div className="p-6 border-b border-border">
        <Link href="/" className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="font-bold text-sm">LLM Learning</div>
            <div className="text-xs text-muted-foreground">Platform v3.0</div>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-4 space-y-1 relative">
        {navigation.map((item, idx) => {
          if ("href" in item) {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={clsx(
                  "relative flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors z-10",
                  isActive
                    ? "text-primary-400 font-medium"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {isActive && (
                  <motion.div
                    layoutId="activeNav"
                    className="absolute inset-0 bg-primary-900/20 border border-primary-500/20 rounded-lg -z-10"
                    transition={{ type: "spring", stiffness: 350, damping: 30 }}
                  />
                )}
                <item.icon className="w-4 h-4" />
                {item.name}
              </Link>
            );
          }

          return (
            <div key={idx} className="pt-4">
              <div className="px-3 mb-2 text-[10px] font-bold text-muted-foreground uppercase tracking-widest opacity-80">
                {item.name}
              </div>
              {item.items.map((sub) => {
                const isActive = pathname === sub.href;
                return (
                  <Link
                    key={sub.href}
                    href={sub.href}
                    className={clsx(
                      "relative flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors z-10",
                      isActive
                        ? "text-primary-400 font-medium"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="activeNav"
                        className="absolute inset-0 bg-primary-900/20 border border-primary-500/20 rounded-lg -z-10"
                        transition={{ type: "spring", stiffness: 350, damping: 30 }}
                      />
                    )}
                    <sub.icon className="w-4 h-4" />
                    {sub.name}
                  </Link>
                );
              })}
            </div>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <Link
          href="/settings"
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
        >
          <Settings className="w-4 h-4" />
          Settings
        </Link>
        <Link
          href="/achievements"
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
        >
          <Trophy className="w-4 h-4" />
          Achievements
        </Link>
      </div>
    </aside>
  );
}
