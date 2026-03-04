"use client";

import { usePathname } from "next/navigation";
import { Search, Bell, User, Flame } from "lucide-react";
import { useUserStore } from "@/lib/stores";
import type { DifficultyMode } from "@/lib/stores";

const PAGE_TITLES: Record<string, string> = {
  "/": "Home",
  "/dashboard": "Dashboard",
  "/modules/tokenization": "Tokenization Lab",
  "/modules/embeddings": "Embedding Explorer",
  "/modules/attention": "Attention Visualizer",
  "/modules/transformer": "Transformer Builder",
  "/modules/training": "Training Dashboard",
  "/modules/inference": "Inference Playground",
  "/modules/rlhf": "RLHF & Alignment Lab",
  "/modules/lora": "LoRA & QLoRA Studio",
  "/modules/evaluation": "Evaluation Suite",
  "/modules/inference-opt": "Inference Optimization",
  "/modules/interpretability": "Interpretability Lab",
  "/modules/distributed": "Distributed Training",
  "/modules/prompt-eng": "Prompt Engineering",
  "/modules/safety": "AI Safety Center",
  "/modules/long-context": "Long Context Explorer",
  "/settings": "Settings",
  "/achievements": "Achievements",
};

export function Header() {
  const pathname = usePathname();
  const title = PAGE_TITLES[pathname] || "LLM Learning Platform";
  const { xp, level, streak, difficultyMode, setDifficultyMode, updateStreak } = useUserStore();

  const xpForCurrentLevel = level * 100;
  const xpInLevel = xp % xpForCurrentLevel || 0;
  const xpProgress = Math.min((xpInLevel / xpForCurrentLevel) * 100, 100);

  return (
    <header className="h-14 border-b border-white/5 bg-background/40 backdrop-blur-xl flex items-center justify-between px-6 z-40">
      <div>
        <h1 className="text-[15px] font-medium text-foreground/90 tracking-wide">{title}</h1>
      </div>

      <div className="flex items-center gap-5">
        {/* Difficulty Mode Toggle */}
        <select
          value={difficultyMode}
          onChange={(e) => setDifficultyMode(e.target.value as DifficultyMode)}
          className="text-[11px] px-2.5 py-1.5 rounded-md border border-white/10 bg-black/20 text-muted-foreground outline-none cursor-pointer focus:border-primary/50 transition-colors uppercase tracking-wider font-medium"
        >
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="expert">Expert</option>
        </select>

        {/* Streak */}
        {streak > 0 && (
          <div className="flex items-center gap-1.5 text-xs text-orange-400/90 font-mono">
            <Flame className="w-3.5 h-3.5" />
            <span>{streak}</span>
          </div>
        )}

        <div className="h-4 w-px bg-white/10 mx-1" />

        {/* Search */}
        <button className="text-muted-foreground hover:text-foreground transition-colors">
          <Search className="w-4 h-4" />
        </button>

        {/* Notifications */}
        <button className="text-muted-foreground hover:text-foreground transition-colors relative">
          <Bell className="w-4 h-4" />
          <span className="absolute -top-1 -right-1 w-2 h-2 bg-primary rounded-full" />
        </button>

        <div className="h-4 w-px bg-white/10 mx-1" />

        {/* XP Display */}
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-bold text-primary tracking-widest uppercase">Lvl {level}</span>
          <div className="w-16 h-1 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-1000 ease-out"
              style={{ width: `${xpProgress}%` }}
            />
          </div>
          <span className="text-[10px] text-muted-foreground font-mono">{xp} XP</span>
        </div>

        {/* User Avatar */}
        <button className="w-7 h-7 ml-2 rounded-full bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors">
          <User className="w-3.5 h-3.5 text-foreground/70" />
        </button>
      </div>
    </header>
  );
}
