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
    <header className="h-14 border-b border-border bg-card/80 backdrop-blur-sm flex items-center justify-between px-6">
      <div>
        <h1 className="text-lg font-semibold">{title}</h1>
      </div>

      <div className="flex items-center gap-4">
        {/* Difficulty Mode Toggle */}
        <select
          value={difficultyMode}
          onChange={(e) => setDifficultyMode(e.target.value as DifficultyMode)}
          className="text-xs px-2 py-1 rounded-lg border border-border bg-background cursor-pointer"
        >
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="expert">Expert</option>
        </select>

        {/* Streak */}
        {streak > 0 && (
          <div className="flex items-center gap-1 text-xs text-orange-500">
            <Flame className="w-3.5 h-3.5" />
            <span className="font-bold">{streak}</span>
          </div>
        )}

        {/* Search */}
        <button className="p-2 rounded-lg hover:bg-muted transition-colors">
          <Search className="w-4 h-4 text-muted-foreground" />
        </button>

        {/* Notifications */}
        <button className="p-2 rounded-lg hover:bg-muted transition-colors relative">
          <Bell className="w-4 h-4 text-muted-foreground" />
          <span className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-red-500 rounded-full border-2 border-card" />
        </button>

        {/* XP Display */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gradient-to-r from-primary-500/10 to-accent-500/10 border border-primary-500/20">
          <span className="text-xs font-medium text-primary-500">LVL {level}</span>
          <div className="w-20 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary-500 to-accent-500 rounded-full transition-all"
              style={{ width: `${xpProgress}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground">{xp} XP</span>
        </div>

        {/* User Avatar */}
        <button className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
          <User className="w-4 h-4 text-white" />
        </button>
      </div>
    </header>
  );
}
