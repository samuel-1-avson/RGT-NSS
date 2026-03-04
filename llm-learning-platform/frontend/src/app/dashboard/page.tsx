"use client";

import { useUserStore } from "@/lib/stores";
import { formatNumber } from "@/lib/utils";
import Link from "next/link";
import {
  BookOpen,
  Brain,
  Sparkles,
  Layers,
  Zap,
  BarChart3,
  Trophy,
  Target,
  Clock,
  TrendingUp,
} from "lucide-react";

const MODULES_LIST = [
  { id: "tokenization-lab", title: "Tokenization Lab", href: "/modules/tokenization", xp: 100 },
  { id: "embedding-explorer", title: "Embedding Explorer", href: "/modules/embeddings", xp: 120 },
  { id: "attention-visualizer", title: "Attention Visualizer", href: "/modules/attention", xp: 200 },
  { id: "transformer-builder", title: "Transformer Builder", href: "/modules/transformer", xp: 250 },
  { id: "training-dashboard", title: "Training Dashboard", href: "/modules/training", xp: 300 },
  { id: "inference-playground", title: "Inference Playground", href: "/modules/inference", xp: 200 },
  { id: "rlhf-lab", title: "RLHF & Alignment Lab", href: "/modules/rlhf", xp: 400 },
  { id: "lora-studio", title: "LoRA & QLoRA Studio", href: "/modules/lora", xp: 350 },
  { id: "evaluation-suite", title: "Evaluation Suite", href: "/modules/evaluation", xp: 300 },
  { id: "inference-opt", title: "Inference Optimization", href: "/modules/inference-opt", xp: 350 },
  { id: "interpretability", title: "Interpretability Lab", href: "/modules/interpretability", xp: 400 },
  { id: "distributed", title: "Distributed Training", href: "/modules/distributed", xp: 400 },
  { id: "prompt-eng", title: "Prompt Engineering", href: "/modules/prompt-eng", xp: 250 },
  { id: "safety", title: "AI Safety Center", href: "/modules/safety", xp: 350 },
  { id: "long-context", title: "Long Context Explorer", href: "/modules/long-context", xp: 350 },
];

export default function DashboardPage() {
  const user = useUserStore();

  const completedCount = Object.values(user.moduleProgress).filter(
    (s) => s === "completed"
  ).length;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
        <p className="text-muted-foreground">
          Track your progress across all learning modules.
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-6">
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-primary-500" />
            </div>
            <div>
              <div className="text-2xl font-bold">{user.xp}</div>
              <div className="text-xs text-muted-foreground">Total XP</div>
            </div>
          </div>
          <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary-500 to-accent-500 rounded-full transition-all"
              style={{ width: `${Math.min((user.xp % 100) / 100 * 100, 100)}%` }}
            />
          </div>
          <div className="text-xs text-muted-foreground mt-1">Level {user.level}</div>
        </div>

        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
              <Target className="w-5 h-5 text-emerald-500" />
            </div>
            <div>
              <div className="text-2xl font-bold">{completedCount}/{MODULES_LIST.length}</div>
              <div className="text-xs text-muted-foreground">Modules Complete</div>
            </div>
          </div>
        </div>

        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
              <Trophy className="w-5 h-5 text-amber-500" />
            </div>
            <div>
              <div className="text-2xl font-bold">{user.achievements.length}</div>
              <div className="text-xs text-muted-foreground">Achievements</div>
            </div>
          </div>
        </div>

        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-rose-100 dark:bg-rose-900/30 flex items-center justify-center">
              <Clock className="w-5 h-5 text-rose-500" />
            </div>
            <div>
              <div className="text-2xl font-bold">{user.streak}</div>
              <div className="text-xs text-muted-foreground">Day Streak</div>
            </div>
          </div>
        </div>
      </div>

      {/* Module Progress */}
      <div className="glass rounded-2xl p-6">
        <h2 className="text-xl font-semibold mb-4">Learning Progress</h2>
        <div className="space-y-3">
          {MODULES_LIST.map((mod) => {
            const status = user.moduleProgress[mod.id] || "not_started";
            return (
              <Link
                key={mod.id}
                href={mod.href}
                className="flex items-center justify-between p-4 rounded-xl border border-border hover:border-primary-300 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-3 h-3 rounded-full ${
                      status === "completed"
                        ? "bg-emerald-500"
                        : status === "in_progress"
                        ? "bg-amber-500"
                        : "bg-gray-300 dark:bg-gray-600"
                    }`}
                  />
                  <span className="font-medium text-sm">{mod.title}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-xs text-muted-foreground">{mod.xp} XP</span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      status === "completed"
                        ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600"
                        : status === "in_progress"
                        ? "bg-amber-100 dark:bg-amber-900/30 text-amber-600"
                        : "bg-gray-100 dark:bg-gray-800 text-gray-500"
                    }`}
                  >
                    {status === "completed"
                      ? "Complete"
                      : status === "in_progress"
                      ? "In Progress"
                      : "Not Started"}
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
