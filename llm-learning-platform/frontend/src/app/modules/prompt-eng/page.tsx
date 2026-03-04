"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { promptEngApi } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function PromptEngPage() {
  const [activeTab, setActiveTab] = useState<"templates" | "analyze" | "compare">("templates");
  const [selectedTemplate, setSelectedTemplate] = useState("zero_shot");
  const [variables, setVariables] = useState<Record<string, string>>({ question: "What is the capital of France?" });
  const [analyzeText, setAnalyzeText] = useState("Let's think step by step. What is the sum of 2+2?\n\nExample: 1+1 = 2\n\nNow answer the question.");
  const [compareTexts, setCompareTexts] = useState([
    "What is 2+2?",
    "Let's think step by step. What is 2+2?",
    "You are a math expert. Calculate: 2+2. Format: JSON {answer: number}",
  ]);

  const techniques = useQuery<any>({ queryKey: ["prompt-techniques"], queryFn: () => promptEngApi.techniques() });
  const templates = useQuery<any>({ queryKey: ["prompt-templates"], queryFn: () => promptEngApi.templates() });

  const renderMutation = useMutation<any>({ mutationFn: () => promptEngApi.render({ template_name: selectedTemplate, variables }) });
  const analyzeMutation = useMutation<any>({ mutationFn: () => promptEngApi.analyze(analyzeText) });
  const compareMutation = useMutation<any>({ mutationFn: () => promptEngApi.compare(compareTexts.filter(Boolean)) });

  const toTechniqueList = (data: any): string[] => {
    if (Array.isArray(data?.detected_techniques)) return data.detected_techniques;
    const flags = data?.techniques_detected;
    if (flags && typeof flags === "object") {
      return Object.entries(flags).filter(([, enabled]) => Boolean(enabled)).map(([name]) => name.replace(/_/g, " "));
    }
    return [];
  };

  const toComparisonItems = (data: any): any[] => {
    if (Array.isArray(data?.prompts)) return data.prompts;
    if (Array.isArray(data?.analyses)) return data.analyses;
    return [];
  };

  const tabs = [
    { id: "templates" as const, label: "Templates", icon: "📝" },
    { id: "analyze" as const, label: "Prompt Analyzer", icon: "🔍" },
    { id: "compare" as const, label: "Compare Prompts", icon: "⚔️" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold mb-2 tracking-tight text-foreground/90">Prompt Engineering Lab</h1>
        <p className="text-muted-foreground max-w-3xl">
          Master prompt techniques — use templates, analyze prompt quality, and compare prompt variants for optimal model interaction.
        </p>
      </div>

      <div className="flex gap-2 border-b border-white/5 pb-0 overflow-x-auto custom-scrollbar">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "relative px-5 py-3 rounded-t-xl text-sm font-medium transition-all outline-none whitespace-nowrap",
              activeTab === tab.id
                ? "text-primary z-10"
                : "text-muted-foreground hover:text-foreground hover:bg-white/5"
            )}
          >
            {activeTab === tab.id && (
              <motion.div
                layoutId="activeTabPrompt"
                className="absolute inset-0 bg-primary/10 border-b-2 border-primary"
                transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
              />
            )}
            <span className="relative z-10 flex items-center gap-2">
              <span className="opacity-80">{tab.icon}</span> {tab.label}
            </span>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeTab === "templates" && (
          <motion.div
            key="templates"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">Prompt Templates</h2>
                <p className="text-sm text-muted-foreground mt-1">Select a template and inject variables to render production-ready prompts.</p>
              </div>
              <div>
                <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block mb-2">Select Template</label>
                <select value={selectedTemplate} onChange={(e) => setSelectedTemplate(e.target.value)}
                  className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-2.5 text-sm text-foreground/90 focus:border-primary/50 outline-none transition-all">
                  {(templates.data as any)?.templates?.map((t: any) => (
                    <option key={t.name} value={t.name}>{t.name} — {t.description}</option>
                  )) || <option>Loading...</option>}
                </select>
              </div>
              <div>
                <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block mb-2">Variables</label>
                {Object.entries(variables).map(([k, v]) => (
                  <div key={k} className="flex gap-2 mb-2">
                    <input value={k} disabled className="w-32 rounded-lg border border-white/10 bg-black/60 px-3 py-1.5 text-xs font-mono text-muted-foreground" />
                    <input value={v} onChange={(e) => setVariables((prev) => ({ ...prev, [k]: e.target.value }))}
                      className="flex-1 rounded-lg border border-white/10 bg-black/40 px-3 py-1.5 text-xs text-foreground/90 focus:border-primary/50 outline-none transition-all" />
                  </div>
                ))}
                <button onClick={() => setVariables((prev) => ({ ...prev, [`var_${Object.keys(prev).length}`]: "" }))}
                  className="text-xs text-primary hover:text-primary/80 transition-colors mt-1">+ Add variable</button>
              </div>
              <button onClick={() => renderMutation.mutate()} disabled={renderMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {renderMutation.isPending ? "Rendering..." : "Render Template"}
              </button>
            </div>
            <div className="glass rounded-2xl p-6 flex flex-col">
              <h2 className="text-lg font-semibold text-foreground/90 mb-4 border-b border-white/10 pb-4">Rendered Prompt</h2>
              {renderMutation.data ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4 flex-1">
                  <pre className="bg-black/40 border border-white/5 rounded-xl p-4 text-sm font-mono whitespace-pre-wrap overflow-auto max-h-80 text-foreground/90 custom-scrollbar">
                    {(renderMutation.data as any).rendered}
                  </pre>
                  {(renderMutation.data as any).metadata && (
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries((renderMutation.data as any).metadata).map(([k, v]) => (
                        <div key={k} className="bg-black/40 border border-white/5 rounded-lg p-2">
                          <div className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">{k}</div>
                          <div className="text-xs font-mono text-foreground/80">{String(v)}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="flex-1 min-h-[250px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                  Select a template and render it
                </div>
              )}
            </div>
          </motion.div>
        )}

        {activeTab === "analyze" && (
          <motion.div
            key="analyze"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">Prompt Analyzer</h2>
                <p className="text-sm text-muted-foreground mt-1">Analyze a prompt to detect techniques, measure complexity, and get improvement suggestions.</p>
              </div>
              <textarea value={analyzeText} onChange={(e) => setAnalyzeText(e.target.value)} rows={6}
                className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm font-mono resize-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all text-foreground/90" />
              <button onClick={() => analyzeMutation.mutate()} disabled={analyzeMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {analyzeMutation.isPending ? "Analyzing..." : "Analyze Prompt"}
              </button>
            </div>
            <div className="glass rounded-2xl p-6 flex flex-col">
              <h2 className="text-lg font-semibold text-foreground/90 mb-4 border-b border-white/10 pb-4">Analysis Results</h2>
              {analyzeMutation.data ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5 flex-1">
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    {["word_count", "character_count", "line_count", "estimated_tokens", "num_examples"].map((k, i) => (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.05 }}
                        key={k}
                        className="bg-black/40 border border-white/5 rounded-xl p-3 flex flex-col justify-center text-center"
                      >
                        <div className="font-mono text-lg font-bold text-accent mb-0.5">{String((analyzeMutation.data as any)[k])}</div>
                        <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground/80">{k.replace(/_/g, " ")}</div>
                      </motion.div>
                    ))}
                  </div>
                  {toTechniqueList(analyzeMutation.data).length > 0 && (
                    <div>
                      <div className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Detected Techniques</div>
                      <div className="flex gap-2 flex-wrap">
                        {toTechniqueList(analyzeMutation.data).map((t: string) => (
                          <span key={t} className="px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {(analyzeMutation.data as any).suggestions?.length > 0 && (
                    <div>
                      <div className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Suggestions</div>
                      <ul className="space-y-1.5">
                        {(analyzeMutation.data as any).suggestions.map((s: string, i: number) => (
                          <li key={i} className="text-xs text-muted-foreground flex gap-2 items-start"><span className="text-primary mt-0.5">→</span> {s}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="flex-1 min-h-[250px] flex items-center justify-center text-center text-muted-foreground text-sm border border-dashed border-white/10 rounded-xl bg-black/20">
                  Enter a prompt and analyze it
                </div>
              )}
            </div>
          </motion.div>
        )}

        {activeTab === "compare" && (
          <motion.div
            key="compare"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            <div className="glass rounded-2xl p-6 space-y-6">
              <div className="pb-4 border-b border-white/10">
                <h2 className="text-lg font-semibold text-foreground/90">Compare Prompt Variants</h2>
                <p className="text-sm text-muted-foreground mt-1">Enter multiple prompt variants and compare their structural properties and detected techniques.</p>
              </div>
              {compareTexts.map((text, i) => (
                <div key={i}>
                  <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest block mb-2">Variant {i + 1}</label>
                  <textarea value={text} onChange={(e) => {
                    const next = [...compareTexts]; next[i] = e.target.value; setCompareTexts(next);
                  }} rows={2} className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm font-mono resize-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 outline-none transition-all text-foreground/90" />
                </div>
              ))}
              <div className="flex gap-3">
                <button onClick={() => setCompareTexts([...compareTexts, ""])}
                  className="text-xs text-primary hover:text-primary/80 transition-colors">+ Add variant</button>
                {compareTexts.length > 2 && (
                  <button onClick={() => setCompareTexts(compareTexts.slice(0, -1))}
                    className="text-xs text-pink-400 hover:text-pink-300 transition-colors">- Remove last</button>
                )}
              </div>
              <button onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium disabled:opacity-50 transition-all shadow-lg shadow-primary/20 text-sm">
                {compareMutation.isPending ? "Comparing..." : "Compare Prompts"}
              </button>
            </div>
            {compareMutation.data && (
              <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="glass rounded-2xl p-6">
                <h2 className="text-lg font-semibold text-foreground/90 mb-4 border-b border-white/10 pb-4">Comparison Results</h2>
                <div className="grid grid-cols-1 gap-4">
                  {toComparisonItems(compareMutation.data).map((p: any, i: number) => (
                    <motion.div
                      initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
                      key={i}
                      className="bg-black/40 border border-white/5 rounded-xl p-4 space-y-2 hover:bg-white/5 transition-colors"
                    >
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-sm text-foreground/80">Variant {i + 1}</span>
                        <span className="text-xs font-mono text-muted-foreground bg-white/5 px-2 py-0.5 rounded">{p.word_count} words</span>
                      </div>
                      <div className="flex gap-2 flex-wrap">
                        {toTechniqueList(p).map((t: string) => (
                          <span key={t} className="px-2 py-0.5 rounded-lg bg-primary/10 border border-primary/20 text-primary text-xs font-semibold">{t}</span>
                        ))}
                        {toTechniqueList(p).length === 0 && (
                          <span className="text-xs text-muted-foreground italic">No techniques detected</span>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {techniques.data && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="glass rounded-2xl p-6 space-y-4"
        >
          <h2 className="text-lg font-semibold text-foreground/90 pb-2 border-b border-white/10">Prompting Techniques</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {((techniques.data as any).techniques || []).map((t: any) => (
              <div key={t.name} className="bg-black/20 border border-white/5 rounded-xl p-4 space-y-2 hover:bg-white/5 transition-colors group cursor-default">
                <h3 className="font-semibold text-primary text-xs uppercase tracking-widest group-hover:text-primary/80 transition-colors">{t.name}</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">{t.description}</p>
                <div className="pt-2 mt-auto border-t border-white/5">
                  <span className={cn(
                    "inline-block text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider",
                    t.difficulty === "beginner" ? "bg-emerald-500/10 text-emerald-400" :
                      t.difficulty === "intermediate" ? "bg-amber-500/10 text-amber-400" :
                        "bg-pink-500/10 text-pink-400"
                  )}>{t.difficulty}</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
