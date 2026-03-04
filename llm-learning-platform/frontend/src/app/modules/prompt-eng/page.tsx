"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { promptEngApi } from "@/lib/api";

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
    if (Array.isArray(data?.detected_techniques)) {
      return data.detected_techniques;
    }
    const flags = data?.techniques_detected;
    if (flags && typeof flags === "object") {
      return Object.entries(flags)
        .filter(([, enabled]) => Boolean(enabled))
        .map(([name]) => name.replace(/_/g, " "));
    }
    return [];
  };

  const toComparisonItems = (data: any): any[] => {
    if (Array.isArray(data?.prompts)) {
      return data.prompts;
    }
    if (Array.isArray(data?.analyses)) {
      return data.analyses;
    }
    return [];
  };

  const tabs = [
    { id: "templates" as const, label: "Templates", icon: "📝" },
    { id: "analyze" as const, label: "Prompt Analyzer", icon: "🔍" },
    { id: "compare" as const, label: "Compare Prompts", icon: "⚔️" },
  ];

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">Prompt Engineering Lab</h1>
        <p className="text-muted-foreground">
          Master prompt techniques — use templates, analyze prompt quality, and compare prompt variants.
        </p>
      </div>

      <div className="flex gap-2 border-b border-border pb-2">
        {tabs.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-colors ${activeTab === tab.id
              ? "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 border-b-2 border-primary-500"
              : "text-muted-foreground hover:text-foreground"}`}>
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "templates" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Prompt Templates</h2>
            <div>
              <label className="text-xs font-medium mb-1 block">Select Template</label>
              <select value={selectedTemplate} onChange={(e) => setSelectedTemplate(e.target.value)}
                className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm">
                {(templates.data as any)?.templates?.map((t: any) => (
                  <option key={t.name} value={t.name}>{t.name} — {t.description}</option>
                )) || <option>Loading...</option>}
              </select>
            </div>
            <div>
              <label className="text-xs font-medium mb-1 block">Variables</label>
              {Object.entries(variables).map(([k, v]) => (
                <div key={k} className="flex gap-2 mb-2">
                  <input value={k} disabled className="w-32 rounded-lg border border-border bg-muted px-2 py-1 text-xs font-mono" />
                  <input value={v} onChange={(e) => setVariables((prev) => ({ ...prev, [k]: e.target.value }))}
                    className="flex-1 rounded-lg border border-border bg-background px-2 py-1 text-xs" />
                </div>
              ))}
              <button onClick={() => setVariables((prev) => ({ ...prev, [`var_${Object.keys(prev).length}`]: "" }))}
                className="text-xs text-primary-500 hover:text-primary-600">+ Add variable</button>
            </div>
            <button onClick={() => renderMutation.mutate()} disabled={renderMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {renderMutation.isPending ? "Rendering..." : "Render Template"}
            </button>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Rendered Prompt</h2>
            {renderMutation.data ? (
              <div className="space-y-3">
                <pre className="bg-background rounded-xl p-4 text-sm font-mono whitespace-pre-wrap overflow-auto max-h-80">
                  {(renderMutation.data as any).rendered}
                </pre>
                {(renderMutation.data as any).metadata && (
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries((renderMutation.data as any).metadata).map(([k, v]) => (
                      <div key={k} className="bg-background rounded-lg p-2">
                        <div className="text-xs text-muted-foreground">{k}</div>
                        <div className="text-xs font-mono">{String(v)}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (<div className="text-center text-muted-foreground py-16">Select a template and render it</div>)}
          </div>
        </div>
      )}

      {activeTab === "analyze" && (
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Prompt Analyzer</h2>
            <p className="text-sm text-muted-foreground">Analyze a prompt to detect techniques, measure complexity, and get improvement suggestions.</p>
            <textarea value={analyzeText} onChange={(e) => setAnalyzeText(e.target.value)} rows={6}
              className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm font-mono resize-none" />
            <button onClick={() => analyzeMutation.mutate()} disabled={analyzeMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {analyzeMutation.isPending ? "Analyzing..." : "Analyze Prompt"}
            </button>
          </div>
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Analysis Results</h2>
            {analyzeMutation.data ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  {["word_count", "character_count", "line_count", "estimated_tokens", "num_examples"].map((k) => (
                    <div key={k} className="bg-background rounded-xl p-3">
                      <div className="text-xs text-muted-foreground">{k.replace(/_/g, " ")}</div>
                      <div className="font-mono font-bold text-primary-500">{String((analyzeMutation.data as any)[k])}</div>
                    </div>
                  ))}
                </div>
                {toTechniqueList(analyzeMutation.data).length > 0 && (
                  <div>
                    <div className="text-sm font-semibold mb-2">Detected Techniques</div>
                    <div className="flex gap-2 flex-wrap">
                      {toTechniqueList(analyzeMutation.data).map((t: string) => (
                        <span key={t} className="px-2 py-1 rounded-lg bg-green-100 dark:bg-green-900/30 text-green-600 text-xs">{t}</span>
                      ))}
                    </div>
                  </div>
                )}
                {(analyzeMutation.data as any).suggestions?.length > 0 && (
                  <div>
                    <div className="text-sm font-semibold mb-2">Suggestions</div>
                    <ul className="space-y-1">
                      {(analyzeMutation.data as any).suggestions.map((s: string, i: number) => (
                        <li key={i} className="text-xs text-muted-foreground flex gap-2"><span className="text-primary-500">→</span> {s}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (<div className="text-center text-muted-foreground py-16">Enter a prompt and analyze it</div>)}
          </div>
        </div>
      )}

      {activeTab === "compare" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 space-y-4">
            <h2 className="text-lg font-semibold">Compare Prompt Variants</h2>
            {compareTexts.map((text, i) => (
              <div key={i}>
                <label className="text-xs font-medium mb-1 block">Variant {i + 1}</label>
                <textarea value={text} onChange={(e) => {
                  const next = [...compareTexts]; next[i] = e.target.value; setCompareTexts(next);
                }} rows={2} className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm font-mono resize-none" />
              </div>
            ))}
            <div className="flex gap-2">
              <button onClick={() => setCompareTexts([...compareTexts, ""])}
                className="text-xs text-primary-500 hover:text-primary-600">+ Add variant</button>
              {compareTexts.length > 2 && (
                <button onClick={() => setCompareTexts(compareTexts.slice(0, -1))}
                  className="text-xs text-red-500 hover:text-red-600">- Remove last</button>
              )}
            </div>
            <button onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}
              className="w-full py-2 rounded-xl bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors">
              {compareMutation.isPending ? "Comparing..." : "Compare Prompts"}
            </button>
          </div>
          {compareMutation.data && (
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Comparison Results</h2>
              <div className="grid grid-cols-1 gap-4">
                {toComparisonItems(compareMutation.data).map((p: any, i: number) => (
                  <div key={i} className="bg-background rounded-xl p-4 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-sm">Variant {i + 1}</span>
                      <span className="text-xs text-muted-foreground">{p.word_count} words</span>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      {toTechniqueList(p).map((t: string) => (
                        <span key={t} className="px-2 py-0.5 rounded bg-primary-100 dark:bg-primary-900/30 text-primary-600 text-xs">{t}</span>
                      ))}
                      {toTechniqueList(p).length === 0 && (
                        <span className="text-xs text-muted-foreground">No techniques detected</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {techniques.data && (
        <div className="glass rounded-2xl p-6 space-y-4">
          <h2 className="text-lg font-semibold">Prompting Techniques</h2>
          <div className="grid grid-cols-4 gap-4">
            {((techniques.data as any).techniques || []).map((t: any) => (
              <div key={t.name} className="bg-background rounded-xl p-4 space-y-1">
                <h3 className="font-semibold text-primary-500 text-sm">{t.name}</h3>
                <p className="text-xs text-muted-foreground">{t.description}</p>
                <span className={`inline-block text-[10px] px-1.5 py-0.5 rounded ${
                  t.difficulty === "beginner" ? "bg-green-100 text-green-700" :
                  t.difficulty === "intermediate" ? "bg-yellow-100 text-yellow-700" :
                  "bg-red-100 text-red-700"
                }`}>{t.difficulty}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
