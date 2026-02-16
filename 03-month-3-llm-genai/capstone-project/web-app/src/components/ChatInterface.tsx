"use client";

import { useState, useEffect, useRef } from "react";
import { Send, Upload, FileText, User, Bot, AlertTriangle } from "lucide-react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import { sendQuery, ChatResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

type Message = {
  id: string;
  role: "user" | "bot";
  content: string;
  sources?: string[];
  context?: string[];
};

export default function ChatInterface() {
  const [query, setQuery] = useState("");
  const [strategy, setStrategy] = useState("simple");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "bot",
      content: "Hello! I'm your Telecom Policy Assistant. Ask me about retention offers, troubleshooting steps, or internal procedures."
    }
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: query
    };

    setMessages(prev => [...prev, userMsg]);
    setQuery("");
    setLoading(true);

    try {
      const data = await sendQuery(userMsg.content, strategy);
      
      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        content: data.error ? "Sorry, I encountered an error." : data.answer,
        sources: data.sources,
        context: data.context
      };
      
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[700px] w-full bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-slate-100 bg-white flex items-center justify-between shadow-sm z-10">
        <div className="flex items-center gap-4">
          <div className="p-2.5 bg-indigo-50 rounded-xl border border-indigo-100">
            <Bot className="w-6 h-6 text-indigo-600" />
          </div>
          <div>
            <h2 className="font-bold text-slate-900 text-lg">Policy Assistant</h2>
            <p className="text-sm text-slate-500">RAG-Powered Knowledge Base</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
           <label className="text-xs font-bold text-slate-400 uppercase tracking-wider pr-1">Strategy:</label>
           <select 
             value={strategy}
             onChange={(e) => setStrategy(e.target.value)}
             className="bg-slate-50 border border-slate-200 text-slate-700 text-xs font-semibold rounded-lg focus:ring-indigo-500 focus:border-indigo-500 p-2 outline-none cursor-pointer hover:bg-white transition-colors"
           >
              <option value="simple">Simple RAG</option>
              <option value="multi_query">Multi-Query</option>
              <option value="hyde">HyDE RAG</option>
              <option value="verified">Verified RAG</option>
           </select>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50/50">
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn(
              "flex gap-4 max-w-[85%]",
              msg.role === "user" ? "ml-auto flex-row-reverse" : ""
            )}
          >
            <div className={cn(
              "w-10 h-10 rounded-full flex items-center justify-center shrink-0 shadow-sm border border-slate-100",
              msg.role === "user" ? "bg-white" : "bg-indigo-600"
            )}>
              {msg.role === "user" ? 
                <User className="w-5 h-5 text-slate-600" /> : 
                <Bot className="w-5 h-5 text-white" />
              }
            </div>
            
            <div className="flex flex-col gap-2 w-full">
              <div className={cn(
                "p-5 rounded-2xl text-base leading-relaxed shadow-sm",
                msg.role === "user" 
                  ? "bg-indigo-600 text-white rounded-tr-none shadow-indigo-200" 
                  : "bg-white border border-slate-200 text-slate-800 rounded-tl-none prose prose-indigo prose-sm max-w-none"
              )}>
                {msg.role === "user" ? (
                  msg.content
                ) : (
                  <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                    {msg.content}
                  </ReactMarkdown>
                )}
              </div>
              
              {/* Citations */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-1 px-1">
                  {msg.sources.map((source, idx) => (
                    <span key={idx} className="flex items-center gap-1.5 text-[11px] uppercase font-bold tracking-wider text-slate-500 bg-white px-3 py-1.5 rounded-lg border border-slate-200 shadow-sm">
                      <FileText className="w-3.5 h-3.5 text-indigo-500" />
                      {source || "Unknown Source"}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        ))}
        
        {loading && (
          <div className="flex gap-4 max-w-[85%]">
             <div className="w-10 h-10 rounded-full bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-200">
                <Bot className="w-5 h-5 text-white" />
             </div>
             <div className="bg-white border border-slate-200 shadow-sm p-5 rounded-2xl rounded-tl-none">
               <div className="flex gap-1.5">
                 <span className="w-2.5 h-2.5 bg-slate-400 rounded-full animate-bounce" />
                 <span className="w-2.5 h-2.5 bg-slate-400 rounded-full animate-bounce delay-75" />
                 <span className="w-2.5 h-2.5 bg-slate-400 rounded-full animate-bounce delay-150" />
               </div>
               <p className="text-[10px] text-slate-400 mt-2 font-bold uppercase tracking-widest animate-pulse">
                 {strategy === 'verified' ? 'Verifying facts...' : strategy === 'multi_query' ? 'Exploring angles...' : strategy === 'hyde' ? 'Thinking hypothetically...' : 'Searching...'}
               </p>
             </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-5 bg-white border-t border-slate-200 shadow-[0_-5px_20px_-10px_rgba(0,0,0,0.05)]">
        <div className="relative flex items-center">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about retention policies, troubleshooting steps..."
            className="w-full pl-5 pr-14 py-4 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all text-base text-slate-800 placeholder:text-slate-400 font-medium"
          />
          <button 
            type="submit" 
            disabled={!query.trim() || loading}
            className="absolute right-2.5 p-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:hover:bg-indigo-600 text-white rounded-lg transition-all shadow-md hover:shadow-lg disabled:shadow-none"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </form>
    </div>
  );
}
