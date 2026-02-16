"use client";

import ChatInterface from '@/components/ChatInterface';
import DocumentManager from '@/components/DocumentManager';
import { ShieldCheck, BookOpen, Database } from 'lucide-react';
import { useState, useEffect } from 'react';


export default function Home() {
  const [activeTab, setActiveTab] = useState<'suggestions' | 'documents'>('suggestions');
  const [highRiskCustomers, setHighRiskCustomers] = useState<any[]>([]);

  useEffect(() => {
    // Fetch high-risk customers from Project B (Port 8000)
    fetch('http://localhost:8000/analytics/customers?limit=3')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) setHighRiskCustomers(data);
      })
      .catch(err => console.error("Failed to fetch high-risk customers:", err));
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50/50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8 gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Support Knowledge Base</h1>
            <p className="text-slate-500">AI-Powered Policy Assistant & Document Retrieval</p>
          </div>
          
          <div className="flex gap-3">
            <div className="px-4 py-2 bg-white rounded-lg border border-slate-200 shadow-sm flex items-center gap-2 text-sm font-medium text-slate-600">
              <Database className="w-4 h-4 text-emerald-500" />
              <span>RAG Active</span>
            </div>
            <div className="px-4 py-2 bg-white rounded-lg border border-slate-200 shadow-sm flex items-center gap-2 text-sm font-medium text-slate-600">
              <ShieldCheck className="w-4 h-4 text-indigo-500" />
              <span>Internal Only</span>
            </div>
          </div>
        </header>

        {/* content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Main Chat Area */}
          <section className="lg:col-span-2">
            <ChatInterface />
          </section>

          {/* Sidebar / Context */}
          <aside className="space-y-6">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-white/60 shadow-lg">
              
              <div className="flex items-center gap-4 mb-6 border-b border-slate-100 pb-2">
                <button 
                  onClick={() => setActiveTab('suggestions')}
                  className={`pb-2 text-sm font-medium transition-colors ${activeTab === 'suggestions' ? 'text-indigo-600 border-b-2 border-indigo-600' : 'text-slate-500 hover:text-slate-700'}`}
                >
                  Suggestions
                </button>
                <button 
                  onClick={() => setActiveTab('documents')}
                  className={`pb-2 text-sm font-medium transition-colors ${activeTab === 'documents' ? 'text-indigo-600 border-b-2 border-indigo-600' : 'text-slate-500 hover:text-slate-700'}`}
                >
                  <span className="flex items-center gap-1">
                    <BookOpen className="w-3 h-3" />
                    Docs
                  </span>
                </button>
              </div>

              {activeTab === 'suggestions' ? (
                <div className="space-y-3">
                  <h3 className="font-semibold text-slate-800 mb-2 flex items-center gap-2 text-xs uppercase tracking-wider">
                    High Risk Customers (Live)
                  </h3>
                  {highRiskCustomers.length > 0 ? highRiskCustomers.map((c, i) => (
                     <button key={i} className="w-full text-left p-3 rounded-xl bg-red-50 hover:bg-red-100 text-sm text-slate-700 hover:text-red-700 transition-colors border border-red-100 group">
                        <div className="font-medium text-xs text-red-600 mb-1">Customer {c.id}</div>
                        "Draft a retention offer for a {c.contract} customer with ${c.charges}/mo charges."
                     </button>
                  )) : (
                    <p className="text-xs text-slate-400 italic">No high-risk customers found via API.</p>
                  )}

                  <h3 className="font-semibold text-slate-800 mb-2 mt-6 flex items-center gap-2 text-xs uppercase tracking-wider">
                    General Queries
                  </h3>
                  <button className="w-full text-left p-3 rounded-xl bg-slate-50 hover:bg-indigo-50 text-sm text-slate-600 hover:text-indigo-700 transition-colors border border-slate-100">
                    "What is the standard retention offer for Platinum tier?"
                  </button>
                  <button className="w-full text-left p-3 rounded-xl bg-slate-50 hover:bg-indigo-50 text-sm text-slate-600 hover:text-indigo-700 transition-colors border border-slate-100">
                    "Troubleshoot fiber LOS light."
                  </button>
                </div>
              ) : (
                <DocumentManager />
              )}
            </div>

            <div className="bg-gradient-to-br from-indigo-600 to-violet-600 rounded-2xl p-6 text-white shadow-xl">
              <h3 className="font-semibold mb-2">Churn Alert Integration</h3>
              <p className="text-indigo-100 text-sm mb-4">
                This system is linked to the Monthly Churn Predictor (Project B). Use high-risk customer profiles to guide your queries.
              </p>
              <div className="w-full h-1 bg-white/20 rounded-full overflow-hidden">
                <div className="w-3/4 h-full bg-white/40" />
              </div>
            </div>
          </aside>
        
        </div>
      </div>
    </main>
  );
}
