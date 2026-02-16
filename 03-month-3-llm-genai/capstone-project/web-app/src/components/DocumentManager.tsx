"use client";

import { useState, useEffect } from "react";
import { Upload, FileText, Trash2, RefreshCw } from "lucide-react";
import { toast } from "sonner";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch("http://localhost:8001/admin/documents");
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error("Failed to fetch documents", err);
      toast.error("Failed to load documents");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    
    setUploading(true);
    const toastId = toast.loading("Uploading file...");
    
    try {
      const res = await fetch("http://localhost:8001/admin/upload", {
        method: "POST",
        body: formData,
      });
      
      if (res.ok) {
        await fetchDocuments();
        toast.success("Upload successful!", {
          id: toastId,
          description: "Don't forget to re-sync the AI."
        });
      } else {
        toast.error("Upload failed", { id: toastId });
      }
    } catch (err) {
      console.error(err);
      toast.error("Error uploading file", { id: toastId });
    } finally {
      setUploading(false);
    }
  };

  const triggerIngestion = async () => {
    const toastId = toast.loading("Starting ingestion...");
    try {
      const res = await fetch("http://localhost:8001/ingest", { method: "POST" });
      const data = await res.json();
      toast.success(data.message || "Ingestion complete!", { id: toastId });
    } catch (err) {
      toast.error("Ingestion failed", { id: toastId });
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
          <FileText className="w-4 h-4 text-indigo-500" />
          Knowledge Base Items
        </h2>
        <button 
          onClick={triggerIngestion}
          className="text-xs font-semibold px-3 py-1.5 bg-indigo-50 text-indigo-600 rounded-lg hover:bg-indigo-100 transition-colors flex items-center gap-1.5"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Re-Sync AI
        </button>
      </div>

      <div className="mb-6">
        <label className="flex items-center justify-center w-full h-28 border-2 border-indigo-100 border-dashed rounded-xl cursor-pointer hover:bg-indigo-50 transition-all group relative overflow-hidden">
          <div className="absolute inset-0 bg-indigo-50/0 group-hover:bg-indigo-50/50 transition-colors" />
          <div className="flex flex-col items-center pt-5 pb-6 relative z-10">
            <div className="p-2 bg-indigo-50 rounded-full mb-3 group-hover:scale-110 transition-transform">
              <Upload className="w-5 h-5 text-indigo-500" />
            </div>
            <p className="text-sm font-medium text-slate-600 group-hover:text-indigo-700">
              {uploading ? "Uploading..." : "Click to upload"}
            </p>
            <p className="text-xs text-slate-400 mt-1">.md or .pdf files</p>
          </div>
          <input type="file" className="hidden" accept=".md,.pdf,.txt" onChange={handleUpload} disabled={uploading} />
        </label>
      </div>

      <div className="space-y-3 overflow-y-auto pr-2 custom-scrollbar flex-1">
        {loading ? (
          <div className="flex items-center justify-center h-20">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600"></div>
          </div>
        ) : documents.length === 0 ? (
          <div className="text-center py-8 bg-slate-50 rounded-xl border border-slate-100 border-dashed">
            <p className="text-sm text-slate-500 font-medium">No documents found.</p>
            <p className="text-xs text-slate-400 mt-1">Upload a file to get started.</p>
          </div>
        ) : (
          documents.map((doc, idx) => (
            <div key={idx} className="flex items-center justify-between p-4 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 hover:shadow-sm transition-all group">
              <div className="flex items-center gap-3 overflow-hidden">
                <div className="p-2 bg-slate-50 rounded-lg border border-slate-100 group-hover:bg-indigo-50 group-hover:border-indigo-100 transition-colors">
                  <FileText className="w-4 h-4 text-slate-400 group-hover:text-indigo-500" />
                </div>
                <span className="text-sm font-medium text-slate-700 truncate max-w-[140px]" title={doc}>
                  {doc}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-bold px-2 py-1 bg-emerald-50 text-emerald-600 rounded-md border border-emerald-100">INDEXED</span>
                <button 
                  onClick={async () => {
                    if(!confirm(`Delete ${doc}?`)) return;
                    try {
                      const res = await fetch(`http://localhost:8001/admin/documents/${doc}`, { method: 'DELETE' });
                      if (res.ok) fetchDocuments();
                    } catch (e) {
                      console.error(e);
                    }
                  }}
                  className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
