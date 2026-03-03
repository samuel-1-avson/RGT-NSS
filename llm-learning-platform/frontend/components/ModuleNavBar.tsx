'use client';

import React from 'react';
import Link from 'next/link';
import { ChevronLeft, ChevronRight, Keyboard } from 'lucide-react';
import { useModuleNavigation, useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { motion } from 'framer-motion';

/**
 * Module navigation bar shown at the bottom of each learning module page.
 * Shows previous/next module links and keyboard shortcut hints.
 */
export default function ModuleNavBar() {
    const { prev, next, currentIndex, totalModules } = useModuleNavigation();

    // Activate keyboard shortcuts
    useKeyboardShortcuts();

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-12 mb-8 mx-auto max-w-7xl px-4 sm:px-6 lg:px-8"
        >
            <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-4 flex items-center justify-between">
                {/* Previous */}
                {prev ? (
                    <Link
                        href={prev.href}
                        className="group flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium text-slate-600 hover:bg-slate-100 transition-colors"
                    >
                        <ChevronLeft className="h-4 w-4 text-slate-400 group-hover:-translate-x-0.5 transition-transform" />
                        <div>
                            <span className="text-[10px] text-slate-400 block">Previous</span>
                            <span className="text-slate-700">{prev.label}</span>
                        </div>
                    </Link>
                ) : (
                    <div />
                )}

                {/* Center — Progress & Shortcuts hint */}
                <div className="flex items-center gap-3">
                    <div className="flex gap-1">
                        {Array.from({ length: totalModules }).map((_, i) => (
                            <div
                                key={i}
                                className={`h-1.5 w-6 rounded-full transition-colors ${i === currentIndex ? 'bg-blue-500' : i < currentIndex ? 'bg-blue-200' : 'bg-slate-200'
                                    }`}
                            />
                        ))}
                    </div>
                    <div className="hidden sm:flex items-center gap-1 text-[10px] text-slate-400">
                        <Keyboard className="h-3 w-3" />
                        <span>← → to navigate</span>
                    </div>
                </div>

                {/* Next */}
                {next ? (
                    <Link
                        href={next.href}
                        className="group flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium text-slate-600 hover:bg-slate-100 transition-colors"
                    >
                        <div className="text-right">
                            <span className="text-[10px] text-slate-400 block">Next</span>
                            <span className="text-slate-700">{next.label}</span>
                        </div>
                        <ChevronRight className="h-4 w-4 text-slate-400 group-hover:translate-x-0.5 transition-transform" />
                    </Link>
                ) : (
                    <div />
                )}
            </div>
        </motion.div>
    );
}
