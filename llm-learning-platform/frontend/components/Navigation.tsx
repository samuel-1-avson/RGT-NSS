'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Brain, BookOpen, Code2, Layers, Eye, Box,
    GraduationCap, Wand2, BarChart3, FileText,
    Menu, X, Home, ChevronRight
} from 'lucide-react';

const navItems = [
    { href: '/', label: 'Home', icon: Home },
    { href: '/learn', label: 'Learn Hub', icon: BookOpen },
    { href: '/learn/tokenization', label: 'Tokenization', icon: Code2, indent: true },
    { href: '/learn/embeddings', label: 'Embeddings', icon: Layers, indent: true },
    { href: '/learn/attention', label: 'Attention', icon: Eye, indent: true },
    { href: '/learn/transformer', label: 'Transformer', icon: Box, indent: true },
    { href: '/learn/training', label: 'Training', icon: GraduationCap, indent: true },
    { href: '/learn/inference', label: 'Inference', icon: Wand2, indent: true },
    { href: '/train', label: 'Training Dashboard', icon: BarChart3 },
    { href: '/docs', label: 'Documentation', icon: FileText },
];

export default function Navigation() {
    const [isOpen, setIsOpen] = useState(false);
    const pathname = usePathname();

    return (
        <>
            {/* Mobile Toggle */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="fixed bottom-6 right-6 z-50 lg:hidden h-14 w-14 rounded-full bg-gradient-to-br from-blue-600 to-purple-600 text-white shadow-lg shadow-blue-200 flex items-center justify-center hover:scale-105 transition-transform"
                aria-label="Toggle navigation"
            >
                {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>

            {/* Overlay */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setIsOpen(false)}
                        className="fixed inset-0 bg-black/30 z-40 lg:hidden"
                    />
                )}
            </AnimatePresence>

            {/* Sidebar */}
            <AnimatePresence>
                {(isOpen || typeof window !== 'undefined') && (
                    <motion.aside
                        initial={{ x: -280 }}
                        animate={{ x: isOpen ? 0 : -280 }}
                        className={`fixed top-0 left-0 h-full w-[280px] bg-white/95 backdrop-blur-xl border-r border-slate-200 z-50 overflow-y-auto lg:translate-x-0 lg:static lg:z-auto ${isOpen ? '' : 'hidden lg:block'
                            }`}
                    >
                        {/* Logo */}
                        <div className="p-5 border-b border-slate-100">
                            <Link href="/" className="flex items-center gap-3 group" onClick={() => setIsOpen(false)}>
                                <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center group-hover:scale-105 transition-transform">
                                    <Brain className="h-5 w-5 text-white" />
                                </div>
                                <div>
                                    <h1 className="text-sm font-bold text-slate-900">LLM Learning</h1>
                                    <p className="text-[10px] text-slate-400 -mt-0.5">Interactive Platform</p>
                                </div>
                            </Link>
                        </div>

                        {/* Nav Items */}
                        <nav className="p-3 space-y-0.5">
                            {navItems.map((item) => {
                                const isActive = pathname === item.href;
                                const Icon = item.icon;

                                return (
                                    <Link
                                        key={item.href}
                                        href={item.href}
                                        onClick={() => setIsOpen(false)}
                                        className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all group ${item.indent ? 'ml-4' : ''
                                            } ${isActive
                                                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-md shadow-blue-100'
                                                : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                                            }`}
                                    >
                                        <Icon className={`h-4 w-4 ${isActive ? 'text-white' : 'text-slate-400 group-hover:text-slate-600'}`} />
                                        {item.label}
                                        {isActive && (
                                            <ChevronRight className="h-3 w-3 ml-auto text-white/70" />
                                        )}
                                    </Link>
                                );
                            })}
                        </nav>

                        {/* Footer */}
                        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-100 bg-white/90">
                            <p className="text-[10px] text-slate-400 text-center">
                                Built with a custom deep learning framework
                            </p>
                        </div>
                    </motion.aside>
                )}
            </AnimatePresence>
        </>
    );
}
