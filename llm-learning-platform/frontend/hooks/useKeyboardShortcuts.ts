'use client';

import React, { useEffect, useCallback } from 'react';
import { useRouter, usePathname } from 'next/navigation';

const MODULE_ORDER = [
    '/learn/tokenization',
    '/learn/embeddings',
    '/learn/attention',
    '/learn/transformer',
    '/learn/training',
    '/learn/inference',
];

/**
 * Global keyboard shortcuts for the learning platform.
 * 
 * Shortcuts:
 *   ← / → : Navigate between modules (when on a module page)
 *   Shift + H : Go home
 *   Shift + L : Go to learn hub
 *   Shift + D : Go to docs
 *   Shift + T : Go to training dashboard
 *   Escape   : Go back to learn hub
 */
export function useKeyboardShortcuts() {
    const router = useRouter();
    const pathname = usePathname();

    const handleKeyDown = useCallback((e: KeyboardEvent) => {
        // Don't trigger when typing in inputs
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

        const currentIndex = MODULE_ORDER.indexOf(pathname);

        switch (true) {
            // Arrow Left — previous module
            case e.key === 'ArrowLeft' && !e.shiftKey && !e.ctrlKey && !e.metaKey:
                if (currentIndex > 0) {
                    e.preventDefault();
                    router.push(MODULE_ORDER[currentIndex - 1]);
                }
                break;

            // Arrow Right — next module
            case e.key === 'ArrowRight' && !e.shiftKey && !e.ctrlKey && !e.metaKey:
                if (currentIndex >= 0 && currentIndex < MODULE_ORDER.length - 1) {
                    e.preventDefault();
                    router.push(MODULE_ORDER[currentIndex + 1]);
                }
                break;

            // Shift+H — Home
            case e.key === 'H' && e.shiftKey && !e.ctrlKey:
                e.preventDefault();
                router.push('/');
                break;

            // Shift+L — Learn hub
            case e.key === 'L' && e.shiftKey && !e.ctrlKey:
                e.preventDefault();
                router.push('/learn');
                break;

            // Shift+D — Docs
            case e.key === 'D' && e.shiftKey && !e.ctrlKey:
                e.preventDefault();
                router.push('/docs');
                break;

            // Shift+T — Training dashboard
            case e.key === 'T' && e.shiftKey && !e.ctrlKey:
                e.preventDefault();
                router.push('/train');
                break;

            // Escape — back to learn hub
            case e.key === 'Escape':
                if (pathname.startsWith('/learn/')) {
                    e.preventDefault();
                    router.push('/learn');
                }
                break;
        }
    }, [router, pathname]);

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown]);
}

/**
 * Returns the previous and next module relative to the current pathname.
 */
export function useModuleNavigation() {
    const pathname = usePathname();
    const currentIndex = MODULE_ORDER.indexOf(pathname);

    const modules: Record<string, string> = {
        '/learn/tokenization': 'Tokenization',
        '/learn/embeddings': 'Embeddings',
        '/learn/attention': 'Attention',
        '/learn/transformer': 'Transformer',
        '/learn/training': 'Training',
        '/learn/inference': 'Inference',
    };

    return {
        prev: currentIndex > 0 ? { href: MODULE_ORDER[currentIndex - 1], label: modules[MODULE_ORDER[currentIndex - 1]] } : null,
        next: currentIndex >= 0 && currentIndex < MODULE_ORDER.length - 1
            ? { href: MODULE_ORDER[currentIndex + 1], label: modules[MODULE_ORDER[currentIndex + 1]] }
            : null,
        currentModule: modules[pathname] || null,
        currentIndex,
        totalModules: MODULE_ORDER.length,
    };
}
