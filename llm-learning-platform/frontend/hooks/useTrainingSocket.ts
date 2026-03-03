"use client";

import { useEffect, useCallback, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { useModelStore, TrainingMetrics, GPTConfig } from '@/stores/modelStore';

interface TrainingConfig {
  model_id: string;
  dataset: string;
  batch_size: number;
  learning_rate: number;
  min_learning_rate?: number;
  warmup_steps: number;
  max_steps: number;
  grad_clip: number;
  optimizer: 'adam' | 'adamw' | 'sgd';
  config: GPTConfig;
}

interface UseTrainingSocketReturn {
  isConnected: boolean;
  error: string | null;
  startTraining: (config: TrainingConfig) => void;
  stopTraining: () => void;
  pauseTraining: () => void;
  resumeTraining: () => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';

export function useTrainingSocket(): UseTrainingSocketReturn {
  const socketRef = useRef<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const updateMetrics = useModelStore((state) => state.updateMetrics);
  const startTrainingStore = useModelStore((state) => state.startTraining);
  const stopTrainingStore = useModelStore((state) => state.stopTraining);
  const activeModelId = useModelStore((state) => state.activeModelId);

  // Initialize socket connection
  useEffect(() => {
    const socket = io(WS_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('Connected to training server');
      setIsConnected(true);
      setError(null);
    });

    socket.on('disconnect', (reason) => {
      console.log('Disconnected from training server:', reason);
      setIsConnected(false);
    });

    socket.on('connect_error', (err) => {
      console.error('Connection error:', err);
      setError('Failed to connect to training server');
      setIsConnected(false);
    });

    // Training events
    socket.on('training:metrics', (data: TrainingMetrics) => {
      console.log('Received metrics:', data);
      updateMetrics(data);
    });

    socket.on('training:started', (data: { session_id: string; model_id: string }) => {
      console.log('Training started:', data);
      startTrainingStore();
    });

    socket.on('training:stopped', () => {
      console.log('Training stopped');
      stopTrainingStore();
    });

    socket.on('training:complete', (data: { summary: string }) => {
      console.log('Training complete:', data);
      stopTrainingStore();
    });

    socket.on('training:error', (error: { message: string }) => {
      console.error('Training error:', error);
      setError(error.message);
      stopTrainingStore();
    });

    socket.on('training:step', (data: TrainingMetrics) => {
      updateMetrics(data);
    });

    socket.on('training:epoch', (data: { epoch: number; metrics: TrainingMetrics }) => {
      console.log('Epoch complete:', data.epoch);
      updateMetrics(data.metrics);
    });

    return () => {
      socket.disconnect();
    };
  }, [updateMetrics, startTrainingStore, stopTrainingStore]);

  const startTraining = useCallback((config: TrainingConfig) => {
    if (!socketRef.current?.connected) {
      setError('Not connected to training server');
      return;
    }

    setError(null);
    socketRef.current.emit('training:start', config);
  }, []);

  const stopTraining = useCallback(() => {
    if (!socketRef.current?.connected) return;
    socketRef.current.emit('training:stop', { model_id: activeModelId });
  }, [activeModelId]);

  const pauseTraining = useCallback(() => {
    if (!socketRef.current?.connected) return;
    socketRef.current.emit('training:pause', { model_id: activeModelId });
  }, [activeModelId]);

  const resumeTraining = useCallback(() => {
    if (!socketRef.current?.connected) return;
    socketRef.current.emit('training:resume', { model_id: activeModelId });
  }, [activeModelId]);

  return {
    isConnected,
    error,
    startTraining,
    stopTraining,
    pauseTraining,
    resumeTraining,
  };
}

// Hook for checking backend health
export function useBackendHealth() {
  const [isHealthy, setIsHealthy] = useState(false);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/api/health`);
        setIsHealthy(response.ok);
      } catch {
        setIsHealthy(false);
      } finally {
        setIsChecking(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s

    return () => clearInterval(interval);
  }, []);

  return { isHealthy, isChecking };
}
