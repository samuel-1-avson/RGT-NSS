"use client";

import { useState, useEffect, useCallback, useRef } from 'react';
import { useModelStore, TrainingMetrics } from '@/stores/modelStore';
import { trainingApi, systemApi } from '@/lib/api';
import type { TrainingConfig } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || API_URL.replace('http', 'ws');

interface WebSocketMessage {
  type: string;
  data?: any;
  error?: string;
}

export function useTraining() {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const { 
    startTraining: startStoreTraining, 
    stopTraining: stopStoreTraining,
    updateMetrics,
    isTraining 
  } = useModelStore();

  // Cleanup WebSocket connection
  const cleanup = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  // Connect to WebSocket
  const connect = useCallback((sid: string) => {
    cleanup();
    
    const wsUrl = `${WS_URL}/ws/training/${sid}`;
    console.log('Connecting to WebSocket:', wsUrl);
    
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        
        // Subscribe to training updates
        ws.send(JSON.stringify({ 
          type: 'subscribe',
          session_id: sid 
        }));
      };
      
      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log('WebSocket message:', message);
          
          switch (message.type) {
            case 'training:metrics':
              if (message.data) {
                updateMetrics({
                  step: message.data.step,
                  loss: message.data.loss,
                  perplexity: message.data.perplexity,
                  learningRate: message.data.learning_rate,
                  gradNorm: message.data.grad_norm,
                  tokensPerSec: message.data.tokens_per_sec,
                });
              }
              break;
              
            case 'training:started':
              startStoreTraining();
              break;
              
            case 'training:stopped':
            case 'training:completed':
              stopStoreTraining();
              break;
              
            case 'training:error':
              setError(message.error || 'Training error occurred');
              stopStoreTraining();
              break;
              
            case 'connected':
              console.log('WebSocket connection confirmed');
              break;
              
            default:
              console.log('Unknown message type:', message.type);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };
      
      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        setError('WebSocket connection error');
        setIsConnected(false);
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt reconnection if still training
        if (isTraining) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('Attempting to reconnect...');
            connect(sid);
          }, 3000);
        }
      };
      
    } catch (err: any) {
      console.error('Failed to create WebSocket:', err);
      setError(err.message || 'Failed to connect to WebSocket');
    }
  }, [cleanup, isTraining, startStoreTraining, stopStoreTraining, updateMetrics]);

  // Start training
  const startTraining = useCallback(async (config: TrainingConfig) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Call REST API to start training
      const { data, error: apiError } = await trainingApi.start(config);
      
      if (apiError) {
        throw new Error(apiError);
      }
      
      if (!data?.session_id) {
        throw new Error('No session ID returned from server');
      }
      
      const sid = data.session_id;
      setSessionId(sid);
      
      // Connect to WebSocket for real-time updates
      connect(sid);
      
      // Update store state
      startStoreTraining();
      
      return sid;
    } catch (err: any) {
      console.error('Failed to start training:', err);
      setError(err.message || 'Failed to start training');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [connect, startStoreTraining]);

  // Stop training
  const stopTraining = useCallback(async () => {
    if (!sessionId) return;
    
    try {
      // Send stop via WebSocket if connected
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'stop_training'
        }));
      }
      
      // Also call REST API to stop
      await trainingApi.stop(sessionId);
      
      stopStoreTraining();
      cleanup();
    } catch (err: any) {
      console.error('Failed to stop training:', err);
      setError(err.message || 'Failed to stop training');
    }
  }, [sessionId, stopStoreTraining, cleanup]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  return {
    isConnected,
    isLoading,
    error,
    sessionId,
    startTraining,
    stopTraining,
  };
}

// Hook for checking backend health
export function useBackendHealth() {
  const [isHealthy, setIsHealthy] = useState(false);
  const [gpuStatus, setGpuStatus] = useState<any>(null);
  const [version, setVersion] = useState<string>('');

  useEffect(() => {
    const checkHealth = async () => {
      const { data, error } = await systemApi.health();
      
      if (data && !error) {
        setIsHealthy(data.status === 'ok' || data.status === 'healthy');
        setGpuStatus(data.gpu);
        setVersion(data.version || '');
      } else {
        setIsHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000);

    return () => clearInterval(interval);
  }, []);

  return { isHealthy, gpuStatus, version };
}

// Hook for getting training status
export function useTrainingStatus(sessionId: string | null) {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setStatus(null);
      return;
    }

    const fetchStatus = async () => {
      setLoading(true);
      const { data, error: apiError } = await trainingApi.getStatus(sessionId);
      
      if (data) {
        setStatus(data);
      }
      if (apiError) {
        setError(apiError);
      }
      setLoading(false);
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);

    return () => clearInterval(interval);
  }, [sessionId]);

  return { status, loading, error };
}

export default useTraining;
