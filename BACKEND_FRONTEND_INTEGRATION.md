# Backend-Frontend Integration Guide

## API Endpoint Mapping

### Backend Endpoints (Available)

#### Model Management
| Endpoint | Method | Frontend Usage |
|----------|--------|----------------|
| `/api/model/create` | POST | Create new GPT model |
| `/api/model/{id}` | GET | Get model info |
| `/api/model/{id}/reset` | POST | Reset model parameters |
| `/api/model/{id}` | DELETE | Delete model |
| `/api/models` | GET | List all models |

#### Training
| Endpoint | Method | Frontend Usage |
|----------|--------|----------------|
| `/api/training/start` | POST | Start training session |
| `/api/training/{id}/stop` | POST | Stop training |
| `/api/training/{id}/status` | GET | Get training status |
| `/api/training/{id}/history` | GET | Get training history |
| `/ws/training/{session_id}` | WebSocket | Real-time updates |

#### Inference
| Endpoint | Method | Frontend Usage |
|----------|--------|----------------|
| `/api/inference/generate` | POST | Generate text |
| `/api/inference/tokenize` | POST | Tokenize text |
| `/api/inference/forward` | POST | Forward pass |

#### Visualization
| Endpoint | Method | Frontend Usage |
|----------|--------|----------------|
| `/api/viz/attention/{model_id}` | GET | Get attention data |
| `/api/viz/embeddings/{model_id}` | GET | Get embeddings |
| `/api/compute/attention` | POST | Compute attention |
| `/api/compute/embeddings` | POST | Compute embeddings |

#### Datasets
| Endpoint | Method | Frontend Usage |
|----------|--------|----------------|
| `/api/datasets` | GET | List datasets |
| `/api/datasets/{id}` | GET | Get dataset |
| `/api/datasets/upload` | POST | Upload dataset |

#### System
| Endpoint | Method | Frontend Usage |
|----------|--------|----------------|
| `/api/gpu/status` | GET | GPU status |
| `/api/health` | GET | Health check |

---

## WebSocket Events

### Client → Server
| Event | Payload | Description |
|-------|---------|-------------|
| `training:start` | `TrainingConfig` | Start training |
| `training:stop` | `{ session_id }` | Stop training |
| `training:pause` | `{ session_id }` | Pause training |
| `training:resume` | `{ session_id }` | Resume training |

### Server → Client
| Event | Payload | Description |
|-------|---------|-------------|
| `training:metrics` | `TrainingMetrics` | Step metrics |
| `training:started` | `{ session_id }` | Training started |
| `training:stopped` | - | Training stopped |
| `training:complete` | `{ summary }` | Training finished |
| `training:error` | `{ message }` | Error occurred |
| `training:step` | `TrainingMetrics` | Step complete |
| `training:epoch` | `{ epoch, metrics }` | Epoch complete |

---

## Frontend Implementation Status

### ✅ Already Implemented
- [x] Zustand store for state management
- [x] WebSocket hook structure
- [x] API client library structure
- [x] Training Dashboard UI
- [x] Model Configurator UI

### ⚠️ Needs Connection to Real APIs
- [ ] WebSocket actual connection testing
- [ ] Training start/stop API integration
- [ ] Model creation API integration
- [ ] Real-time metrics streaming
- [ ] Dataset fetching
- [ ] Attention data fetching

---

## Environment Configuration

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### Production
```
NEXT_PUBLIC_API_URL=https://api.llm-learning-platform.vercel.app
NEXT_PUBLIC_WS_URL=wss://api.llm-learning-platform.vercel.app
```

---

## Testing Backend Connection

### 1. Health Check
```bash
curl http://localhost:8000/api/health
```

### 2. Create Model
```bash
curl -X POST http://localhost:8000/api/model/create \
  -H "Content-Type: application/json" \
  -d '{
    "vocab_size": 256,
    "d_model": 128,
    "num_layers": 4,
    "num_heads": 4,
    "backend": "pytorch"
  }'
```

### 3. Start Training
```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "your-model-id",
    "batch_size": 16,
    "learning_rate": 0.001,
    "max_steps": 1000
  }'
```

### 4. WebSocket Test
```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/training/your-session-id');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

---

## Data Flow Architecture

```
Frontend (Next.js)
├── Zustand Store (State Management)
├── React Query (Server State)
├── WebSocket Client (Real-time)
└── API Client (REST)
         │
         ▼
Backend (FastAPI)
├── REST Endpoints
├── WebSocket Manager
├── Training Engine
└── Model Storage
```

---

## Error Handling Strategy

### Frontend
1. **API Errors**: Display user-friendly messages
2. **WebSocket Disconnection**: Auto-retry with backoff
3. **Validation Errors**: Show inline field errors
4. **Network Errors**: Offline mode indication

### Backend
1. **Validation**: Pydantic models with detailed errors
2. **WebSocket**: Connection limits and heartbeats
3. **Training**: Graceful error recovery
4. **Logging**: Structured logging for debugging

---

## Security Considerations

### CORS
- Backend allows all origins in development
- Production: Restrict to specific domains

### Rate Limiting
- Training start: 5 req/min per IP
- Model creation: 10 req/min per IP
- WebSocket: Max 5 connections per session

### Authentication (Future)
- JWT tokens for API access
- Session-based WebSocket auth
- API key for external access

---

## Performance Optimization

### Frontend
1. **Debouncing**: Config slider updates (300ms)
2. **Throttling**: WebSocket message handling
3. **Memoization**: Chart data processing
4. **Lazy Loading**: Heavy visualization components

### Backend
1. **Async Processing**: Training in background
2. **Caching**: Model configs in memory
3. **Streaming**: WebSocket batch updates
4. **Batching**: Metrics every N steps

---

## Deployment Checklist

### Backend Deployment
- [ ] Docker image built
- [ ] Environment variables set
- [ ] Database/Redis connected
- [ ] GPU drivers installed
- [ ] Health endpoint responding
- [ ] CORS origins configured

### Frontend Deployment
- [ ] API_URL environment variable set
- [ ] WS_URL environment variable set
- [ ] Build successful
- [ ] Static assets uploaded
- [ ] Routes tested

### Integration Testing
- [ ] Model creation works
- [ ] Training starts successfully
- [ ] WebSocket connects
- [ ] Metrics stream received
- [ ] Charts update in real-time

---

## Debugging Guide

### WebSocket Not Connecting
1. Check backend is running
2. Verify CORS settings
3. Check firewall rules
4. Test with wscat: `wscat -c ws://localhost:8000/api/ws/training/test`

### Training Not Starting
1. Check GPU availability
2. Verify model ID exists
3. Check dataset availability
4. Review backend logs

### Metrics Not Updating
1. Check WebSocket connection status
2. Verify training is actually running
3. Check browser console for errors
4. Test backend metrics endpoint directly

---

## Future Enhancements

### Short Term
- [ ] Add request retry logic
- [ ] Implement offline mode
- [ ] Add request caching
- [ ] Better error messages

### Long Term
- [ ] GraphQL API
- [ ] Server-Sent Events fallback
- [ ] Multi-region deployment
- [ ] CDN for static assets
