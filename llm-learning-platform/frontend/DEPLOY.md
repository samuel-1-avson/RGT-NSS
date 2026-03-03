# Deployment Guide

## Build Status
✅ **Build Successful** - All 12 pages generated

## Backend Configuration
The frontend is configured to connect to:
- **API URL**: `https://047b-154-161-146-65.ngrok-free.app`
- **WebSocket URL**: `wss://047b-154-161-146-65.ngrok-free.app`

## Deployment Options

### Option 1: GitHub Pages (Recommended)
The GitHub Actions workflow (`.github/workflows/deploy-frontend.yml`) is configured for automatic deployment.

**Setup:**
1. Push this code to a GitHub repository
2. Go to Settings → Pages
3. Set Source to "GitHub Actions"
4. The workflow will auto-deploy on push

**Environment Variables (GitHub):**
Go to Settings → Secrets and Variables → Actions → Variables, add:
- `NEXT_PUBLIC_API_URL` = `https://047b-154-161-146-65.ngrok-free.app`
- `NEXT_PUBLIC_WS_URL` = `wss://047b-154-161-146-65.ngrok-free.app`

### Option 2: Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd llm-learning-platform/frontend
vercel --prod
```

### Option 3: Netlify
```bash
# Install Netlify CLI
npm i -g netlify-cli

# Deploy
cd llm-learning-platform/frontend
netlify deploy --prod --dir=dist
```

### Option 4: Manual Upload
Upload the contents of the `dist/` folder to any static hosting service:
- AWS S3 + CloudFront
- Azure Static Web Apps
- Google Cloud Storage
- Surge.sh
- Any web server (Apache, Nginx, etc.)

## Build Output
```
dist/
├── index.html          # Home page
├── 404.html            # 404 page
├── train/index.html    # Training Dashboard
├── models/index.html   # Model Configurator
├── inference/index.html # Inference Page
├── learn/              # Educational modules
│   ├── index.html
│   ├── tokenization/
│   ├── attention/
│   ├── transformer/
│   └── llm-building/
└── _next/              # Static assets
```

## Post-Deployment Verification

1. **Check build**: Open the deployed URL
2. **Test backend connection**: Visit `/health` on the backend URL
3. **Verify WebSocket**: The Training Dashboard should show "Backend Connected"
4. **Test training**: Start a training session to verify real-time updates

## Updating the Backend URL

If the ngrok tunnel URL changes:

1. Update `next.config.mjs`:
```javascript
env: {
  NEXT_PUBLIC_API_URL: 'https://NEW-URL.ngrok-free.app',
  NEXT_PUBLIC_WS_URL: 'wss://NEW-URL.ngrok-free.app',
}
```

2. Rebuild:
```bash
npm run build
```

3. Redeploy the `dist/` folder

## Features Deployed

### API Integration
- ✅ Model management (create, get, list, reset, delete)
- ✅ Training control (start, stop, status, history)
- ✅ Real-time metrics via WebSocket
- ✅ Inference (generate, tokenize, forward)
- ✅ Visualization (attention heatmaps, embeddings)
- ✅ System health monitoring

### UI Components
- ✅ Training Dashboard with live charts
- ✅ Model Configurator with presets
- ✅ Educational learning modules
- ✅ Inference playground
- ✅ Real-time connection status

### State Management
- ✅ Zustand store with persistence
- ✅ Training metrics history
- ✅ Model checkpoints

## Troubleshooting

**Build fails:**
```bash
# Clean and rebuild
rm -rf dist .next
npm run build
```

**Backend not connecting:**
- Verify ngrok tunnel is running
- Check CORS settings on backend
- Confirm environment variables are set

**WebSocket issues:**
- Ensure `wss://` is used for HTTPS deployments
- Check firewall/proxy settings
- Verify backend WebSocket endpoint is active
