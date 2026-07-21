# AI-lixir Scientific OS — Frontend

React (Vite) SPA deployed to Vercel. Talks to the FastAPI backend on Hugging Face Spaces.

## Setup

```bash
npm install
```

## Development

```bash
npm run dev
```

Set your backend URL in [`src/config.js`](./src/config.js):
```js
export const BACKEND_URL = "https://omarfadlallah-scientific-operating-system.hf.space";
```

## Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy (from inside the frontend/ folder)
vercel
```

Or connect your GitHub repo to Vercel and set **Root Directory** = `frontend`.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Chat — streaming AI responses with voice |
| `/monitor` | Live system metrics & Recharts charts |
| `/rag` | RAG knowledge base ingestion |

## Features

- **Keep-alive**: Pings HF Space `/health` every 4 minutes to prevent free-tier sleep
- **WebSocket voice**: Real-time STT + TTS via `/ws/voice`
- **Streaming responses**: Text streams word-by-word from `/orchestrate`
- **Live charts**: Request volume, latency, agent distribution, token usage (Recharts)
- **RAG ingestion**: Drag-and-drop with 5-step animated pipeline progress
