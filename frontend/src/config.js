// Backend configuration
// Set VITE_BACKEND_URL (e.g. in frontend/.env.local or the Vercel project settings)
// to point at another backend, such as http://localhost:7860 for local development.
// Falls back to the deployed Hugging Face Space.
const DEFAULT_BACKEND_URL = "https://omarfadlallah-scientific-operating-system.hf.space";

export const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || DEFAULT_BACKEND_URL).replace(/\/+$/, "");

// All versioned API calls go through /api/v1
export const API_BASE = `${BACKEND_URL}/api/v1`;

// WebSocket voice channel (https -> wss, http -> ws)
export const WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/api/v1/ws/voice`;
