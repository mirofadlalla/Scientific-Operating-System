// Backend configuration
// Update BACKEND_URL to match your deployed Hugging Face Space URL.
export const BACKEND_URL = "https://omarfadlallah-scientific-operating-system.hf.space";

// All versioned API calls go through /api/v1
export const API_BASE = `${BACKEND_URL}/api/v1`;

// WebSocket voice channel
export const WS_URL = `${BACKEND_URL.replace("https://", "wss://")}/api/v1/ws/voice`;
