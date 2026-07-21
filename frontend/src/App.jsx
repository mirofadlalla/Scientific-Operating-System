import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import NavBar from './components/NavBar';
import ChatPage from './pages/ChatPage';
import MonitorPage from './pages/MonitorPage';
import RagPage from './pages/RagPage';
import { BACKEND_URL } from './config';
import './index.css';

// Ping HF Space every 4 minutes to prevent free-tier sleep
function useKeepAlive() {
  useEffect(() => {
    const ping = () =>
      fetch(`${BACKEND_URL}/health`, { mode: 'cors' }).catch(() => {});
    ping(); // immediate ping on mount
    const id = setInterval(ping, 4 * 60 * 1000);
    return () => clearInterval(id);
  }, []);
}

export default function App() {
  useKeepAlive();
  return (
    <BrowserRouter>
      <div className="page-layout">
        <NavBar />
        <div className="page-content">
          <Routes>
            <Route path="/"        element={<ChatPage />} />
            <Route path="/monitor" element={<MonitorPage />} />
            <Route path="/rag"     element={<RagPage />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}
