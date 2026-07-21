import { NavLink } from 'react-router-dom';

export default function NavBar() {
  return (
    <nav className="navbar">
      <NavLink to="/" className="nav-logo" style={{ textDecoration: 'none' }}>
        <div className="nav-logo-icon">🧪</div>
        <div className="nav-logo-text">
          <h1>AI-lixir Scientific OS</h1>
          <p>v2.0 · Drug Discovery Platform</p>
        </div>
      </NavLink>

      <div className="nav-links">
        <NavLink
          to="/"
          end
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          💬 Chat
        </NavLink>
        <NavLink
          to="/monitor"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          📊 Monitor
        </NavLink>
        <NavLink
          to="/rag"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          🗄️ Knowledge Base
        </NavLink>
      </div>

      <div className="nav-right">
        <div className="badge">
          <span className="dot"></span>
          API Connected
        </div>
      </div>
    </nav>
  );
}
