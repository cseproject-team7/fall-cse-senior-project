import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from './context/AuthContext'; // 1. Import useAuth
import './Sidebar.css';

function Sidebar() {
  const { logout } = useAuth(); // 2. Get the logout function

  return (
    <nav className="sidebar">
      <h2>Senior Project</h2>
      <ul>
        <li>
          <NavLink to="/" end>
            Predictions
          </NavLink>
        </li>
        <li>
          <NavLink to="/dashboard">
            Dashboard
          </NavLink>
        </li>
      </ul>
      
      {/* 3. --- ADD THIS LOGOUT BUTTON --- */}
      <div className="sidebar-footer">
        <button onClick={logout} className="logout-btn">
          Sign Out
        </button>
      </div>
      {/* --- END OF NEW BUTTON --- */}

    </nav>
  );
}

export default Sidebar;