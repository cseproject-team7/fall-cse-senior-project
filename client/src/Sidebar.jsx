import React from 'react';
// NavLink is special: it knows what page you're on and can style it
import { NavLink } from 'react-router-dom';
import './Sidebar.css'; // We'll create this next

function Sidebar() {
  return (
    <nav className="sidebar">
      <h2>Senior Project</h2>
      <ul>
        <li>
          {/* "end" tells it to only be active on the exact "/" path */}
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
    </nav>
  );
}

export default Sidebar;