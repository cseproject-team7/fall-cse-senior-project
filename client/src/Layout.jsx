import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

function Layout() {
  return (
    <div className="app-layout">
      {/* The sidebar is now part of the main layout */}
      <Sidebar />

      {/* The "main-content" div will get padding to make 
          room for the sidebar. We'll add this style next. */}
      <main className="main-content">
        {/* Outlet is the placeholder where your pages
            (App.jsx and LogDashboard.jsx) will be rendered */}
        <Outlet />
      </main>
    </div>
  );
}

export default Layout;