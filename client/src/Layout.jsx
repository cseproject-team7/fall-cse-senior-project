import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

function Layout() {
  return (
    // Updated background color to match index.css
    <div className="min-h-screen bg-[#f3f4f6]">
      {/* The Sidebar component is fixed to the left */}
      <Sidebar />

      {/* Main Content Area */}
      <main className="ml-64 min-h-screen">
        {/* This outlet renders your Dashboard or Predictions page */}
        <Outlet />
      </main>
    </div>
  );
}

export default Layout;