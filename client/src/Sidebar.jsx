import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import { LayoutDashboard, BrainCircuit, LogOut, Shield } from 'lucide-react';

function Sidebar() {
  const { logout } = useAuth();

  return (
    <nav className="fixed top-0 left-0 h-screen w-64 bg-[#006747] text-white flex flex-col shadow-2xl z-50">
      
      {/* Header / Logo Area */}
      <div className="p-6 border-b border-[#005238]">
        <div className="flex items-center gap-3 mb-1">
            <Shield className="w-8 h-8 text-[#EDEBD1]" />
            <span className="font-bold text-lg tracking-wide">AuthAnalytics</span>
        </div>
        <p className="text-xs text-[#EDEBD1] opacity-80 pl-11">USF Senior Project</p>
      </div>

      {/* Navigation Links */}
      <ul className="flex-1 py-6 px-3 space-y-2">
        <li>
          <NavLink 
            to="/" 
            end
            className={({ isActive }) => 
              `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                isActive 
                  ? 'bg-[#EDEBD1] text-[#006747] font-bold shadow-md' 
                  : 'text-white hover:bg-[#005238] hover:pl-5'
              }`
            }
          >
            <BrainCircuit className="w-5 h-5" />
            Predictions
          </NavLink>
        </li>
        <li>
          <NavLink 
            to="/dashboard"
            className={({ isActive }) => 
              `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                isActive 
                  ? 'bg-[#EDEBD1] text-[#006747] font-bold shadow-md' 
                  : 'text-white hover:bg-[#005238] hover:pl-5'
              }`
            }
          >
            <LayoutDashboard className="w-5 h-5" />
            Dashboard
          </NavLink>
        </li>
      </ul>
      
      {/* Footer / Logout */}
      <div className="p-4 border-t border-[#005238]">
        <button 
          onClick={logout} 
          className="flex items-center gap-3 w-full px-4 py-3 text-sm font-medium text-white bg-[#005238] rounded-lg hover:bg-[#c0392b] transition-colors"
        >
          <LogOut className="w-5 h-5" />
          Sign Out
        </button>
      </div>

    </nav>
  );
}

export default Sidebar;