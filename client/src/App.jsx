import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import Layout from './components/Layout';
import LogDashboard from './pages/LogDashboard';
import PredictionsPage from './pages/PredictionsPage';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Route - Login */}
          <Route path="/login" element={<LoginPage />} />
          
          {/* Protected Routes - Require Authentication */}
          <Route element={<ProtectedRoute />}>
            <Route element={<Layout />}>
              <Route path="/" element={<PredictionsPage />} />
              <Route path="/dashboard" element={<LogDashboard />} />
            </Route>
          </Route>

          {/* Catch all - redirect to predictions */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
