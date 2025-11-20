import React from 'react';
import { useAuth } from '../context/AuthContext';
import { Navigate, Outlet } from 'react-router-dom';

function ProtectedRoute() {
    const { token } = useAuth();

    if (!token) {
        // If there is no token, redirect to the /login page
        return <Navigate to="/login" replace />;
    }

    // If there is a token, show the nested child routes (your app)
    return <Outlet />;
}

export default ProtectedRoute;