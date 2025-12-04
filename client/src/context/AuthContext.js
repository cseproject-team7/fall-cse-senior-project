import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

// Create the context
const AuthContext = createContext();

// Create the AuthProvider component
export const AuthProvider = ({ children }) => {
    const [token, setToken] = useState(localStorage.getItem('token') || null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        // If we have a token, set it as the default auth header for all axios requests
        if (token) {
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            localStorage.setItem('token', token);
        } else {
            delete axios.defaults.headers.common['Authorization'];
            localStorage.removeItem('token');
        }

        // Setup axios interceptor to handle token expiration
        const interceptor = axios.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.response?.status === 401 && error.response?.data?.message?.includes('expired')) {
                    // Token expired, logout user
                    setToken(null);
                    window.location.href = '/login';
                }
                return Promise.reject(error);
            }
        );

        // Cleanup interceptor on unmount
        return () => {
            axios.interceptors.response.eject(interceptor);
        };
    }, [token]);

    // Login function
    const login = async (email, password) => {
        setLoading(true);
        setError(null);
        try {
            const API_URL = window.location.hostname === 'localhost' 
                ? 'http://localhost:8080/api/auth/login' 
                : '/api/auth/login';
            const response = await axios.post(API_URL, {
                email,
                password,
            });

            if (response.data.success) {
                setToken(response.data.token);
                setLoading(false);
                return true; // Success
            }
        } catch (err) {
            console.error('Login failed:', err);
            setError(err.response?.data?.message || 'Login failed. Please try again.');
            setLoading(false);
            return false; // Failure
        }
    };

    // Logout function
    const logout = () => {
        setToken(null);
    };

    return (
        <AuthContext.Provider value={{ token, login, logout, loading, error }}>
            {children}
        </AuthContext.Provider>
    );
};

// Custom hook to use the auth context
export const useAuth = () => {
    return useContext(AuthContext);
};