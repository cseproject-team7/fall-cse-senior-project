import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate, Navigate } from 'react-router-dom';
// We use Recharts to create a "Mock Analytics" visual for the design
import { AreaChart, Area, ResponsiveContainer } from 'recharts'; 
import { Lock, Mail, ArrowRight, Activity, ShieldCheck, AlertCircle } from 'lucide-react';

// Mock data for the visual flair on the left side
const mockChartData = [
  { value: 10 }, { value: 35 }, { value: 15 }, { value: 40 }, { value: 25 }, { value: 60 }, { value: 50 }
];

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [localError, setLocalError] = useState('');
    
    // Get auth methods from our context
    const { login, loading, token, error: authError } = useAuth();
    const navigate = useNavigate();

    // If already logged in, kick them to the dashboard
    if (token) {
        return <Navigate to="/" replace />;
    }

    // --- Strict USF Email Validation ---
    const validateEmail = (email) => {
        return email.toLowerCase().endsWith('@usf.edu');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLocalError(''); // Clear previous errors

        // 1. Validation Checks
        if (!email) {
            setLocalError('Email is required.');
            return;
        }
        
        if (!validateEmail(email)) {
            setLocalError('Access Restricted: Please use your official @usf.edu email address.');
            return;
        }

        if (!password) {
            setLocalError('Please enter your password.');
            return;
        }

        // 2. Attempt Login
        const success = await login(email, password);
        if (success) {
            navigate('/');
        }
    };

    return (
        <div className="min-h-screen flex w-full font-sans">
            
            {/* --- LEFT SIDE: Branding & Visuals --- */}
            <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden flex-col justify-between p-12"
                 style={{ backgroundColor: '#006747', color: '#EDEBD1' }}>
                
                {/* Subtle Grid Background Pattern */}
                <div className="absolute inset-0 opacity-10" 
                     style={{ backgroundImage: 'radial-gradient(#EDEBD1 1px, transparent 1px)', backgroundSize: '24px 24px' }}>
                </div>

                {/* Brand Header */}
                <div className="relative z-10">
                    <div className="flex items-center gap-2 mb-3 opacity-90">
                        <ShieldCheck className="w-6 h-6" />
                        <span className="text-sm font-bold tracking-wider uppercase">Secure Environment</span>
                    </div>
                    <h1 className="text-5xl font-bold leading-tight mb-4">
                        Authentication <br/> Analytics
                    </h1>
                    <p className="text-lg opacity-80 max-w-md leading-relaxed">
                        Advanced pattern recognition and session monitoring for University of South Florida student systems.
                    </p>
                </div>

                {/* Aesthetic Mock Chart (Reinforces the "Data" theme) */}
                <div className="relative z-10 w-full max-w-md bg-[#005238] rounded-xl p-5 border border-[#007a55] shadow-2xl backdrop-blur-sm bg-opacity-50">
                     <div className="flex justify-between items-center mb-4">
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4" />
                            <span className="text-sm font-medium">Live Traffic Monitoring</span>
                        </div>
                        <span className="text-xs px-2 py-1 bg-[#006747] rounded-full font-mono text-[#EDEBD1]">REAL-TIME</span>
                     </div>
                     <div className="h-32 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={mockChartData}>
                                <defs>
                                    <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#EDEBD1" stopOpacity={0.3}/>
                                        <stop offset="95%" stopColor="#EDEBD1" stopOpacity={0}/>
                                    </linearGradient>
                                </defs>
                                <Area type="monotone" dataKey="value" stroke="#EDEBD1" strokeWidth={2} fillOpacity={1} fill="url(#colorVal)" />
                            </AreaChart>
                        </ResponsiveContainer>
                     </div>
                </div>

                <div className="relative z-10 text-xs opacity-50 font-mono">
                    © 2025 USF Senior Project • Team 7 • Authorized Personnel Only
                </div>
            </div>

            {/* --- RIGHT SIDE: Login Form --- */}
            <div className="w-full lg:w-1/2 flex items-center justify-center p-8" 
                 style={{ backgroundColor: '#EDEBD1' }}>
                
                <div className="w-full max-w-md bg-white rounded-2xl shadow-xl p-8 md:p-10 border border-[#d4d1b4]">
                    <div className="mb-8 text-center lg:text-left">
                        <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-[#e6f0ec] mb-4">
                            <Lock className="w-6 h-6 text-[#006747]" />
                        </div>
                        <h2 className="text-2xl font-bold text-gray-900">Welcome Back</h2>
                        <p className="text-gray-600 mt-1 text-sm">Please verify your identity to continue.</p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-5">
                        
                        {/* Error Alert Box */}
                        {(localError || authError) && (
                            <div className="flex items-start gap-3 p-4 text-sm text-red-800 bg-red-50 border border-red-100 rounded-lg animate-pulse">
                                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                                <div>
                                    <span className="font-semibold block mb-1">Authentication Error</span>
                                    {localError || authError}
                                </div>
                            </div>
                        )}

                        {/* Email Input */}
                        <div className="space-y-1">
                            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                                Institutional Email
                            </label>
                            <div className="relative">
                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                    <Mail className="h-5 w-5 text-gray-400" />
                                </div>
                                <input
                                    id="email"
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#006747] focus:border-transparent transition-all text-gray-900 placeholder-gray-400 bg-gray-50"
                                    placeholder="netid@usf.edu"
                                    autoComplete="email"
                                />
                            </div>
                        </div>

                        {/* Password Input */}
                        <div className="space-y-1">
                            <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                                Password
                            </label>
                            <div className="relative">
                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                    <Lock className="h-5 w-5 text-gray-400" />
                                </div>
                                <input
                                    id="password"
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#006747] focus:border-transparent transition-all text-gray-900 placeholder-gray-400 bg-gray-50"
                                    placeholder="••••••••"
                                    autoComplete="current-password"
                                />
                            </div>
                        </div>

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full flex items-center justify-center gap-2 py-3 px-4 border border-transparent rounded-lg shadow-md text-sm font-bold text-[#EDEBD1] bg-[#006747] hover:bg-[#005238] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#006747] disabled:opacity-70 disabled:cursor-not-allowed transition-all transform active:scale-[0.99] mt-2"
                        >
                            {loading ? (
                                <>
                                    <div className="w-4 h-4 border-2 border-[#EDEBD1] border-t-transparent rounded-full animate-spin"></div>
                                    <span>Verifying Credentials...</span>
                                </>
                            ) : (
                                <>
                                    <span>Sign In to Dashboard</span>
                                    <ArrowRight className="w-4 h-4" />
                                </>
                            )}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}