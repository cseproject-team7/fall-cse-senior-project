import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Activity, List, BarChart2, Search } from 'lucide-react'; // Added 'Search' icon

function LogDashboard() {
    const [logs, setLogs] = useState([]);
    const [appCountData, setAppCountData] = useState([]);
    const [patternData, setPatternData] = useState([]);
    const [isLoadingLogs, setIsLoadingLogs] = useState(true);
    const [isLoadingPatterns, setIsLoadingPatterns] = useState(true);
    const [error, setError] = useState(null); 
    
    // --- NEW: State for Search ---
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const API_URL = window.location.hostname === 'localhost' 
                    ? 'http://localhost:8080' 
                    : '';
                const response = await axios.get(`${API_URL}/api/logs`); 
                setLogs(response.data);
                
                const appCounts = {};
                response.data.forEach(log => {
                    let appName = log.appDisplayName;
                    if (appName === "MyUSF (OASIS)") appName = "MyUSF";
                    appCounts[appName] = (appCounts[appName] || 0) + 1;
                });
                const chartData = Object.keys(appCounts).map(appName => ({
                    name: appName,
                    logins: appCounts[appName]
                }));
                setAppCountData(chartData);
            } catch (err) {
                setError(err.message);
                console.error("Failed to fetch logs:", err);
            } finally {
                setIsLoadingLogs(false);
            }
        };
        const fetchPatterns = async () => {
            try {
                const API_URL = window.location.hostname === 'localhost' 
                    ? 'http://localhost:8080' 
                    : '';
                const response = await axios.get(`${API_URL}/api/patterns`);
                setPatternData(response.data);
            } catch (err) {
                setError(err.message);
                console.error("Failed to fetch patterns:", err);
            } finally {
                setIsLoadingPatterns(false);
            }
        };
        fetchLogs();
        fetchPatterns();
    }, []); 

    const truncate = (str, n) => {
        if (!str) return "";
        return str.length > n ? str.substr(0, n - 1) + "..." : str;
    };

    // --- NEW: Filtering Logic ---
    const filteredLogs = logs.filter(log => {
        const query = searchQuery.toLowerCase();
        return (
            log.userPrincipalName.toLowerCase().includes(query) ||
            log.appDisplayName.toLowerCase().includes(query)
        );
    });

    // --- Loading State ---
    if (isLoadingLogs || isLoadingPatterns) {
        return (
            <div className="flex flex-col justify-center items-center h-[80vh]">
                <div className="loader"></div> 
                <p className="text-gray-600 mt-6 font-medium text-lg">Loading Dashboard Data...</p>
            </div>
        );
    }

    // --- Error State ---
    if (error) {
        return (
            <div className="p-8 text-center">
                <div className="bg-red-50 text-red-600 p-6 rounded-xl border border-red-200 inline-block">
                    <h2 className="text-xl font-bold mb-2">Error Loading Data</h2>
                    <p>{error}</p>
                </div>
            </div>
        );
    }

    return (
        <div className="p-8 space-y-8">
            {/* Page Header */}
            <header>
                <h1 className="text-3xl font-bold text-gray-900">Dashboard Overview</h1>
                <p className="text-gray-500 mt-1">Real-time analysis of authentication logs and user patterns.</p>
            </header>

            {/* --- Charts Grid (2 Columns) --- */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                
                {/* Chart 1: Logins Per App */}
                <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
                    <div className="flex items-center gap-2 mb-6 border-b border-gray-100 pb-4">
                        <BarChart2 className="w-5 h-5 text-[#006747]" />
                        <h2 className="text-lg font-bold text-gray-800">Logins Per Application</h2>
                    </div>
                    
                    <div className="h-[350px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={appCountData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis dataKey="name" stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis allowDecimals={false} stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
                                <Tooltip 
                                    contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                    itemStyle={{ color: '#006747' }}
                                />
                                <Bar dataKey="logins" fill="#006747" radius={[4, 4, 0, 0]} /> 
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Chart 2: Patterns */}
                <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
                     <div className="flex items-center gap-2 mb-6 border-b border-gray-100 pb-4">
                        <Activity className="w-5 h-5 text-[#CDB87D]" />
                        <h2 className="text-lg font-bold text-gray-800">Common User Journeys</h2>
                    </div>

                    <div className="h-[350px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                                data={patternData}
                                layout="vertical" 
                                margin={{ top: 5, right: 30, left: 40, bottom: 5 }} 
                            >
                                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f0f0f0" />
                                <XAxis type="number" allowDecimals={false} stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis 
                                    type="category" 
                                    dataKey="name" 
                                    stroke="#6b7280" 
                                    fontSize={11}
                                    width={150}
                                    tickFormatter={(tick) => truncate(tick, 25)} 
                                    tickLine={false} 
                                    axisLine={false}
                                />
                                <Tooltip 
                                    contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                    itemStyle={{ color: '#CDB87D' }}
                                />
                                <Bar dataKey="count" fill="#CDB87D" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* --- Table Section with Search --- */}
            <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
                
                {/* MODIFIED HEADER: Now includes Search Bar */}
                <div className="p-6 border-b border-gray-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <List className="w-5 h-5 text-gray-500" />
                        <h2 className="text-lg font-bold text-gray-800">Recent Activity Logs</h2>
                    </div>
                    
                    {/* Search Input */}
                    <div className="relative">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Search className="h-4 w-4 text-gray-400" />
                        </div>
                        <input
                            type="text"
                            placeholder="Search user or app..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="pl-10 pr-4 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#006747] focus:border-transparent w-full sm:w-64 transition-all"
                        />
                    </div>
                </div>
                
                <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="text-xs text-gray-700 uppercase bg-gray-50 sticky top-0 z-10">
                            <tr>
                                <th className="px-6 py-3 font-bold tracking-wider">User</th>
                                <th className="px-6 py-3 font-bold tracking-wider">Application</th>
                                <th className="px-6 py-3 font-bold tracking-wider">Date</th>
                                <th className="px-6 py-3 font-bold tracking-wider">Time</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                            {/* Render filteredLogs instead of logs */}
                            {filteredLogs.length > 0 ? (
                                filteredLogs.map((log, index) => (
                                    <tr key={index} className="bg-white hover:bg-[#fdfcf6] transition-colors">
                                        <td className="px-6 py-4 font-medium text-gray-900">{log.userPrincipalName}</td>
                                        <td className="px-6 py-4 text-gray-600">
                                            <span className="px-2 py-1 bg-green-50 text-green-700 rounded-full text-xs font-semibold">
                                                {log.appDisplayName}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-gray-500">{log.date}</td>
                                        <td className="px-6 py-4 text-gray-500 font-mono">{log.time}</td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan="4" className="px-6 py-8 text-center text-gray-500 italic">
                                        No logs found matching "{searchQuery}"
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

        </div>
    );
}

export default LogDashboard;