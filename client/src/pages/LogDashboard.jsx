import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Activity, List, BarChart2, Search, TrendingUp } from 'lucide-react';
import PatternChainViewer from '../components/PatternChainViewer';

function LogDashboard() {
    const [logs, setLogs] = useState([]);
    const [appCountData, setAppCountData] = useState([]);
    const [patternData, setPatternData] = useState([]);
    const [patternChainData, setPatternChainData] = useState([]);
    const [isLoadingLogs, setIsLoadingLogs] = useState(true);
    const [isLoadingPatterns, setIsLoadingPatterns] = useState(true);
    const [error, setError] = useState(null); 
    const [selectedPattern, setSelectedPattern] = useState(null);
    
    // --- State for Search ---
    const [searchQuery, setSearchQuery] = useState('');

    // Helper to get auth headers
    const getAuthHeaders = () => {
        const token = localStorage.getItem('token');
        return token ? { Authorization: `Bearer ${token}` } : {};
    };

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const API_URL = window.location.hostname === 'localhost' 
                    ? 'http://localhost:8080' 
                    : '';
                const response = await axios.get(`${API_URL}/api/logs?limit=1000`, {
                    headers: getAuthHeaders()
                });
                
                // Handle new pagination format
                const logsData = response.data.logs || response.data;
                setLogs(logsData);
                
                const appCounts = {};
                logsData.forEach(log => {
                    let appName = log.appDisplayName;
                    if (appName === "MyUSF (OASIS)") appName = "MyUSF";
                    appCounts[appName] = (appCounts[appName] || 0) + 1;
                });
                // Sort by count and take top 10
                const chartData = Object.keys(appCounts)
                    .map(appName => ({
                        name: appName,
                        logins: appCounts[appName]
                    }))
                    .sort((a, b) => b.logins - a.logins)
                    .slice(0, 10);
                setAppCountData(chartData);
                
                // Show info about total logs if paginated
                if (response.data.total) {
                    console.log(`Showing ${logsData.length} of ${response.data.total} total logs`);
                }
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
                const response = await axios.get(`${API_URL}/api/patterns`, {
                    headers: getAuthHeaders()
                });
                setPatternData(response.data);
                
                // Fetch pattern chains
                const chainResponse = await axios.get(`${API_URL}/api/pattern-chains`, {
                    headers: getAuthHeaders()
                });
                setPatternChainData(chainResponse.data);
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

    const handlePatternClick = (data) => {
        // For ML behavioral patterns - clicking the button directly
        setSelectedPattern(data);
    };

    const handleCloseChainViewer = () => {
        setSelectedPattern(null);
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
            {/* Pattern Chain Viewer Modal */}
            <PatternChainViewer 
                selectedPattern={selectedPattern}
                chainData={patternChainData}
                onClose={handleCloseChainViewer}
            />

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
                        <h2 className="text-lg font-bold text-gray-800">Top 10 Applications by Login Count</h2>
                    </div>
                    
                    <div className="h-[350px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart 
                                data={appCountData} 
                                margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis 
                                    dataKey="name" 
                                    stroke="#6b7280" 
                                    fontSize={11} 
                                    tickLine={false} 
                                    axisLine={false}
                                    angle={-45}
                                    textAnchor="end"
                                    height={80}
                                />
                                <YAxis allowDecimals={false} stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
                                <Tooltip 
                                    contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                    itemStyle={{ color: '#006747' }}
                                    labelStyle={{ fontWeight: 'bold' }}
                                />
                                <Bar dataKey="logins" fill="#006747" radius={[4, 4, 0, 0]} /> 
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Chart 2: Common User Journeys (App Sequences) */}
                <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
                     <div className="flex items-center gap-2 mb-6 border-b border-gray-100 pb-4">
                        <Activity className="w-5 h-5 text-[#CDB87D]" />
                        <h2 className="text-lg font-bold text-gray-800">Common User Journeys</h2>
                        <span className="ml-auto text-xs text-gray-500">Hover to see full sequence</span>
                    </div>

                    {patternData.length > 0 ? (
                        <div className="h-[350px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={patternData.slice(0, 8).map((item, idx) => ({
                                        ...item,
                                        displayName: `Pattern ${idx + 1}`
                                    }))}
                                    layout="vertical" 
                                    margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f0f0f0" />
                                    <XAxis type="number" allowDecimals={false} stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
                                    <YAxis 
                                        type="category" 
                                        dataKey="displayName" 
                                        stroke="#6b7280" 
                                        fontSize={12}
                                        width={70}
                                        tickLine={false} 
                                        axisLine={false}
                                    />
                                    <Tooltip 
                                        contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)', maxWidth: '400px' }}
                                        itemStyle={{ color: '#CDB87D' }}
                                        labelStyle={{ fontWeight: 'bold', fontSize: '12px', marginBottom: '4px' }}
                                        formatter={(value, name, props) => [
                                            `${value} occurrences`,
                                            props.payload.name.replace(/Microsoft 365 Sign-in →\s*/gi, '')
                                        ]}
                                    />
                                    <Bar dataKey="count" fill="#CDB87D" radius={[0, 4, 4, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div className="h-[350px] flex items-center justify-center text-gray-400">
                            <p>No journey patterns detected</p>
                        </div>
                    )}
                </div>
            </div>

            {/* --- NEW: ML Behavioral Patterns Section --- */}
            <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
                <div className="flex items-center gap-2 mb-4 border-b border-gray-100 pb-4">
                    <TrendingUp className="w-5 h-5 text-purple-600" />
                    <h2 className="text-lg font-bold text-gray-800">ML-Learned Behavioral Patterns</h2>
                    <div className="ml-auto flex items-center gap-2 text-xs text-gray-500">
                        <span>Click a pattern to see flow transitions</span>
                    </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {patternChainData.map((item) => (
                        <button
                            key={item.pattern}
                            onClick={() => setSelectedPattern(item.pattern)}
                            className="px-4 py-3 bg-gradient-to-br from-purple-50 to-blue-50 hover:from-purple-100 hover:to-blue-100 border-2 border-purple-200 hover:border-purple-400 rounded-lg font-bold text-purple-800 transition-all hover:shadow-md text-sm"
                        >
                            {item.pattern}
                        </button>
                    ))}
                </div>
                <p className="text-xs text-center text-gray-500 mt-4">
                    8 behavioral patterns learned from training data
                </p>
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