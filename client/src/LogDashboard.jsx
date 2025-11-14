import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
    BarChart, 
    Bar, 
    XAxis, 
    YAxis, 
    CartesianGrid, 
    Tooltip, 
    Legend, 
    ResponsiveContainer 
} from 'recharts';
// No new imports needed

function LogDashboard() {
    // ... (All your useState hooks are the same) ...
    const [logs, setLogs] = useState([]);
    const [appCountData, setAppCountData] = useState([]);
    const [patternData, setPatternData] = useState([]);
    const [isLoadingLogs, setIsLoadingLogs] = useState(true);
    const [isLoadingPatterns, setIsLoadingPatterns] = useState(true);
    const [error, setError] = useState(null); 

    useEffect(() => {
        // ... (Your fetchLogs and fetchPatterns functions are exactly the same) ...
        const fetchLogs = async () => {
            try {
                const response = await axios.get('http://localhost:8080/api/logs'); 
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
                const response = await axios.get('http://localhost:8080/api/patterns');
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

    // --- 1. NEW HELPER FUNCTION ---
    // This function shortens any string longer than 'n' chars
    const truncate = (str, n) => {
        if (!str) return "";
        return str.length > n ? str.substr(0, n - 1) + "..." : str;
    };
    // -----------------------------

    // --- Loading State ---
    if (isLoadingLogs || isLoadingPatterns) {
        return (
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                height: '400px', 
                marginTop: '50px',
            }}>
                <div className="loader"></div> 
                <p style={{ color: '#f5f5f5', marginTop: '20px', fontSize: '18px' }}>
                    Loading Dashboard Data...
                </p>
            </div>
        );
    }

    // --- Error State ---
    if (error) {
        return (
            <div style={{
                color: '#ff6b6b', 
                padding: '20px',
                background: '#2c3e50',
                borderRadius: '5px',
                margin: '20px',
                border: '1px solid #ff6b6b'
            }}>
                <h2>Error Loading Dashboard</h2>
                <p>{error}</p>
            </div>
        );
    }

    // --- Content (Data Loaded) ---
    return (
        <div className="dashboard-container" style={{ padding: '20px' }}>
            <h1 style={{ color: '#f5f5f5' }}>Student Authentication Dashboard</h1>

            {/* --- 
              CHART 1: MODIFIED MARGINS
            --- */}
            <h2 style={{ marginTop: '40px', color: '#f5f5f5' }}>Logins Per Application</h2>
            <ResponsiveContainer 
              width="100%" 
              height={300} 
              style={{ background: '#333', borderRadius: '5px', padding: '10px' }}
            >
                {/* 1. --- MODIFIED THIS LINE --- 
                       Increased 'left' from 20 to 30 (for Y-axis labels)
                       Increased 'bottom' from 5 to 20 (for the legend)
                */}
                <BarChart data={appCountData} margin={{ top: 5, right: 30, left: 30, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#555" />
                    <XAxis dataKey="name" stroke="#f5f5f5" />
                    <YAxis allowDecimals={false} stroke="#f5f5f5" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#222', color: '#f5f5f5', border: 'none' }} 
                      itemStyle={{ color: '#f5f5f5' }}
                    />
                    <Legend wrapperStyle={{ color: '#f5f5f5' }} />
                    <Bar dataKey="logins" fill="#CDB87D" /> 
                </BarChart>
            </ResponsiveContainer>
            {/* --- END OF MODIFIED CHART 1 --- */}


            {/* --- Chart 2: Common Application Patterns (Unchanged) --- */}
            <h2 style={{ marginTop: '40px', color: '#f5f5f5' }}>Popular App Patterns</h2>
            <div 
              style={{ 
                height: `${Math.max(patternData.length * 50, 350)}px`, // Dynamic height
                background: '#333',
                borderRadius: '5px',
                padding: '10px'
              }}
            >
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={patternData}
                        layout="vertical" 
                        margin={{ top: 20, right: 30, left: 50, bottom: 5 }} 
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#555" />
                        <XAxis type="number" allowDecimals={false} stroke="#f5f5f5" />
                        <YAxis 
                            type="category" 
                            dataKey="name" 
                            stroke="#f5f5f5" 
                            interval={0} 
                            width={200} 
                            tickFormatter={(tick) => truncate(tick, 30)} 
                        />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#222', color: '#f5f5f5', border: 'none' }} 
                          itemStyle={{ color: '#f5f5f5' }}
                        />
                        <Bar dataKey="count" fill="#CDB87D" />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            
            {/* --- Scrollable Table (Unchanged) --- */}
            <h2 style={{ marginTop: '40px', color: '#f5f5f5' }}>All Logs</h2>
            <div 
              className="table-scroll-container" 
              style={{ 
                height: '500px', 
                overflowY: 'auto', 
                marginTop: '20px', 
                border: '1px solid #555',
                backgroundColor: '#CDB87D' 
              }}
            >
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead style={{ position: 'sticky', top: 0 }}>
                        <tr style={{ background: '#333' }}> 
                            <th style={{ padding: '8px', border: '1px solid #555', textAlign: 'left', color: 'white' }}>User</th>
                            <th style={{ padding: '8px', border: '1px solid #555', textAlign: 'left', color: 'white' }}>Application</th>
                            <th style={{ padding: '8px', border: '1px solid #555', textAlign: 'left', color: 'white' }}>Date</th>
                            <th style={{ padding: '8px', border: '1px solid #555', textAlign: 'left', color: 'white' }}>Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {logs.map((log, index) => (
                            <tr key={index} style={{ color: 'black' }}> 
                                <td style={{ padding: '8px', border: '1px solid #aaa' }}>{log.userPrincipalName}</td>
                                <td style={{ padding: '8px', border: '1px solid #aaa' }}>{log.appDisplayName}</td>
                                <td style={{ padding: '8px', border: '1px solid #aaa' }}>{log.date}</td>
                                <td style={{ padding: '8px', border: '1px solid #aaa' }}>{log.time}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            {/* --- END OF MODIFIED TABLE --- */}

        </div>
    );
}

export default LogDashboard;