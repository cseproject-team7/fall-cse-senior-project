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

function LogDashboard() {
    const [logs, setLogs] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const response = await axios.get('http://localhost:3001/api/logs'); 
                setLogs(response.data); 
            } catch (err) {
                setError(err.message);
                console.error("Failed to fetch logs:", err);
            } finally {
                setIsLoading(false);
            }
        };

        fetchLogs();
    }, []); 

    // --- Data Processing for the Chart (Same as before) ---
    const appCounts = {};
    logs.forEach(log => {
        let appName = log.appDisplayName;
        if (appName === "MyUSF (OASIS)") {
            appName = "OASIS";
        }
        appCounts[appName] = (appCounts[appName] || 0) + 1;
    });
    const chartData = Object.keys(appCounts).map(appName => ({
        name: appName,
        logins: appCounts[appName]
    }));
    // --- End of Data Processing ---


    if (isLoading) {
        return <div>Loading log data from server...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }

    return (
        <div className="dashboard-container" style={{ padding: '20px' }}>
            <h1>Student Authentication Dashboard</h1>

            {/* --- Bar Chart Section (Same as before) --- */}
            <h2 style={{ marginTop: '40px' }}>Logins Per Application</h2>
            <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name"/>
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="logins" fill="#006747" />
                </BarChart>
            </ResponsiveContainer>
            {/* --- End of Bar Chart Section --- */}


            {/* --- MODIFIED: Scrollable Table Section --- */}
            <h2 style={{ marginTop: '40px' }}>All Logs</h2>

            {/* 1. NEW: Wrapper div to control scrolling */}
            <div 
              className="table-scroll-container" 
              style={{
                height: '500px', // <-- You can change this height
                overflowY: 'auto', // <-- This adds the vertical scrollbar
                marginTop: '20px',
                border: '1px solid #ddd' // Adds a nice border
              }}
            >
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        {/* 2. MODIFIED: Made the header row sticky */}
                        <tr style={{ 
                            background: '#f0f0f0',
                            position: 'sticky', // <-- Makes header stay
                            top: 0               // <-- Sticks it to the top
                        }}>
                            <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>User</th>
                            <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>Application</th>
                            <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>Date</th>
                            <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {logs.map((log, index) => (
                            <tr key={index}>
                                <td style={{ padding: '8px', border: '1px solid #ddd' }}>{log.userPrincipalName}</td>
                                <td style={{ padding: '8px', border: '1.5px solid #ddd' }}>{log.appDisplayName}</td>
                                <td style={{ padding: '8px', border: '1px solid #ddd' }}>{log.date}</td>
                                <td style={{ padding: '8px', border: '1px solid #ddd' }}>{log.time}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            {/* --- End of Modified Table Section --- */}

        </div>
    );
}

export default LogDashboard;