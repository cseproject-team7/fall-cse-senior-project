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
    const [logs, setLogs] = useState([]); // For the table
    const [appCountData, setAppCountData] = useState([]); // For 1st chart
    const [patternData, setPatternData] = useState([]); // For 2nd chart
    
    const [isLoadingLogs, setIsLoadingLogs] = useState(true);
    const [isLoadingPatterns, setIsLoadingPatterns] = useState(true);
    
    const [error, setError] = useState(null); 

    useEffect(() => {
        // --- 1. Fetch Logs (for table and app counts) ---
        const fetchLogs = async () => {
            try {
                // Use the correct 8080 port
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

        // --- 2. Fetch App Patterns ---
        const fetchPatterns = async () => {
            try {
                // Use the correct 8080 port
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

    if (isLoadingLogs || isLoadingPatterns) {
        return <div>Loading dashboard data...</div>;
    }

    if (error) {
        return <div style={{ color: 'red', padding: '20px' }}>Error: {error}</div>;
    }

    // --- NEW: Calculate dynamic height for the horizontal chart ---
    // Give 40px height for each bar, with a 200px minimum
    const patternChartHeight = Math.max(200, patternData.length * 40);


    return (
        <div className="dashboard-container" style={{ padding: '20px' }}>
            <h1>Student Authentication Dashboard</h1>

            {/* --- CHART 1 (Same as before) --- */}
            <h2 style={{ marginTop: '40px' }}>Logins Per Application</h2>
            <ResponsiveContainer 
              width="100%" 
              height={300} 
              style={{ 
                background: '#333',
                borderRadius: '5px', 
                padding: '10px' 
              }}
            >
                <BarChart data={appCountData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#555" />
                    <XAxis dataKey="name" stroke="#f5f5f5" />
                    <YAxis allowDecimals={false} stroke="#f5f5f5" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#222', color: '#f5f5f5' }} 
                      itemStyle={{ color: '#f5f5f5' }}
                    />
                    <Legend wrapperStyle={{ color: '#f5f5f5' }} />
                    <Bar dataKey="logins" fill="#CDB87D" /> 
                </BarChart>
            </ResponsiveContainer>


            {/* --- 
              CHART 2: MODIFIED FOR HORIZONTAL & SCROLLING 
              --- 
            */}
            <h2 style={{ marginTop: '40px' }}>Common Application Patterns (Ranked)</h2>

            {/* 1. NEW: Added a wrapper div for scrolling */}
            <div 
              className="chart-scroll-container" 
              style={{ 
                height: '600px', // Set a fixed max height for the container
                overflowY: 'auto', // Add vertical scroll
                background: '#333',
                borderRadius: '5px', 
                padding: '10px' 
              }}
            >
                {/* 2. Set container height to be dynamic based on data */}
                <ResponsiveContainer width="100%" height={patternChartHeight}>
                    {/* 3. Added layout="vertical" */}
                    <BarChart 
                      data={patternData} 
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 200, bottom: 5 }} // Increased left margin
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#555" />
                        
                        {/* 4. Swapped X and Y axis */}
                        <XAxis type="number" allowDecimals={false} stroke="#f5f5f5" />
                        <YAxis 
                          dataKey="name" 
                          type="category" 
                          stroke="#f5f5f5" 
                          width={150} // Give space for long labels
                        />
                        
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#222', color: '#f5f5f5' }} 
                          itemStyle={{ color: '#f5f5f5' }}
                        />
                        <Legend wrapperStyle={{ color: '#f5f5f5' }} />
                        <Bar dataKey="count" fill="#CDB87D" />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            {/* --- END OF MODIFIED CHART 2 --- */}


            {/* --- Scrollable Table (Same as before) --- */}
            <h2 style={{ marginTop: '40px' }}>All Logs</h2>
            <div 
              className="table-scroll-container" 
              style={{ height: '500px', overflowY: 'auto', marginTop: '20px', border: '1px solid #ddd' }}
            >
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead style={{ position: 'sticky', top: 0 }}>
                        <tr style={{ background: '#f0f0f0' }}>
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

        </div>
    );
}

export default LogDashboard;