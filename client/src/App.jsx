import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [personas, setPersonas] = useState([]);
  const [selectedPersona, setSelectedPersona] = useState('');
  const [logs, setLogs] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_BASE = window.location.hostname !== 'localhost' 
    ? window.location.origin 
    : '';

  // Fetch available personas on mount
  useEffect(() => {
    fetchPersonas();
  }, [fetchPersonas]);

  // Fetch logs when persona changes
  useEffect(() => {
    if (selectedPersona) {
      fetchLogs(selectedPersona);
    }
  }, [fetchLogs, selectedPersona]);

  const fetchPersonas = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/logs/personas`);
      const data = await response.json();
      if (data.success) {
        setPersonas(data.personas);
        if (data.personas.length > 0) {
          setSelectedPersona(data.personas[0]);
        }
      }
    } catch (err) {
      console.error('Error fetching personas:', err);
    }
  };

  const fetchLogs = async (persona) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/logs/${persona}`);
      const data = await response.json();
      
      if (data.success) {
        setLogs(data.logs);
        // Auto-generate predictions for the logs
        if (data.logs.length > 0) {
          await generatePredictions(data.logs);
        }
      } else {
        setError(data.error);
      }
    } catch (err) {
      console.error('Error fetching logs:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const generatePredictions = async (logData) => {
    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: logData })
      });

      const data = await response.json();
      
      if (data.success) {
        setPredictions(data.prediction ? [data.prediction] : []);
      } else {
        setError(data.error);
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message);
    }
  };

  const formatHour = (hour) => {
    const h = hour % 12 || 12;
    const ampm = hour < 12 ? 'AM' : 'PM';
    return `${h}:00 ${ampm}`;
  };

  const formatWeekday = (day) => {
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    return days[day] || day;
  };

  return (
    <div className="App">
      <div className="container">
        {/* Header with Persona Selector */}
        <header className="header">
          <h1>Authentication Analytics</h1>
          <p className="subtitle">USF Authentication Prediction System</p>
          
          <div className="persona-selector">
            <label htmlFor="persona-select">Select Persona:</label>
            <select 
              id="persona-select"
              value={selectedPersona} 
              onChange={(e) => setSelectedPersona(e.target.value)}
              disabled={loading}
            >
              {personas.map(persona => (
                <option key={persona} value={persona}>
                  {persona.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>
        </header>

        {error && (
          <div className="error-box">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {/* Side-by-side comparison */}
        <div className="comparison-container">
          {/* Input Logs Section */}
          <div className="section">
            <div className="section-header">
              <h2>📊 Input Activity Logs</h2>
              <span className="count-badge">{logs.length} logs</span>
            </div>
            <div className="card logs-card">
              {loading ? (
                <div className="loading">Loading logs...</div>
              ) : logs.length === 0 ? (
                <div className="empty-state">No logs available for this persona</div>
              ) : (
                <div className="logs-list">
                  {logs.map((log, index) => (
                    <div key={index} className="log-item">
                      <div className="log-header">
                        <span className="log-app">{log.appDisplayName}</span>
                        <span className="log-time">{formatHour(log.hour)}</span>
                      </div>
                      <div className="log-details">
                        <span className="log-day">{formatWeekday(log.weekday)}</span>
                        {log.timestamp && (
                          <span className="log-timestamp">
                            {new Date(log.timestamp).toLocaleString()}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Predictions Section */}
          <div className="section">
            <div className="section-header">
              <h2>🎯 ML Predictions</h2>
              <span className="count-badge">{predictions.length} predictions</span>
            </div>
            <div className="card predictions-card">
              {loading ? (
                <div className="loading">Generating predictions...</div>
              ) : predictions.length === 0 ? (
                <div className="empty-state">No predictions available</div>
              ) : (
                <div className="predictions-list">
                  {predictions.map((pred, index) => (
                    <div key={index} className="prediction-item">
                      <div className="prediction-label">Predicted Next App:</div>
                      <div className="prediction-value">{pred.pred_app}</div>
                      <div className="prediction-raw">
                        <details>
                          <summary>View Raw Response</summary>
                          <pre>{JSON.stringify(pred, null, 2)}</pre>
                        </details>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <footer className="footer">
          <p>University of South Florida - Team 7</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
