import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import { normalizePredictions } from './utils/predictionFormatter';

function App() {
  const [personas, setPersonas] = useState([]);
  const [selectedPersona, setSelectedPersona] = useState('');
  const [logs, setLogs] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [logsLimit, setLogsLimit] = useState('all');
  const [totalLogs, setTotalLogs] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_BASE = window.location.hostname !== 'localhost' 
    ? window.location.origin 
    : '';

  const limitOptions = ['all', '5', '10', '20', '50'];

  const fetchPersonas = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/logs/personas`);
      const data = await response.json();
      if (data.success) {
        setPersonas(data.personas);
        setSelectedPersona((current) => current || data.personas[0] || '');
      }
    } catch (err) {
      console.error('Error fetching personas:', err);
    }
  }, [API_BASE]);

  const generatePredictions = useCallback(async (logData) => {
    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: logData })
      });

      const data = await response.json();
      
      if (data.success) {
        const normalized = data.prediction
          ? normalizePredictions(data.prediction)
          : [];
        setPredictions(normalized);
      } else {
        setError(data.error);
        setPredictions([]);
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message);
      setPredictions([]);
    }
  }, [API_BASE]);

  const fetchLogs = useCallback(async (persona, limitOverride) => {
    const effectiveLimit = limitOverride ?? logsLimit;
    const limitQuery = effectiveLimit === 'all'
      ? 'all'
      : Number(effectiveLimit);

    setLoading(true);
    setError(null);
    setPredictions([]);
    try {
      const response = await fetch(
        `${API_BASE}/api/logs/${persona}?limit=${limitQuery}`
      );
      const data = await response.json();
      
      if (data.success) {
        setLogs(data.logs);
        const totalFromResponse = typeof data.total === 'number'
          ? data.total
          : typeof data.count === 'number'
            ? data.count
            : Array.isArray(data.logs)
              ? data.logs.length
              : 0;
        setTotalLogs(totalFromResponse);
        // Auto-generate predictions for the logs
        if (data.logs.length > 0) {
          await generatePredictions(data.logs);
        } else {
          setPredictions([]);
        }
      } else {
        setError(data.error);
        setLogs([]);
        setPredictions([]);
        setTotalLogs(0);
      }
    } catch (err) {
      console.error('Error fetching logs:', err);
      setError(err.message);
      setLogs([]);
      setPredictions([]);
      setTotalLogs(0);
    } finally {
      setLoading(false);
    }
  }, [API_BASE, generatePredictions, logsLimit]);

  // Fetch available personas on mount
  useEffect(() => {
    fetchPersonas();
  }, [fetchPersonas]);

  // Fetch logs when persona or limit changes
  useEffect(() => {
    if (selectedPersona) {
      fetchLogs(selectedPersona);
    }
  }, [selectedPersona, fetchLogs]);

  const formatHour = (hour) => {
    const h = hour % 12 || 12;
    const ampm = hour < 12 ? 'AM' : 'PM';
    return `${h}:00 ${ampm}`;
  };

  const formatWeekday = (day) => {
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    return days[day] || day;
  };

  const handleLimitChange = (event) => {
    setLogsLimit(event.target.value);
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
              <div className="section-actions">
                <span className="count-badge">
                  {totalLogs} logs
                </span>
                <div className="logs-limit-control">
                  <label htmlFor="logs-limit">Show</label>
                  <select
                    id="logs-limit"
                    value={logsLimit}
                    onChange={handleLimitChange}
                    disabled={loading}
                  >
                    {limitOptions.map((option) => (
                      <option key={option} value={option}>
                        {option === 'all' ? 'All' : option}
                      </option>
                    ))}
                  </select>
                  <span className="logs-limit-suffix">
                    {logsLimit === 'all' ? 'logs (all)' : 'logs'}
                  </span>
                </div>
              </div>
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
                  {predictions.map((pred) => (
                    <div key={pred.id} className="prediction-item">
                      <div className="prediction-header">
                        <span className="prediction-app">{pred.title}</span>
                        {pred.time && (
                          <span className="prediction-time">{pred.time}</span>
                        )}
                      </div>

                      {(pred.subtitle || pred.meta.length > 0) && (
                        <div className="prediction-details">
                          {pred.subtitle && (
                            <span className="prediction-subtitle">
                              {pred.subtitle}
                            </span>
                          )}
                          {pred.meta.length > 0 && (
                            <div className="prediction-meta">
                              {pred.meta.map((item, metaIndex) => (
                                <div
                                  key={`${pred.id}-meta-${metaIndex}`}
                                  className="prediction-meta-item"
                                >
                                  <span className="meta-label">
                                    {item.label}
                                  </span>
                                  <span className="meta-value">
                                    {item.value}
                                  </span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      <div className="prediction-raw">
                        <details>
                          <summary>View Raw Response</summary>
                          <pre>{pred.rawText}</pre>
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
