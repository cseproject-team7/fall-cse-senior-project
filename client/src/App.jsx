import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import './App.css';
import { normalizePredictions } from './utils/predictionFormatter';

const WEEKDAY_LABELS = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday'
];

const HOURS = Array.from({ length: 24 }, (_, hour) => hour);

const WEEKDAY_LOOKUP = WEEKDAY_LABELS.reduce((acc, label, index) => {
  acc[label.toLowerCase()] = index;
  return acc;
}, {});

const tryParseJson = (value) => {
  if (typeof value !== 'string') {
    return value;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return value;
  }
  const startsWithBracket = trimmed.startsWith('{') || trimmed.startsWith('[');
  const endsWithBracket = trimmed.endsWith('}') || trimmed.endsWith(']');
  if (startsWithBracket && endsWithBracket) {
    try {
      return JSON.parse(trimmed);
    } catch {
      return value;
    }
  }
  return value;
};

const parseHourValue = (value) => {
  if (value === null || value === undefined || value === '') {
    return null;
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  const str = value.toString().trim().toLowerCase();
  if (!str) {
    return null;
  }

  const numeric = Number(str);
  if (Number.isFinite(numeric)) {
    return numeric;
  }

  const match = str.match(/(\d{1,2})(?::(\d{2}))?\s*(am|pm)?/);
  if (!match) {
    return null;
  }

  let hour = parseInt(match[1], 10);
  if (Number.isNaN(hour)) {
    return null;
  }

  const period = match[3];
  if (period === 'pm' && hour < 12) {
    hour += 12;
  } else if (period === 'am' && hour === 12) {
    hour = 0;
  }

  if (hour < 0) {
    return null;
  }

  if (hour > 23) {
    return hour % 24;
  }

  return hour;
};

const parseWeekdayValue = (value) => {
  if (value === null || value === undefined || value === '') {
    return null;
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  const str = value.toString().trim().toLowerCase();
  if (!str) {
    return null;
  }

  if (/^\d+$/.test(str)) {
    const numeric = parseInt(str, 10);
    return Number.isNaN(numeric) ? null : numeric;
  }

  return WEEKDAY_LOOKUP[str] ?? null;
};

const normalizeAppName = (value) =>
  value ? value.toString().trim().toLowerCase() : '';

const buildManualLogPool = (manualLogs = []) =>
  manualLogs.map((log) => ({
    log,
    normalizedApp: normalizeAppName(log.appDisplayName),
    hour: Number.isFinite(Number(log.hour)) ? Number(log.hour) : null,
    weekday: Number.isFinite(Number(log.weekday)) ? Number(log.weekday) : null,
    used: false
  }));

const findBestManualLogMatch = (info, pool) => {
  if (!pool || pool.length === 0) {
    return null;
  }

  const normalizedApp = info.app ? normalizeAppName(info.app) : null;
  let bestIndex = -1;
  let bestScore = -Infinity;
  let fallbackIndex = -1;

  pool.forEach((candidate, index) => {
    if (candidate.used) {
      return;
    }

    const appMatches = normalizedApp
      ? candidate.normalizedApp === normalizedApp
      : !!candidate.normalizedApp;

    if (!appMatches) {
      return;
    }

    let score = 0;
    score += 10; // base score for app match

    if (info.hour !== null && info.hour !== undefined) {
      if (candidate.hour !== null && candidate.hour === Number(info.hour)) {
        score += 4;
      } else {
        score -= 2;
      }
    }

    if (info.weekday !== null && info.weekday !== undefined) {
      if (candidate.weekday !== null && candidate.weekday === Number(info.weekday)) {
        score += 3;
      } else {
        score -= 1;
      }
    }

    if (bestIndex === -1) {
      fallbackIndex = index;
    }

    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });

  if (bestIndex === -1 && fallbackIndex !== -1) {
    bestIndex = fallbackIndex;
  }

  if (bestIndex !== -1) {
    pool[bestIndex].used = true;
    return pool[bestIndex].log;
  }

  return null;
};

const matchPredictionsWithManualLogs = (predictionEntries, manualLogs) => {
  if (!Array.isArray(predictionEntries) || predictionEntries.length === 0) {
    return [];
  }

  if (!Array.isArray(manualLogs) || manualLogs.length === 0) {
    return predictionEntries.map((entry) => ({
      entry,
      matchStatus: 'neutral',
      matchedLog: null,
      matchedLogSnapshot: null
    }));
  }

  const pool = buildManualLogPool(manualLogs);

  return predictionEntries.map((entry) => {
    const info = extractPredictionInfo(entry);
    const matchedLog = findBestManualLogMatch(info, pool);
    const status = matchedLog ? 'match' : 'mismatch';

    return {
      entry,
      matchStatus: status,
      matchedLog,
      matchedLogSnapshot: matchedLog
        ? {
            id: matchedLog.id,
            appDisplayName: matchedLog.appDisplayName,
            hour: matchedLog.hour,
            weekday: matchedLog.weekday,
            createdAt: matchedLog.createdAt
          }
        : null
    };
  });
};

const extractFromSource = (source) => {
  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    return {};
  }

  const app =
    source.appDisplayName ||
    source.app ||
    source.next_app ||
    source.predicted_app ||
    source.application ||
    source.target ||
    source.label ||
    source.title ||
    null;

  const hour = parseHourValue(
    source.hour ??
    source.predicted_hour ??
    source.time ??
    source.timestamp ??
    null
  );

  const weekday = parseWeekdayValue(
    source.weekday ??
    source.day ??
    source.day_of_week ??
    source.date ??
    null
  );

  return { app, hour, weekday };
};

const extractPredictionInfo = (prediction) => {
  if (!prediction) {
    return { app: null, hour: null, weekday: null };
  }

  let app = null;
  let hour = null;
  let weekday = null;

  const applyCandidate = (candidate) => {
    if (candidate && !app) {
      app = candidate;
    }
  };

  const mergeInfo = (info) => {
    if (!info) {
      return;
    }
    if (!app && info.app) {
      app = info.app;
    }
    if (hour === null || hour === undefined) {
      hour = info.hour ?? null;
    }
    if (weekday === null || weekday === undefined) {
      weekday = info.weekday ?? null;
    }
  };

  mergeInfo(extractFromSource(tryParseJson(prediction.raw)));

  if (typeof prediction.title === 'string') {
    applyCandidate(prediction.title);
  }

  if (typeof prediction.time === 'string') {
    const parsedHour = parseHourValue(prediction.time);
    if (parsedHour !== null && parsedHour !== undefined) {
      hour = parsedHour;
    }
  }

  if (Array.isArray(prediction.meta)) {
    prediction.meta.forEach(({ label, value }) => {
      const normalizedLabel = (label || '').toLowerCase();
      if (!app && normalizedLabel.includes('app')) {
        applyCandidate(value);
      }
      if (
        (hour === null || hour === undefined) &&
        (normalizedLabel.includes('hour') || normalizedLabel.includes('time'))
      ) {
        const parsedHour = parseHourValue(value);
        if (parsedHour !== null && parsedHour !== undefined) {
          hour = parsedHour;
        }
      }
      if (
        (weekday === null || weekday === undefined) &&
        (normalizedLabel.includes('week') || normalizedLabel.includes('day'))
      ) {
        const parsedWeekday = parseWeekdayValue(value);
        if (parsedWeekday !== null && parsedWeekday !== undefined) {
          weekday = parsedWeekday;
        }
      }
    });
  }

  if (!app) {
    applyCandidate(prediction.subtitle);
  }

  const normalizedHour =
    typeof hour === 'number' && Number.isFinite(hour) ? hour : null;
  const normalizedWeekday =
    typeof weekday === 'number' && Number.isFinite(weekday) ? weekday : null;

  return {
    app: app ? app.toString().trim() : null,
    hour: normalizedHour,
    weekday: normalizedWeekday
  };
};

function App() {
  const [personas, setPersonas] = useState([]);
  const [selectedPersona, setSelectedPersona] = useState('');
  const [logs, setLogs] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [logsLimit, setLogsLimit] = useState('all');
  const [totalLogs, setTotalLogs] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [addedLogs, setAddedLogs] = useState([]);
  const [newLogForm, setNewLogForm] = useState({
    appDisplayName: '',
    hour: '12',
    weekday: '0'
  });
  const [predicting, setPredicting] = useState(false);

  const addedLogsRef = useRef([]);
  const logsCardRef = useRef(null);
  const predictionsCardRef = useRef(null);

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

  const generatePredictions = useCallback(async (logData, manualLogs = []) => {
    setPredicting(true);
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
        const runId = `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const runTimestamp = new Date();
        const runTimestampIso = runTimestamp.toISOString();
        const runTimestampValue = runTimestamp.getTime();

        const comparisonLogs =
          Array.isArray(manualLogs) && manualLogs.length > 0
            ? manualLogs.slice(0, 1)
            : [];

        const matchedResults = matchPredictionsWithManualLogs(
          normalized,
          comparisonLogs
        );

        const decorated = matchedResults.map((result, index) => {
          const { entry, matchStatus, matchedLog, matchedLogSnapshot } = result;
          const baseId = entry.id || `prediction-${index}`;
          return {
            ...entry,
            id: `${baseId}-${runId}`,
            runId,
            runTimestamp: runTimestampIso,
            runTimestampValue,
            runOrder: index,
            isLatest: true,
            matchStatus,
            matchedLog,
            matchedLogSnapshot,
            matchedLogId: matchedLog?.id || matchedLogSnapshot?.id || null
          };
        });

        setPredictions((current) => {
          const getTimestamp = (prediction) => {
            if (typeof prediction.runTimestampValue === 'number') {
              return prediction.runTimestampValue;
            }
            const parsed = Date.parse(prediction.runTimestamp ?? '');
            return Number.isFinite(parsed) ? parsed : 0;
          };

          const updatedExisting = current.map((pred, index) => ({
            ...pred,
            isLatest: false,
            runOrder: pred.runOrder ?? index
          }));

          return [...decorated, ...updatedExisting].sort((a, b) => {
            const timestampDiff = getTimestamp(b) - getTimestamp(a);
            if (timestampDiff !== 0) {
              return timestampDiff;
            }
            return (a.runOrder ?? 0) - (b.runOrder ?? 0);
          });
        });
      } else {
        setError(data.error);
        setPredictions([]);
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message);
      setPredictions([]);
    } finally {
      setPredicting(false);
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
        if (data.logs.length > 0 || addedLogsRef.current.length > 0) {
          const combinedData = [
            ...addedLogsRef.current,
            ...data.logs
          ];
          await generatePredictions(
            combinedData,
            addedLogsRef.current.slice(0, 1)
          );
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

  useEffect(() => {
    addedLogsRef.current = addedLogs;
  }, [addedLogs]);

  const displayedLogs = useMemo(
    () => [...addedLogs, ...logs],
    [addedLogs, logs]
  );

  const displayedLogCount = useMemo(
    () => totalLogs + addedLogs.length,
    [totalLogs, addedLogs]
  );

  const latestRunId = useMemo(
    () => (predictions.length > 0 ? predictions[0].runId : null),
    [predictions]
  );

  const latestPredictionCount = useMemo(() => {
    if (!latestRunId) {
      return predictions.length;
    }
    return predictions.reduce(
      (count, prediction) => count + (prediction.runId === latestRunId ? 1 : 0),
      0
    );
  }, [predictions, latestRunId]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const logsElement = logsCardRef.current;
    const predictionsElement = predictionsCardRef.current;

    if (!logsElement || !predictionsElement) {
      return;
    }

    const defaultMinHeight =
      parseFloat(window.getComputedStyle(predictionsElement).minHeight) || 0;

    const updatePredictionCardHeight = () => {
      if (!logsCardRef.current || !predictionsCardRef.current) {
        return;
      }

      if (
        typeof window.matchMedia === 'function' &&
        window.matchMedia('(max-width: 1024px)').matches
      ) {
        predictionsCardRef.current.style.minHeight = '';
        return;
      }

      const logsHeight = logsCardRef.current.getBoundingClientRect().height || 0;
      const targetHeight = Math.max(logsHeight, defaultMinHeight);

      if (targetHeight > 0) {
        predictionsCardRef.current.style.minHeight = `${Math.round(targetHeight)}px`;
      } else {
        predictionsCardRef.current.style.minHeight = defaultMinHeight
          ? `${defaultMinHeight}px`
          : '';
      }
    };

    updatePredictionCardHeight();

    let resizeObserver;

    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(updatePredictionCardHeight);
      resizeObserver.observe(logsElement);
    }

    window.addEventListener('resize', updatePredictionCardHeight);

    return () => {
      window.removeEventListener('resize', updatePredictionCardHeight);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      if (predictionsElement) {
        predictionsElement.style.minHeight = '';
      }
    };
  }, [displayedLogs.length, predictions.length, loading, predicting]);

  const handleNewLogFieldChange = (field) => (event) => {
    setNewLogForm((current) => ({
      ...current,
      [field]: event.target.value
    }));
  };

  const handleAddLog = async (event) => {
    event.preventDefault();
    const trimmedApp = newLogForm.appDisplayName.trim();
    if (!trimmedApp) {
      setError('Application name is required to add a log.');
      return;
    }

    const manualLog = {
      id: `manual-${Date.now()}`,
      appDisplayName: trimmedApp,
      hour: Number(newLogForm.hour),
      weekday: Number(newLogForm.weekday),
      isManual: true,
      createdAt: new Date().toISOString()
    };

    const updatedManualLogs = [manualLog, ...addedLogsRef.current];
    setAddedLogs(updatedManualLogs);

    const combinedLogs = [...updatedManualLogs, ...logs];

    try {
      await generatePredictions(combinedLogs, updatedManualLogs.slice(0, 1));
      setError(null);
    } catch (err) {
      console.error('Error adding manual log:', err);
    } finally {
      setNewLogForm((current) => ({
        ...current,
        appDisplayName: ''
      }));
    }
  };

  const handlePersonaChange = (event) => {
    const nextPersona = event.target.value;
    const hasManualLogs = addedLogsRef.current.length > 0 || addedLogs.length > 0;
    if (hasManualLogs) {
      addedLogsRef.current = [];
      setAddedLogs([]);
    }
    setSelectedPersona(nextPersona);
  };

  const formatHour = (hour) => {
    const h = hour % 12 || 12;
    const ampm = hour < 12 ? 'AM' : 'PM';
    return `${h}:00 ${ampm}`;
  };

  const formatWeekday = (day) => {
    const numeric = Number(day);
    if (!Number.isNaN(numeric) && WEEKDAY_LABELS[numeric]) {
      return WEEKDAY_LABELS[numeric];
    }
    if (typeof day === 'string') {
      const normalized = day.trim().toLowerCase();
      if (WEEKDAY_LOOKUP[normalized] !== undefined) {
        return WEEKDAY_LABELS[WEEKDAY_LOOKUP[normalized]];
      }
    }
    return day;
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
              onChange={handlePersonaChange}
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
                  {displayedLogCount} logs
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
            <div ref={logsCardRef} className="card logs-card">
              {loading ? (
                <div className="loading">Loading logs...</div>
              ) : displayedLogs.length === 0 ? (
                <div className="empty-state">No logs available for this persona</div>
              ) : (
                <div className="logs-list">
                  {displayedLogs.map((log, index) => {
                    const timestamp = log.timestamp || log.createdAt;
                    const key = log.id || `${log.appDisplayName}-${log.hour}-${log.weekday}-${index}`;
                    return (
                      <div
                        key={key}
                        className={`log-item${log.isManual ? ' log-item--manual' : ''}`}
                      >
                        <div className="log-header">
                          <span className="log-app">{log.appDisplayName}</span>
                          <span className="log-time">{formatHour(log.hour)}</span>
                        </div>
                        <div className="log-details">
                          <span className="log-day">{formatWeekday(log.weekday)}</span>
                          {timestamp && (
                            <span className="log-timestamp">
                              {new Date(timestamp).toLocaleString()}
                            </span>
                          )}
                          {log.isManual && (
                            <span className="log-manual-indicator">Added locally</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            <div className="card add-log-card">
              <h3>Add Log</h3>
              <p className="add-log-description">
                Simulate a new activity log for the selected persona. Added logs stay local for demo purposes.
              </p>
              <form className="add-log-form" onSubmit={handleAddLog}>
                <div className="form-grid">
                  <label className="form-field">
                    <span>Application</span>
                    <input
                      type="text"
                      value={newLogForm.appDisplayName}
                      onChange={handleNewLogFieldChange('appDisplayName')}
                      placeholder="e.g., Canvas"
                      disabled={loading || predicting}
                      required
                    />
                  </label>
                  <label className="form-field">
                    <span>Hour</span>
                    <select
                      value={newLogForm.hour}
                      onChange={handleNewLogFieldChange('hour')}
                      disabled={loading || predicting}
                    >
                      {HOURS.map((hour) => (
                        <option key={hour} value={hour}>
                          {formatHour(hour)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="form-field">
                    <span>Weekday</span>
                    <select
                      value={newLogForm.weekday}
                      onChange={handleNewLogFieldChange('weekday')}
                      disabled={loading || predicting}
                    >
                      {WEEKDAY_LABELS.map((label, index) => (
                        <option key={label} value={index}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="add-log-actions">
                  <button
                    type="submit"
                    className="add-log-button"
                    disabled={loading || predicting}
                  >
                    ➕ Add Log
                  </button>
                  {(loading || predicting) && (
                    <span className="add-log-status">
                      {loading ? 'Refreshing logs…' : 'Updating predictions…'}
                    </span>
                  )}
                </div>
              </form>
            </div>
          </div>

          {/* Predictions Section */}
          <div className="section">
            <div className="section-header">
              <h2>🎯 ML Predictions</h2>
              <span className="count-badge">
                {latestPredictionCount} predictions
              </span>
            </div>
            <div ref={predictionsCardRef} className="card predictions-card">
              {loading || predicting ? (
                <div className="loading">
                  {loading ? 'Generating predictions...' : 'Updating predictions...'}
                </div>
              ) : predictions.length === 0 ? (
                <div className="empty-state">No predictions available</div>
              ) : (
                <div className="predictions-list">
                  {predictions.map((pred) => {
                    const matchedLogForDisplay =
                      pred.matchedLog || pred.matchedLogSnapshot;

                    return (
                      <div
                        key={pred.id}
                        className={`prediction-item prediction-item--${pred.matchStatus || 'neutral'} ${pred.isLatest ? 'prediction-item--latest' : 'prediction-item--history'}`}
                      >
                        <div className="prediction-header">
                          <span className="prediction-app">{pred.title}</span>
                          {pred.matchStatus !== 'neutral' && (
                            <span className={`prediction-status prediction-status--${pred.matchStatus}`}>
                              {pred.matchStatus === 'match' ? '✅ Matches added log' : '❌ No match'}
                            </span>
                          )}
                          {pred.time && (
                            <span className="prediction-time">{pred.time}</span>
                          )}
                        </div>

                        {(pred.subtitle || pred.meta.length > 0 || (pred.matchStatus === 'match' && matchedLogForDisplay)) && (
                          <div className="prediction-details">
                            {pred.subtitle && (
                              <span className="prediction-subtitle">
                                {pred.subtitle}
                              </span>
                            )}
                            {pred.matchStatus === 'match' && matchedLogForDisplay && (
                              <span className="prediction-match">
                                Matches {matchedLogForDisplay.appDisplayName} · {formatWeekday(matchedLogForDisplay.weekday)} at {formatHour(matchedLogForDisplay.hour)}
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
                    );
                  })}
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
