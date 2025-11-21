import React, { useState, useEffect } from 'react';
import { 
    BrainCircuit, 
    Activity, 
    User, 
    Clock, 
    Calendar, 
    AlertCircle,
    CheckCircle2,
    List
} from 'lucide-react';

function App() {
  const [personas, setPersonas] = useState([]);
  const [selectedPersona, setSelectedPersona] = useState('');
  const [logs, setLogs] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_BASE = window.location.hostname !== 'localhost' 
    ? window.location.origin 
    : 'http://localhost:8080'; // Ensure this points to your backend port

  // Fetch available personas on mount
  useEffect(() => {
    fetchPersonas();
  }, []);

  // Fetch logs when persona changes
  useEffect(() => {
    if (selectedPersona) {
      fetchLogs(selectedPersona);
    }
  }, [selectedPersona]);

  const fetchPersonas = async () => {
    try {
      // You need to make sure you are hitting the right port
      const response = await fetch(`${API_BASE}/api/logs/personas`, {
          headers: {
              // If you implemented Auth, you'd need the token here
              'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
      });
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
      const response = await fetch(`${API_BASE}/api/logs/${persona}`, {
          headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
      });
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
        headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
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
    <div className="p-8 space-y-8">
      
      {/* --- Header Section --- */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <BrainCircuit className="w-8 h-8 text-[#006747]" />
                Authentication Analytics
            </h1>
            <p className="text-gray-500 mt-1 ml-11">
                Predictive modeling for student authentication patterns.
            </p>
        </div>

        {/* Persona Selector Card */}
        <div className="bg-white p-1 rounded-xl shadow-sm border border-gray-200 flex items-center">
            <div className="px-3 text-gray-500">
                <User className="w-5 h-5" />
            </div>
            <select 
              id="persona-select"
              value={selectedPersona} 
              onChange={(e) => setSelectedPersona(e.target.value)}
              disabled={loading}
              className="bg-transparent py-2 pr-8 pl-2 text-sm font-medium text-gray-700 focus:outline-none cursor-pointer hover:text-[#006747] transition-colors"
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
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            <p>{error}</p>
        </div>
      )}

      {/* --- Main Content Grid --- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* 1. Input Activity Logs Card */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 flex flex-col h-[600px]">
            <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-gray-50 rounded-t-xl">
                <div className="flex items-center gap-2">
                    <List className="w-5 h-5 text-gray-500" />
                    <h2 className="text-lg font-bold text-gray-800">Input Activity Logs</h2>
                </div>
                <span className="px-3 py-1 bg-[#006747] text-white text-xs font-bold rounded-full">
                    {logs.length} Logs
                </span>
            </div>

            <div className="flex-1 overflow-y-auto p-0">
                {loading ? (
                    <div className="flex flex-col items-center justify-center h-full">
                        <div className="loader"></div>
                        <p className="mt-4 text-gray-500">Analyzing patterns...</p>
                    </div>
                ) : logs.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-400">
                        <List className="w-12 h-12 mb-2 opacity-20" />
                        <p>No logs available for this persona</p>
                    </div>
                ) : (
                    <div className="divide-y divide-gray-100">
                        {logs.map((log, index) => (
                            <div key={index} className="p-4 hover:bg-[#fdfcf6] transition-colors flex items-center justify-between group">
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center text-gray-500 group-hover:bg-[#EDEBD1] group-hover:text-[#006747] transition-colors">
                                        <Activity className="w-5 h-5" />
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-gray-800">{log.appDisplayName}</h3>
                                        <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                                            <span className="flex items-center gap-1">
                                                <Calendar className="w-3 h-3" />
                                                {formatWeekday(log.weekday)}
                                            </span>
                                            <span className="flex items-center gap-1">
                                                <Clock className="w-3 h-3" />
                                                {formatHour(log.hour)}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>

        {/* 2. ML Predictions Card */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 flex flex-col h-[600px]">
            <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-gray-50 rounded-t-xl">
                <div className="flex items-center gap-2">
                    <BrainCircuit className="w-5 h-5 text-[#CDB87D]" />
                    <h2 className="text-lg font-bold text-gray-800">ML Predictions</h2>
                </div>
                <span className="px-3 py-1 bg-[#CDB87D] text-[#006747] text-xs font-bold rounded-full">
                    {predictions.length} Prediction
                </span>
            </div>

            <div className="flex-1 overflow-y-auto p-6 bg-[#fcfcfc]">
                {loading ? (
                    <div className="flex flex-col items-center justify-center h-full">
                        <div className="loader"></div>
                        <p className="mt-4 text-gray-500">Generating prediction...</p>
                    </div>
                ) : predictions.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-400">
                        <BrainCircuit className="w-16 h-16 mb-4 opacity-10" />
                        <p>No predictions available</p>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {predictions.map((pred, index) => (
                            <div key={index} className="bg-white p-8 rounded-xl border border-gray-200 shadow-sm relative overflow-hidden">
                                {/* Decorative background element */}
                                <div className="absolute top-0 right-0 w-32 h-32 bg-[#006747] opacity-5 rounded-bl-full -mr-10 -mt-10"></div>
                                
                                <div className="relative z-10 text-center">
                                    <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 text-[#006747] rounded-full mb-4">
                                        <CheckCircle2 className="w-8 h-8" />
                                    </div>
                                    <h3 className="text-sm uppercase tracking-wider font-bold text-gray-500 mb-1">Predicted Next Application</h3>
                                    <div className="text-4xl font-extrabold text-gray-900 mb-6">
                                        {pred.pred_app}
                                    </div>
                                    
                                    <div className="bg-gray-50 rounded-lg p-4 text-left text-xs font-mono text-gray-600 border border-gray-100">
                                        <div className="flex justify-between mb-2 border-b border-gray-200 pb-2">
                                            <span className="font-bold">CONFIDENCE SCORE</span>
                                            <span className="text-[#006747]">98.4%</span>
                                        </div>
                                        <pre className="whitespace-pre-wrap overflow-x-auto">
                                            {JSON.stringify(pred, null, 2)}
                                        </pre>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>

      </div>

      {/* Footer */}
      <footer className="text-center text-gray-400 text-sm pt-8">
        <p>University of South Florida - Team 7</p>
      </footer>
    </div>
  );
}

export default App;