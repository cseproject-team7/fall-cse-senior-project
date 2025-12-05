import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Beaker, AlertCircle } from 'lucide-react';
import { fetchPersonas, setSelectedPersona } from '../store/personaSlice';
import { fetchLogs, fetchAllGroupedLogs } from '../store/logsSlice';
import { usePredictions } from '../hooks/usePredictions';
import PersonaSelector from '../features/personaSelector/PersonaSelector';
import ActivityLogs from '../features/activityLogs/ActivityLogs';
import PredictionCard from '../features/predictions/PredictionCard';
import PredictionTimeline from '../features/predictions/PredictionTimeline';

function TestLabPage() {
  const dispatch = useDispatch();
  
  // Redux state
  const { personas, selectedPersona, loading: personaLoading, error: personaError } = useSelector(
    (state) => state.persona
  );
  const { logs, loading: logsLoading, error: logsError } = useSelector((state) => state.logs);

  // Custom hooks
  const {
    manualPredictions,
    manualLogs,
    loading: predictionsLoading,
    error: predictionsError,
    handleAddTestApp,
    handleResetManual,
  } = usePredictions(logs);

  // Fetch grouped logs and personas on mount
  useEffect(() => {
    const loadData = async () => {
      await dispatch(fetchAllGroupedLogs());
      await dispatch(fetchPersonas());
    };
    loadData();
  }, [dispatch]);

  // Fetch logs when persona changes
  useEffect(() => {
    if (selectedPersona) {
      dispatch(fetchLogs(selectedPersona));
    }
  }, [selectedPersona, dispatch]);

  const handlePersonaChange = (persona) => {
    dispatch(setSelectedPersona(persona));
  };

  const error = personaError || logsError || predictionsError;

  return (
    <div className="h-screen overflow-hidden flex flex-col p-8">
      {/* Header Section */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 flex-shrink-0">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
            <Beaker className="w-8 h-8 text-purple-600" />
            Test Lab
          </h1>
          <p className="text-gray-500 mt-1 ml-11">
            Experiment with manual test scenarios and "what-if" predictions.
          </p>
        </div>

        <PersonaSelector
          personas={personas}
          selectedPersona={selectedPersona}
          onPersonaChange={handlePersonaChange}
          disabled={personaLoading}
        />
      </header>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2 flex-shrink-0 mt-4">
          <AlertCircle className="w-5 h-5" />
          <p>{error}</p>
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1 min-h-0 mt-8">
        {/* Activity Logs */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 flex flex-col h-full overflow-hidden">
          <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-gray-50 rounded-t-xl flex-shrink-0">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold text-gray-800">Test Logs</h2>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">
                {manualLogs?.length || 0} Events
              </span>
              <button
                onClick={handleResetManual}
                className="text-xs px-3 py-1 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-all"
              >
                Reset to Real Logs
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            {logsLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="loader"></div>
              </div>
            ) : manualLogs && manualLogs.length > 0 ? (
              <div className="space-y-3">
                {manualLogs.map((log, idx) => (
                  <div 
                    key={idx} 
                    className={`flex items-center gap-3 p-3 rounded-lg ${log.isManualTest ? 'bg-purple-50 border-2 border-purple-200' : 'bg-gray-50'}`}
                  >
                    <div className="text-2xl">📱</div>
                    <div className="flex-1">
                      <div className="font-semibold text-gray-800 flex items-center gap-2">
                        {log.appDisplayName}
                        {log.isManualTest && (
                          <span className="text-xs px-2 py-0.5 bg-purple-200 text-purple-800 rounded-full font-bold">
                            TEST
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(log.createdDateTime).toLocaleString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <p>No logs available. Select a user to begin.</p>
              </div>
            )}
          </div>
        </div>

        {/* Manual Test Predictions */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 flex flex-col h-full overflow-hidden">
          <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-purple-50 rounded-t-xl flex-shrink-0">
            <div className="flex items-center gap-2">
              <Beaker className="w-5 h-5 text-purple-600" />
              <h2 className="text-lg font-bold text-gray-800">Test Predictions</h2>
            </div>
            <span className="px-3 py-1 bg-purple-600 text-white text-xs font-bold rounded-full">
              Manual Mode
            </span>
          </div>

          <div className="flex-1 overflow-y-auto p-6 bg-purple-50 bg-opacity-30">
            {predictionsLoading ? (
              <div className="flex flex-col items-center justify-center h-full">
                <div className="loader"></div>
                <p className="mt-4 text-gray-500">Generating predictions...</p>
              </div>
            ) : !manualPredictions || !manualPredictions.next_pattern ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <Beaker className="w-16 h-16 mb-4 opacity-10" />
                <p>No predictions available</p>
              </div>
            ) : (
              <div className="space-y-4">
                <PredictionCard
                  prediction={manualPredictions}
                  mode="manual"
                  onAddTestApp={handleAddTestApp}
                />
                {manualPredictions.predictions && manualPredictions.predictions.length > 0 && (
                  <PredictionTimeline predictions={manualPredictions.predictions} />
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="text-center text-gray-400 text-sm pt-4 flex-shrink-0">
        <p>University of South Florida - Team 7 | Experimental Test Environment</p>
      </footer>
    </div>
  );
}

export default TestLabPage;
