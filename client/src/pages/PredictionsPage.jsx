import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { BrainCircuit, AlertCircle } from 'lucide-react';
import { fetchPersonas, setSelectedPersona } from '../store/personaSlice';
import { fetchLogs } from '../store/logsSlice';
import { usePredictions } from '../hooks/usePredictions';
import { useFeedback } from '../hooks/useFeedback';
import PersonaSelector from '../features/personaSelector/PersonaSelector';
import ActivityLogs from '../features/activityLogs/ActivityLogs';
import Predictions from '../features/predictions/Predictions';
import FeedbackModal from '../features/feedback/FeedbackModal';

function PredictionsPage() {
  const dispatch = useDispatch();
  
  // Redux state
  const { personas, selectedPersona, loading: personaLoading, error: personaError } = useSelector(
    (state) => state.persona
  );
  const { logs, loading: logsLoading, error: logsError } = useSelector((state) => state.logs);

  // Custom hooks
  const {
    predictions,
    loading: predictionsLoading,
    error: predictionsError,
    recordingApp,
    handleRecordAppAccess,
  } = usePredictions(logs);

  const {
    showFeedbackModal,
    setShowFeedbackModal,
    handleIncorrectPrediction,
    submitFeedback,
  } = useFeedback(logs, predictions, selectedPersona);

  // Fetch personas on mount
  useEffect(() => {
    dispatch(fetchPersonas());
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
            <BrainCircuit className="w-8 h-8 text-[#006747]" />
            Authentication Analytics
          </h1>
          <p className="text-gray-500 mt-1 ml-11">
            Predictive modeling for student authentication patterns.
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
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2 flex-shrink-0">
          <AlertCircle className="w-5 h-5" />
          <p>{error}</p>
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1 min-h-0 mt-8">
        <ActivityLogs logs={logs} loading={logsLoading} />
        <Predictions
          predictions={predictions}
          loading={predictionsLoading}
          onIncorrectPrediction={handleIncorrectPrediction}
          onRecordAppAccess={handleRecordAppAccess}
          recordingApp={recordingApp}
        />
      </div>

      {/* Feedback Modal */}
      {showFeedbackModal && (
        <FeedbackModal
          onSubmit={submitFeedback}
          onClose={() => setShowFeedbackModal(false)}
        />
      )}

      {/* Footer */}
      <footer className="text-center text-gray-400 text-sm pt-4 flex-shrink-0">
        <p>University of South Florida - Team 7</p>
      </footer>
    </div>
  );
}

export default PredictionsPage;
