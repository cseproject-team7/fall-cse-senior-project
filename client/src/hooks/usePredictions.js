import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { 
  generateAutoPredictions, 
  generateManualPredictions,
  addManualTestApp,
  resetManualLogs
} from '../store/predictionsSlice';

export const usePredictions = (logs) => {
  const dispatch = useDispatch();
  const { autoPredictions, manualPredictions, manualLogs, loading, error } = useSelector(
    (state) => state.predictions
  );

  // Initialize manual logs with real logs
  useEffect(() => {
    if (logs && logs.length > 0) {
      dispatch(resetManualLogs(logs));
    }
  }, [logs, dispatch]);

  // Generate automatic predictions (real logs only)
  useEffect(() => {
    if (logs && logs.length > 0) {
      dispatch(generateAutoPredictions(logs));
    }
  }, [logs, dispatch]);

  // Generate manual predictions (with test apps)
  useEffect(() => {
    if (manualLogs && manualLogs.length > 0) {
      dispatch(generateManualPredictions(manualLogs));
    }
  }, [manualLogs, dispatch]);

  // Add test app to manual mode
  const handleAddTestApp = (appDisplayName) => {
    dispatch(addManualTestApp({ 
      appDisplayName, 
      timestamp: new Date().toISOString() 
    }));
  };

  // Reset manual mode to match real logs
  const handleResetManual = () => {
    dispatch(resetManualLogs(logs));
  };

  return {
    autoPredictions,
    manualPredictions,
    manualLogs,
    loading,
    error,
    handleAddTestApp,
    handleResetManual,
  };
};
