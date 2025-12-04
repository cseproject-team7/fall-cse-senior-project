import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { generatePredictions, recordAppAccess } from '../store/predictionsSlice';

export const usePredictions = (logs) => {
  const dispatch = useDispatch();
  const { predictions, loading, error, recordingApp } = useSelector(
    (state) => state.predictions
  );

  useEffect(() => {
    if (logs && logs.length > 0) {
      dispatch(generatePredictions(logs));
    }
  }, [logs, dispatch]);

  const handleRecordAppAccess = (appDisplayName) => {
    dispatch(recordAppAccess({ logs, appDisplayName }));
  };

  return {
    predictions,
    loading,
    error,
    recordingApp,
    handleRecordAppAccess,
  };
};
