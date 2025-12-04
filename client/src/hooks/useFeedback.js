import { useState } from 'react';
import { feedbackApi } from '../api/feedbackApi';

export const useFeedback = (logs, predictions, selectedPersona) => {
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleIncorrectPrediction = () => {
    setShowFeedbackModal(true);
  };

  const submitFeedback = async (actualApp) => {
    setSubmitting(true);
    setError(null);

    try {
      await feedbackApi.submitFeedback(logs, predictions, actualApp, selectedPersona);
      setShowFeedbackModal(false);
      alert('Thank you! Your feedback has been stored for model retraining.');
    } catch (err) {
      setError(err.message);
      console.error('Submit feedback error:', err);
    } finally {
      setSubmitting(false);
    }
  };

  return {
    showFeedbackModal,
    setShowFeedbackModal,
    handleIncorrectPrediction,
    submitFeedback,
    submitting,
    error,
  };
};
