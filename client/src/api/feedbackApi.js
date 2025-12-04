import apiClient from './client';

export const feedbackApi = {
  // Submit incorrect prediction feedback
  submitFeedback: async (logs, prediction, actualApp, userId) => {
    const response = await apiClient.post('/feedback', {
      logs,
      prediction,
      actualApp,
      userId,
    });
    return response.data;
  },

  // Get feedback statistics
  getFeedbackStats: async () => {
    const response = await apiClient.get('/feedback/stats');
    return response.data;
  },
};
