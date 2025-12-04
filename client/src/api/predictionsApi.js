import apiClient from './client';

export const predictionsApi = {
  // Generate predictions from logs
  predict: async (logs) => {
    const response = await apiClient.post('/predict', { data: logs });
    return response.data;
  },

  // Record app access and get updated predictions
  recordAppAccess: async (logs, appDisplayName) => {
    const response = await apiClient.post('/record-app-access', {
      logs,
      appDisplayName,
    });
    return response.data;
  },
};
