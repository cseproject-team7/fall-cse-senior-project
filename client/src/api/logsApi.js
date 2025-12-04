import apiClient from './client';

export const logsApi = {
  // Get all available personas
  getPersonas: async () => {
    const response = await apiClient.get('/logs/personas');
    return response.data;
  },

  // Get logs for a specific persona
  getLogsByPersona: async (persona) => {
    const response = await apiClient.get(`/logs/${persona}`);
    return response.data;
  },
};
