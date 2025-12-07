const localMLService = require('../../services/localMLService');
const axios = require('axios');

// Mock axios
jest.mock('axios');

describe('localMLService', () => {
  let originalEnv;

  beforeAll(() => {
    originalEnv = process.env;
  });

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = {
      ...originalEnv,
      ML_SERVER_URL: 'http://localhost:5001',
      AZURE_ML_KEY: ''
    };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  describe('predict', () => {
    const mockMLResponse = {
      success: true,
      predictions: [
        {
          pattern: 'Canvas → Outlook',
          confidence: 92,
          top_apps: ['Outlook', 'Teams']
        }
      ],
      sessions_analyzed: 15
    };

    it('should handle array data and call local ML server', async () => {
      axios.post.mockResolvedValue({ data: mockMLResponse });

      const logs = [
        { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }
      ];

      const result = await localMLService.predict(logs);

      expect(axios.post).toHaveBeenCalledWith(
        'http://localhost:5001/predict',
        {
          logs: logs.map(log => ({
            appDisplayName: log.appDisplayName,
            createdDateTime: log.createdDateTime
          })),
          num_predictions: 10
        },
        expect.objectContaining({
          headers: { 'Content-Type': 'application/json' },
          timeout: 30000
        })
      );

      expect(result).toEqual({
        success: true,
        next_pattern: 'Canvas → Outlook',
        pattern_confidence: 92,
        next_apps: ['Outlook', 'Teams'],
        predictions: mockMLResponse.predictions,
        sessions_analyzed: 15
      });
    });

    it('should handle object with data field', async () => {
      axios.post.mockResolvedValue({ data: mockMLResponse });

      const input = {
        data: [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }]
      };

      await localMLService.predict(input);

      expect(axios.post).toHaveBeenCalled();
      const callArgs = axios.post.mock.calls[0][1];
      expect(callArgs.logs).toHaveLength(1);
    });

    it('should parse string JSON response from Azure ML', async () => {
      process.env.ML_SERVER_URL = 'https://test.inference.ml.azure.com/score';
      process.env.AZURE_ML_KEY = 'test-key';

      axios.post.mockResolvedValue({ data: JSON.stringify(mockMLResponse) });

      const logs = [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }];
      const result = await localMLService.predict(logs);

      expect(axios.post).toHaveBeenCalledWith(
        'https://test.inference.ml.azure.com/score',
        expect.any(Object),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-key'
          })
        })
      );

      expect(result.success).toBe(true);
    });

    it('should throw error when ML response success is false', async () => {
      axios.post.mockResolvedValue({
        data: { success: false, error: 'Model failed' }
      });

      const logs = [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }];

      await expect(localMLService.predict(logs)).rejects.toThrow('Model failed');
    });

    it('should throw error when axios request fails', async () => {
      axios.post.mockRejectedValue(new Error('Network error'));

      const logs = [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }];

      await expect(localMLService.predict(logs)).rejects.toThrow('Network error');
    });
  });

  describe('predictWithNewAccess', () => {
    const mockMLResponse = {
      success: true,
      predictions: [
        {
          pattern: 'Outlook → Teams',
          confidence: 88,
          top_apps: ['Teams', 'Canvas']
        }
      ],
      sessions_analyzed: 20,
      logs_used: 5
    };

    it('should add new_app_access to request payload', async () => {
      axios.post.mockResolvedValue({ data: mockMLResponse });

      const logs = [
        { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }
      ];
      const newApp = 'Outlook';

      const result = await localMLService.predictWithNewAccess(logs, newApp);

      expect(axios.post).toHaveBeenCalled();
      const callArgs = axios.post.mock.calls[0][1];
      expect(callArgs.new_app_access).toEqual({
        appDisplayName: newApp,
        createdDateTime: expect.any(String)
      });
      expect(result.recorded_app).toBe(newApp);
    });

    it('should throw error when ML service fails', async () => {
      axios.post.mockRejectedValue(new Error('Service unavailable'));

      const logs = [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }];

      await expect(
        localMLService.predictWithNewAccess(logs, 'Outlook')
      ).rejects.toThrow('Service unavailable');
    });
  });

  describe('predictPersona', () => {
    it('should validate inputs', async () => {
      await expect(
        localMLService.predictPersona(null, [])
      ).rejects.toThrow('Expected userId and records');

      await expect(
        localMLService.predictPersona('user1', 'not-array')
      ).rejects.toThrow('Expected userId and records');
    });

    it('should require at least 5 records', async () => {
      const records = [
        { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }
      ];

      await expect(
        localMLService.predictPersona('user1', records)
      ).rejects.toThrow('at least 5 app access records');
    });

    it('should validate record structure', async () => {
      const records = [
        { appDisplayName: 'Canvas' }, // Missing createdDateTime
        { appDisplayName: 'Outlook', createdDateTime: '2025-01-01T09:00:00Z' },
        { appDisplayName: 'Teams', createdDateTime: '2025-01-01T10:00:00Z' },
        { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T11:00:00Z' },
        { appDisplayName: 'Outlook', createdDateTime: '2025-01-01T12:00:00Z' }
      ];

      await expect(
        localMLService.predictPersona('user1', records)
      ).rejects.toThrow('must have appDisplayName and createdDateTime');
    });

    it('should call ML server with valid inputs', async () => {
      axios.post.mockResolvedValue({
        data: { success: true, persona: 'engineering_junior' }
      });

      const records = [
        { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' },
        { appDisplayName: 'Outlook', createdDateTime: '2025-01-01T10:00:00Z' },
        { appDisplayName: 'Teams', createdDateTime: '2025-01-01T11:00:00Z' },
        { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T12:00:00Z' },
        { appDisplayName: 'MATLAB', createdDateTime: '2025-01-01T13:00:00Z' }
      ];

      const result = await localMLService.predictPersona('user1', records);

      expect(axios.post).toHaveBeenCalledWith(
        'http://localhost:5001/api/predict-persona',
        { userId: 'user1', records },
        expect.any(Object)
      );
      expect(result.persona).toBe('engineering_junior');
    });
  });
});

