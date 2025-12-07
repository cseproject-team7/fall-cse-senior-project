const request = require('supertest');
const createTestApp = require('../utils/testApp');

// Mock the services to avoid external dependencies
jest.mock('../../services/eventHubService');
jest.mock('../../services/localMLService');
jest.mock('../../services/kafkaService');
jest.mock('@azure/storage-blob');

const eventHubService = require('../../services/eventHubService');
const localMLService = require('../../services/localMLService');
const kafkaService = require('../../services/kafkaService');

describe('Public API Integration Tests', () => {
  let app;

  beforeAll(() => {
    app = createTestApp();
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('GET /api/health', () => {
    it('should return 200 with ok status', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect(200);

      expect(response.body.status).toBe('ok');
      expect(response.body.timestamp).toBeDefined();
      
      // Verify timestamp is a valid ISO string
      const timestamp = new Date(response.body.timestamp);
      expect(timestamp.toString()).not.toBe('Invalid Date');
    });
  });

  describe('GET /api/logs/personas', () => {
    it('should return list of personas', async () => {
      const mockPersonas = ['user_1', 'user_2', 'user_3'];
      eventHubService.getPersonas.mockResolvedValue(mockPersonas);

      const response = await request(app)
        .get('/api/logs/personas')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.personas).toEqual(mockPersonas);
    });

    it('should return 500 when service fails', async () => {
      eventHubService.getPersonas.mockRejectedValue(new Error('Service error'));

      const response = await request(app)
        .get('/api/logs/personas')
        .expect(500);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeDefined();
    });
  });

  describe('GET /api/logs/:persona', () => {
    it('should return logs for specified persona', async () => {
      const mockLogs = [
        { appDisplayName: 'Canvas', hour: 9, weekday: 1, createdDateTime: '2025-01-01T09:00:00Z' },
        { appDisplayName: 'Outlook', hour: 10, weekday: 1, createdDateTime: '2025-01-01T10:00:00Z' }
      ];
      eventHubService.getLogsByPersona.mockResolvedValue(mockLogs);

      const response = await request(app)
        .get('/api/logs/user_1')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.persona).toBe('user_1');
      expect(response.body.count).toBe(2);
      expect(response.body.logs).toEqual(mockLogs);
    });

    it('should respect limit query parameter', async () => {
      const mockLogs = [
        { appDisplayName: 'Canvas', hour: 9, weekday: 1, createdDateTime: '2025-01-01T09:00:00Z' }
      ];
      eventHubService.getLogsByPersona.mockResolvedValue(mockLogs);

      await request(app)
        .get('/api/logs/user_2?limit=10')
        .expect(200);

      expect(eventHubService.getLogsByPersona).toHaveBeenCalledWith('user_2', 10);
    });

    it('should use default limit when not specified', async () => {
      eventHubService.getLogsByPersona.mockResolvedValue([]);

      await request(app)
        .get('/api/logs/user_3')
        .expect(200);

      expect(eventHubService.getLogsByPersona).toHaveBeenCalledWith('user_3', 50);
    });
  });

  describe('GET /api/logs-grouped', () => {
    it('should return logs grouped by user', async () => {
      // Since this doesn't require auth in current code
      const response = await request(app)
        .get('/api/logs-grouped')
        .expect(200);

      // Response should have users array and logsByUser object
      expect(response.body.users).toBeDefined();
      expect(response.body.logsByUser).toBeDefined();
    });
  });

  describe('POST /api/predict', () => {
    it('should return prediction for valid input', async () => {
      const mockPrediction = {
        success: true,
        next_pattern: 'Canvas → Outlook',
        pattern_confidence: 92,
        next_apps: ['Outlook', 'Teams'],
        predictions: [{ pattern: 'Canvas → Outlook', confidence: 92 }],
        sessions_analyzed: 15
      };
      localMLService.predict.mockResolvedValue(mockPrediction);

      const response = await request(app)
        .post('/api/predict')
        .send({
          data: [
            { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }
          ]
        })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.prediction).toBeDefined();
      expect(response.body.timestamp).toBeDefined();
    });

    it('should handle array body format', async () => {
      const mockPrediction = {
        success: true,
        next_pattern: 'Canvas → Outlook',
        predictions: []
      };
      localMLService.predict.mockResolvedValue(mockPrediction);

      const response = await request(app)
        .post('/api/predict')
        .send([
          { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }
        ])
        .expect(200);

      expect(response.body.success).toBe(true);
    });

    it('should return 500 when ML service fails', async () => {
      localMLService.predict.mockRejectedValue(new Error('ML server timeout'));

      const response = await request(app)
        .post('/api/predict')
        .send({
          data: [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }]
        })
        .expect(500);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeDefined();
    });
  });

  describe('POST /api/record-app-access', () => {
    it('should record app access and return updated prediction', async () => {
      const mockPrediction = {
        success: true,
        next_pattern: 'Outlook → Teams',
        recorded_app: 'Outlook'
      };
      localMLService.predictWithNewAccess.mockResolvedValue(mockPrediction);

      const response = await request(app)
        .post('/api/record-app-access')
        .send({
          logs: [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }],
          appDisplayName: 'Outlook'
        })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.recorded_app).toBe('Outlook');
      expect(response.body.prediction).toBeDefined();
    });

    it('should return 400 when logs array is missing', async () => {
      const response = await request(app)
        .post('/api/record-app-access')
        .send({ appDisplayName: 'Outlook' })
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toContain('Logs array is required');
    });

    it('should return 400 when appDisplayName is missing', async () => {
      const response = await request(app)
        .post('/api/record-app-access')
        .send({ logs: [] })
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toContain('appDisplayName is required');
    });
  });

  describe('POST /api/feedback', () => {
    it('should store feedback successfully', async () => {
      const mockFeedbackId = 'feedback_123';
      kafkaService.storeFeedback.mockResolvedValue({ success: true, feedbackId: mockFeedbackId });

      const response = await request(app)
        .post('/api/feedback')
        .send({
          logs: [{ appDisplayName: 'Canvas' }],
          prediction: { next_pattern: 'Canvas → Outlook' },
          actualApp: 'Teams',
          userId: 'user_1'
        })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.feedbackId).toBe(mockFeedbackId);
      expect(response.body.message).toContain('Feedback stored');
    });

    it('should return 400 when required fields are missing', async () => {
      const response = await request(app)
        .post('/api/feedback')
        .send({ logs: [] })
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeDefined();
    });
  });

  describe('GET /api/feedback/stats', () => {
    it('should return feedback statistics', async () => {
      const mockStats = {
        total: 10,
        byPattern: { 'Canvas → Outlook': 5 },
        recent: []
      };
      kafkaService.getFeedbackStats.mockReturnValue(mockStats);

      const response = await request(app)
        .get('/api/feedback/stats')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.stats).toEqual(mockStats);
    });
  });
});

