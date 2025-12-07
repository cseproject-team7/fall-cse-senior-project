const predictionController = require('../../controllers/predictionController');
const localMLService = require('../../services/localMLService');
const kafkaService = require('../../services/kafkaService');
const { createReq, createRes } = require('../utils/httpMocks');

// Mock the services
jest.mock('../../services/localMLService');
jest.mock('../../services/kafkaService');

describe('predictionController', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('predict', () => {
    const mockPrediction = {
      success: true,
      next_pattern: 'Canvas → Outlook',
      pattern_confidence: 92,
      next_apps: ['Outlook', 'Teams'],
      predictions: [{ pattern: 'Canvas → Outlook', confidence: 92 }],
      sessions_analyzed: 15
    };

    it('should handle array body and return prediction', async () => {
      localMLService.predict.mockResolvedValue(mockPrediction);

      const req = createReq({
        body: [
          { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }
        ]
      });
      const res = createRes();

      await predictionController.predict(req, res);

      expect(localMLService.predict).toHaveBeenCalledWith(req.body);
      expect(res.json).toHaveBeenCalled();
      const response = res.json.mock.calls[0][0];
      expect(response.success).toBe(true);
      expect(response.prediction).toEqual(mockPrediction);
      expect(response.timestamp).toBeDefined();
    });

    it('should handle body with data field', async () => {
      localMLService.predict.mockResolvedValue(mockPrediction);

      const logsArray = [
        { appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }
      ];
      const req = createReq({
        body: { data: logsArray }
      });
      const res = createRes();

      await predictionController.predict(req, res);

      expect(localMLService.predict).toHaveBeenCalledWith(logsArray);
      expect(res.json).toHaveBeenCalled();
      const response = res.json.mock.calls[0][0];
      expect(response.success).toBe(true);
    });

    it('should return 500 when service throws error', async () => {
      const errorMessage = 'ML server timeout';
      localMLService.predict.mockRejectedValue(new Error(errorMessage));

      const req = createReq({
        body: [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }]
      });
      const res = createRes();

      await predictionController.predict(req, res);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: errorMessage
      });
    });
  });

  describe('recordAppAccess', () => {
    const mockPrediction = {
      success: true,
      next_pattern: 'Outlook → Teams',
      recorded_app: 'Outlook'
    };

    it('should return 400 when logs array is missing', async () => {
      const req = createReq({
        body: { appDisplayName: 'Outlook' }
      });
      const res = createRes();

      await predictionController.recordAppAccess(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: 'Logs array is required'
      });
    });

    it('should return 400 when logs is not an array', async () => {
      const req = createReq({
        body: { logs: 'not-an-array', appDisplayName: 'Outlook' }
      });
      const res = createRes();

      await predictionController.recordAppAccess(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: 'Logs array is required'
      });
    });

    it('should return 400 when appDisplayName is missing', async () => {
      const req = createReq({
        body: { logs: [] }
      });
      const res = createRes();

      await predictionController.recordAppAccess(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: 'appDisplayName is required'
      });
    });

    it('should record app access and return updated prediction', async () => {
      localMLService.predictWithNewAccess.mockResolvedValue(mockPrediction);

      const req = createReq({
        body: {
          logs: [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }],
          appDisplayName: 'Outlook'
        }
      });
      const res = createRes();

      await predictionController.recordAppAccess(req, res);

      expect(localMLService.predictWithNewAccess).toHaveBeenCalledWith(
        req.body.logs,
        'Outlook'
      );
      expect(res.json).toHaveBeenCalled();
      const response = res.json.mock.calls[0][0];
      expect(response.success).toBe(true);
      expect(response.prediction).toEqual(mockPrediction);
      expect(response.recorded_app).toBe('Outlook');
      expect(response.timestamp).toBeDefined();
    });

    it('should return 500 when service throws error', async () => {
      localMLService.predictWithNewAccess.mockRejectedValue(new Error('Service error'));

      const req = createReq({
        body: {
          logs: [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }],
          appDisplayName: 'Outlook'
        }
      });
      const res = createRes();

      await predictionController.recordAppAccess(req, res);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalled();
    });
  });

  describe('submitFeedback', () => {
    it('should return 400 when required fields are missing', async () => {
      const req = createReq({
        body: { logs: [] }
      });
      const res = createRes();

      await predictionController.submitFeedback(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: 'logs, prediction, and actualApp are required'
      });
    });

    it('should store feedback and return feedbackId', async () => {
      const mockFeedbackId = 'feedback_123';
      kafkaService.storeFeedback.mockResolvedValue({ success: true, feedbackId: mockFeedbackId });

      const req = createReq({
        body: {
          logs: [{ appDisplayName: 'Canvas' }],
          prediction: { next_pattern: 'Canvas → Outlook' },
          actualApp: 'Teams',
          userId: 'user_1'
        }
      });
      const res = createRes();

      await predictionController.submitFeedback(req, res);

      expect(kafkaService.storeFeedback).toHaveBeenCalledWith({
        logs: req.body.logs,
        prediction: req.body.prediction,
        actualApp: req.body.actualApp,
        userId: 'user_1',
        timestamp: expect.any(String)
      });
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        feedbackId: mockFeedbackId,
        message: 'Feedback stored for model retraining'
      });
    });

    it('should use anonymous userId when not provided', async () => {
      kafkaService.storeFeedback.mockResolvedValue({ success: true, feedbackId: 'feedback_456' });

      const req = createReq({
        body: {
          logs: [{ appDisplayName: 'Canvas' }],
          prediction: { next_pattern: 'Canvas → Outlook' },
          actualApp: 'Teams'
        }
      });
      const res = createRes();

      await predictionController.submitFeedback(req, res);

      expect(kafkaService.storeFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          userId: 'anonymous'
        })
      );
    });

    it('should return 500 when service throws error', async () => {
      kafkaService.storeFeedback.mockRejectedValue(new Error('Storage error'));

      const req = createReq({
        body: {
          logs: [{ appDisplayName: 'Canvas' }],
          prediction: { next_pattern: 'Canvas → Outlook' },
          actualApp: 'Teams'
        }
      });
      const res = createRes();

      await predictionController.submitFeedback(req, res);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalled();
    });
  });

  describe('getFeedbackStats', () => {
    it('should return feedback statistics', async () => {
      const mockStats = {
        total: 10,
        byPattern: { 'Canvas → Outlook': 5, 'Teams → Canvas': 3 },
        recent: []
      };
      kafkaService.getFeedbackStats.mockReturnValue(mockStats);

      const req = createReq();
      const res = createRes();

      await predictionController.getFeedbackStats(req, res);

      expect(kafkaService.getFeedbackStats).toHaveBeenCalled();
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        stats: mockStats
      });
    });

    it('should return 500 when service throws error', async () => {
      kafkaService.getFeedbackStats.mockImplementation(() => {
        throw new Error('Stats error');
      });

      const req = createReq();
      const res = createRes();

      await predictionController.getFeedbackStats(req, res);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalled();
    });
  });
});

