const logsController = require('../../controllers/logsController');
const eventHubService = require('../../services/eventHubService');
const { createReq, createRes } = require('../utils/httpMocks');

// Mock the eventHubService
jest.mock('../../services/eventHubService');

// Mock Azure Blob Storage
jest.mock('@azure/storage-blob', () => {
  return {
    BlobServiceClient: {
      fromConnectionString: jest.fn()
    }
  };
});

describe('logsController', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getPersonas', () => {
    it('should return success with personas array', async () => {
      const mockPersonas = ['user_1', 'user_2', 'user_3'];
      eventHubService.getPersonas.mockResolvedValue(mockPersonas);

      const req = createReq();
      const res = createRes();

      await logsController.getPersonas(req, res);

      expect(eventHubService.getPersonas).toHaveBeenCalled();
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        personas: mockPersonas
      });
    });

    it('should return 500 when service throws error', async () => {
      const errorMessage = 'Service unavailable';
      eventHubService.getPersonas.mockRejectedValue(new Error(errorMessage));

      const req = createReq();
      const res = createRes();

      await logsController.getPersonas(req, res);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: errorMessage
      });
    });
  });

  describe('getLogsByPersona', () => {
    it('should return logs for specified persona with default limit', async () => {
      const mockLogs = [
        { appDisplayName: 'Canvas', hour: 9, weekday: 1 },
        { appDisplayName: 'Outlook', hour: 10, weekday: 1 }
      ];
      eventHubService.getLogsByPersona.mockResolvedValue(mockLogs);

      const req = createReq({
        params: { persona: 'user_1' },
        query: {}
      });
      const res = createRes();

      await logsController.getLogsByPersona(req, res);

      expect(eventHubService.getLogsByPersona).toHaveBeenCalledWith('user_1', 50);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        persona: 'user_1',
        count: 2,
        logs: mockLogs
      });
    });

    it('should respect custom limit from query params', async () => {
      const mockLogs = [
        { appDisplayName: 'Canvas', hour: 9, weekday: 1 }
      ];
      eventHubService.getLogsByPersona.mockResolvedValue(mockLogs);

      const req = createReq({
        params: { persona: 'user_2' },
        query: { limit: '10' }
      });
      const res = createRes();

      await logsController.getLogsByPersona(req, res);

      expect(eventHubService.getLogsByPersona).toHaveBeenCalledWith('user_2', 10);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        persona: 'user_2',
        count: 1,
        logs: mockLogs
      });
    });

    it('should return 500 when service throws error', async () => {
      const errorMessage = 'Persona not found';
      eventHubService.getLogsByPersona.mockRejectedValue(new Error(errorMessage));

      const req = createReq({
        params: { persona: 'invalid_user' },
        query: {}
      });
      const res = createRes();

      await logsController.getLogsByPersona(req, res);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: errorMessage
      });
    });
  });

  describe('getAllLogs', () => {
    it('should return 503 when Azure Storage is not configured', async () => {
      // Since blobServiceClient is null when AZURE_STORAGE_CONNECTION_STRING is not set
      const req = createReq({ query: {} });
      const res = createRes();

      await logsController.getAllLogs(req, res);

      expect(res.status).toHaveBeenCalledWith(503);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: 'Azure Storage not configured'
      });
    });
  });

  describe('getLogPatterns', () => {
    it('should return 503 when Azure Storage is not configured', async () => {
      const req = createReq();
      const res = createRes();

      await logsController.getLogPatterns(req, res);

      expect(res.status).toHaveBeenCalledWith(503);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: 'Azure Storage not configured'
      });
    });
  });

  describe('getAllLogsGroupedByUser', () => {
    it('should return 503 when Azure Storage is not configured', async () => {
      const req = createReq();
      const res = createRes();

      await logsController.getAllLogsGroupedByUser(req, res);

      expect(res.status).toHaveBeenCalledWith(503);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: 'Azure Storage not configured'
      });
    });
  });

  describe('getPatternChains', () => {
    it('should return 500 with fallback when ML server is unavailable', async () => {
      const req = createReq();
      const res = createRes();

      await logsController.getPatternChains(req, res);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalled();
      const response = res.json.mock.calls[0][0];
      expect(response.message).toContain('ML server unavailable');
      expect(response.fallback).toEqual([]);
    });
  });
});

