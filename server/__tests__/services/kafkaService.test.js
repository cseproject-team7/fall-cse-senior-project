const fs = require('fs');
const path = require('path');
const kafkaService = require('../../services/kafkaService');

// Mock fs module
jest.mock('fs');

describe('kafkaService', () => {
  const mockFeedbackFile = path.join(__dirname, '../../feedback_data.json');
  let mockFeedbackData;

  beforeEach(() => {
    jest.clearAllMocks();
    mockFeedbackData = { feedback: [] };
    
    // Default mock implementations
    fs.existsSync.mockReturnValue(true);
    fs.readFileSync.mockReturnValue(JSON.stringify(mockFeedbackData));
    fs.writeFileSync.mockImplementation(() => {});
  });

  describe('storeFeedback', () => {
    it('should store feedback and return feedbackId', async () => {
      const feedbackData = {
        logs: [{ appDisplayName: 'Canvas', createdDateTime: '2025-01-01T09:00:00Z' }],
        prediction: { next_pattern: 'Canvas → Outlook', pattern_confidence: 92 },
        actualApp: 'Teams',
        userId: 'user_1',
        timestamp: '2025-01-01T10:00:00Z'
      };

      const result = await kafkaService.storeFeedback(feedbackData);

      expect(result.success).toBe(true);
      expect(result.feedbackId).toBeDefined();
      expect(result.feedbackId).toMatch(/^feedback_/);

      // Verify writeFileSync was called
      expect(fs.writeFileSync).toHaveBeenCalled();
      const writeCall = fs.writeFileSync.mock.calls[0];
      expect(writeCall[0]).toContain('feedback_data.json');
      
      // Parse the written data
      const writtenData = JSON.parse(writeCall[1]);
      expect(writtenData.feedback).toHaveLength(1);
      expect(writtenData.feedback[0]).toMatchObject(feedbackData);
      expect(writtenData.feedback[0].id).toBeDefined();
      expect(writtenData.feedback[0].createdAt).toBeDefined();
    });

    it('should append to existing feedback', async () => {
      mockFeedbackData.feedback = [
        {
          id: 'feedback_existing',
          logs: [],
          prediction: {},
          actualApp: 'Outlook',
          userId: 'user_2',
          timestamp: '2025-01-01T08:00:00Z',
          createdAt: '2025-01-01T08:00:00Z'
        }
      ];
      fs.readFileSync.mockReturnValue(JSON.stringify(mockFeedbackData));

      const newFeedback = {
        logs: [{ appDisplayName: 'Teams' }],
        prediction: { next_pattern: 'Teams → Canvas' },
        actualApp: 'Outlook',
        userId: 'user_3',
        timestamp: '2025-01-01T11:00:00Z'
      };

      await kafkaService.storeFeedback(newFeedback);

      const writeCall = fs.writeFileSync.mock.calls[0];
      const writtenData = JSON.parse(writeCall[1]);
      expect(writtenData.feedback).toHaveLength(2);
      expect(writtenData.feedback[0].id).toBe('feedback_existing');
      expect(writtenData.feedback[1].userId).toBe('user_3');
    });

    it('should handle errors gracefully', async () => {
      fs.readFileSync.mockImplementation(() => {
        throw new Error('File read error');
      });

      const feedbackData = {
        logs: [],
        prediction: {},
        actualApp: 'Canvas',
        userId: 'user_1',
        timestamp: '2025-01-01T10:00:00Z'
      };

      await expect(kafkaService.storeFeedback(feedbackData)).rejects.toThrow('File read error');
    });
  });

  describe('getAllFeedback', () => {
    it('should return all feedback entries', () => {
      mockFeedbackData.feedback = [
        { id: 'feedback_1', actualApp: 'Outlook' },
        { id: 'feedback_2', actualApp: 'Teams' }
      ];
      fs.readFileSync.mockReturnValue(JSON.stringify(mockFeedbackData));

      const result = kafkaService.getAllFeedback();

      expect(result).toHaveLength(2);
      expect(result[0].id).toBe('feedback_1');
      expect(result[1].id).toBe('feedback_2');
    });

    it('should return empty array on error', () => {
      fs.readFileSync.mockImplementation(() => {
        throw new Error('File not found');
      });

      const result = kafkaService.getAllFeedback();

      expect(result).toEqual([]);
    });
  });

  describe('getFeedbackStats', () => {
    it('should calculate statistics correctly', () => {
      mockFeedbackData.feedback = [
        {
          id: 'feedback_1',
          prediction: { next_pattern: 'Canvas → Outlook' },
          actualApp: 'Teams'
        },
        {
          id: 'feedback_2',
          prediction: { next_pattern: 'Canvas → Outlook' },
          actualApp: 'Outlook'
        },
        {
          id: 'feedback_3',
          prediction: { next_pattern: 'Teams → Canvas' },
          actualApp: 'Canvas'
        },
        {
          id: 'feedback_4',
          prediction: { next_pattern: 'Canvas → Outlook' },
          actualApp: 'Outlook'
        }
      ];
      fs.readFileSync.mockReturnValue(JSON.stringify(mockFeedbackData));

      const stats = kafkaService.getFeedbackStats();

      expect(stats.total).toBe(4);
      expect(stats.byPattern['Canvas → Outlook']).toBe(3);
      expect(stats.byPattern['Teams → Canvas']).toBe(1);
      expect(stats.recent).toHaveLength(4);
    });

    it('should handle missing prediction gracefully', () => {
      mockFeedbackData.feedback = [
        { id: 'feedback_1', actualApp: 'Outlook' }, // No prediction
        { id: 'feedback_2', prediction: {}, actualApp: 'Teams' } // Empty prediction
      ];
      fs.readFileSync.mockReturnValue(JSON.stringify(mockFeedbackData));

      const stats = kafkaService.getFeedbackStats();

      expect(stats.total).toBe(2);
      expect(stats.byPattern['unknown']).toBe(2);
    });

    it('should limit recent feedback to last 10 entries', () => {
      mockFeedbackData.feedback = Array.from({ length: 20 }, (_, i) => ({
        id: `feedback_${i}`,
        prediction: { next_pattern: 'Pattern' },
        actualApp: 'App'
      }));
      fs.readFileSync.mockReturnValue(JSON.stringify(mockFeedbackData));

      const stats = kafkaService.getFeedbackStats();

      expect(stats.recent).toHaveLength(10);
      expect(stats.recent[0].id).toBe('feedback_10'); // Last 10 items
    });

    it('should return default stats on error', () => {
      fs.readFileSync.mockImplementation(() => {
        throw new Error('Read error');
      });

      const stats = kafkaService.getFeedbackStats();

      expect(stats).toEqual({
        total: 0,
        byPattern: {},
        recent: []
      });
    });
  });
});

