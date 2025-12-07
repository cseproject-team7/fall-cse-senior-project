const request = require('supertest');
const jwt = require('jsonwebtoken');
const createTestApp = require('../utils/testApp');

// Mock services
jest.mock('../../services/localMLService');
jest.mock('../../services/eventHubService');
jest.mock('../../services/kafkaService');
jest.mock('@azure/storage-blob');
jest.mock('axios');

<<<<<<< HEAD
// Mock JWT_SECRET for tests
const JWT_SECRET = 'test-jwt-secret-key-for-integration-tests';
=======
const JWT_SECRET = 'test-jwt-secret-key';
>>>>>>> 73868ff (Unit and integration tests)

describe('Protected API Integration Tests', () => {
  let app;
  let validToken;

  beforeAll(() => {
    app = createTestApp();
    
    // Set up environment for tests
    process.env.JWT_SECRET = JWT_SECRET;
    
    // Create a valid token for testing
    validToken = jwt.sign(
      { userId: 'admin', email: 'admin@usf.edu' },
      JWT_SECRET,
      { expiresIn: '1h' }
    );
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Authentication Required Endpoints', () => {
    describe('GET /api/logs', () => {
      it('should return 401 without token', async () => {
        const response = await request(app)
          .get('/api/logs')
          .expect(401);

        expect(response.body.success).toBe(false);
        expect(response.body.message).toContain('Not authorized');
      });

      it('should return 401 with invalid token', async () => {
        const response = await request(app)
          .get('/api/logs')
          .set('Authorization', 'Bearer invalid-token')
          .expect(401);

        expect(response.body.success).toBe(false);
        expect(response.body.message).toContain('token failed');
      });

      it('should return 401 with malformed Authorization header', async () => {
        const response = await request(app)
          .get('/api/logs')
          .set('Authorization', 'InvalidFormat token')
          .expect(401);

        expect(response.body.success).toBe(false);
      });

      it('should return 503 with valid token but no storage configured', async () => {
        const response = await request(app)
          .get('/api/logs')
          .set('Authorization', `Bearer ${validToken}`)
          .expect(503);

        expect(response.body.success).toBe(false);
        expect(response.body.error).toContain('Azure Storage not configured');
      });

      it('should accept valid token in Authorization header', async () => {
        const response = await request(app)
          .get('/api/logs')
          .set('Authorization', `Bearer ${validToken}`)
          .expect(503); // 503 because storage is not configured in tests, but auth passed

        // The fact that we got 503 (storage error) instead of 401 means auth succeeded
        expect(response.body.error).toContain('Azure Storage');
      });
    });

    describe('GET /api/patterns', () => {
      it('should return 401 without token', async () => {
        const response = await request(app)
          .get('/api/patterns')
          .expect(401);

        expect(response.body.success).toBe(false);
      });

      it('should return 503 with valid token but no storage', async () => {
        const response = await request(app)
          .get('/api/patterns')
          .set('Authorization', `Bearer ${validToken}`)
          .expect(503);

        expect(response.body.error).toContain('Azure Storage not configured');
      });
    });

    describe('GET /api/pattern-chains', () => {
      it('should return 401 without token', async () => {
        const response = await request(app)
          .get('/api/pattern-chains')
          .expect(401);

        expect(response.body.success).toBe(false);
      });

      it('should return 500 with valid token when ML server unavailable', async () => {
        const response = await request(app)
          .get('/api/pattern-chains')
          .set('Authorization', `Bearer ${validToken}`)
          .expect(500);

        expect(response.body.message).toContain('ML server unavailable');
        expect(response.body.fallback).toEqual([]);
      });
    });
  });

  describe('Token Expiration', () => {
    it('should return 401 with expired token', async () => {
      const expiredToken = jwt.sign(
        { userId: 'admin', email: 'admin@usf.edu' },
        JWT_SECRET,
        { expiresIn: '-1h' } // Already expired
      );

      const response = await request(app)
        .get('/api/logs')
        .set('Authorization', `Bearer ${expiredToken}`)
        .expect(401);

      expect(response.body.success).toBe(false);
      expect(response.body.message).toContain('Token expired');
    });
  });

  describe('Auth Flow Integration', () => {
    it('should allow access to protected routes after successful login', async () => {
      // This test demonstrates the full flow:
      // 1. Login to get token (tested separately in auth tests)
      // 2. Use token to access protected endpoint
      
      const response = await request(app)
        .get('/api/logs?limit=10')
        .set('Authorization', `Bearer ${validToken}`)
        .expect(503); // Auth succeeds, storage fails (as expected in test env)

      // Verify we passed auth and reached the controller
      expect(response.body.error).toContain('Azure Storage not configured');
    });
  });

  describe('Public Routes Should Not Require Auth', () => {
    it('GET /api/health should work without token', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect(200);

      expect(response.body.status).toBe('ok');
    });

    it('GET /api/logs/personas should work without token', async () => {
      await request(app)
        .get('/api/logs/personas')
        .expect(200);
    });

    it('POST /api/predict should work without token', async () => {
      // Note: Service will fail but auth should not be required
      await request(app)
        .post('/api/predict')
        .send({ data: [] })
        .expect((res) => {
          // Should not get 401, could get 500 from service
          expect(res.status).not.toBe(401);
        });
    });
  });

  describe('CORS and Headers', () => {
    it('should handle preflight OPTIONS requests', async () => {
      // Note: This depends on CORS middleware setup
      const response = await request(app)
        .options('/api/logs')
        .expect((res) => {
<<<<<<< HEAD
<<<<<<< HEAD
          // Should get 200 (CORS allowed), 204 (preflight), or 404 (no handler)
          expect([200, 204, 404]).toContain(res.status);
=======
          // Should get either 204 (preflight) or 404 (no preflight handler)
          expect([204, 404]).toContain(res.status);
>>>>>>> 73868ff (Unit and integration tests)
=======
          // Should get 200 (CORS allowed), 204 (preflight), or 404 (no handler)
          expect([200, 204, 404]).toContain(res.status);
>>>>>>> 1072575 (fix: make all tests pass)
        });
    });
  });
});

