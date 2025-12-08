const authController = require('../../controllers/authController');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { createReq, createRes } = require('../utils/httpMocks');

// Mock environment variables for tests (independent of .env file)
const ADMIN_USER = 'admin@usf.edu';
const ADMIN_PASSWORD = 'TestPassword123!';
const JWT_SECRET = 'test-jwt-secret-key-for-unit-tests';

describe('authController', () => {
  let originalEnv;

  beforeAll(() => {
    originalEnv = process.env;
    // Create a bcrypt hash for the test password
    const ADMIN_HASH = bcrypt.hashSync(ADMIN_PASSWORD, 10);
    process.env = {
      ...originalEnv,
      ADMIN_USER,
      ADMIN_HASH,
      JWT_SECRET
    };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  describe('login', () => {
    it('should return 400 when email is missing', async () => {
      const req = createReq({ body: { password: 'test' } });
      const res = createRes();

      await authController.login(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        message: 'Please provide email and password'
      });
    });

    it('should return 400 when password is missing', async () => {
      const req = createReq({ body: { email: 'admin@usf.edu' } });
      const res = createRes();

      await authController.login(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        message: 'Please provide email and password'
      });
    });

    it('should return 401 when email does not match ADMIN_USER', async () => {
      const req = createReq({
        body: { email: 'wrong@usf.edu', password: ADMIN_PASSWORD }
      });
      const res = createRes();

      await authController.login(req, res);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        message: 'Invalid credentials'
      });
    });

    it('should return 401 when password is incorrect', async () => {
      const req = createReq({
        body: { email: ADMIN_USER, password: 'WrongPassword' }
      });
      const res = createRes();

      await authController.login(req, res);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        message: 'Invalid credentials'
      });
    });

    it('should return 200 with token when credentials are valid', async () => {
      const req = createReq({
        body: { email: ADMIN_USER, password: ADMIN_PASSWORD }
      });
      const res = createRes();

      await authController.login(req, res);

      expect(res.json).toHaveBeenCalled();
      const response = res.json.mock.calls[0][0];
      expect(response.success).toBe(true);
      expect(response.message).toBe('Login successful');
      expect(response.token).toBeDefined();

      // Verify token is valid and contains correct data
      const decoded = jwt.verify(response.token, JWT_SECRET);
      expect(decoded.userId).toBe('admin');
      expect(decoded.email).toBe(ADMIN_USER);
      expect(decoded.exp).toBeDefined();
    });

    it('should handle email case insensitively', async () => {
      const req = createReq({
        body: { email: 'ADMIN@USF.EDU', password: ADMIN_PASSWORD }
      });
      const res = createRes();

      await authController.login(req, res);

      expect(res.json).toHaveBeenCalled();
      const response = res.json.mock.calls[0][0];
      expect(response.success).toBe(true);
      expect(response.token).toBeDefined();
    });

    it('should return 500 when an error occurs', async () => {
      // Temporarily break the environment to cause an error
      const tempEnv = process.env.JWT_SECRET;
      delete process.env.JWT_SECRET;

      const req = createReq({
        body: { email: ADMIN_USER, password: ADMIN_PASSWORD }
      });
      const res = createRes();

      await authController.login(req, res);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        message: 'Server error during login'
      });

      // Restore environment
      process.env.JWT_SECRET = tempEnv;
    });
  });
});

