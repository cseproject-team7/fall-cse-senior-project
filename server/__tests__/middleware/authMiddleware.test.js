const jwt = require('jsonwebtoken');
const authMiddleware = require('../../middleware/authMiddleware');
const { createReq, createRes, createNext } = require('../utils/httpMocks');

const JWT_SECRET = 'test-jwt-secret-key';

describe('authMiddleware', () => {
  let originalEnv;

  beforeAll(() => {
    originalEnv = process.env;
    process.env = {
      ...originalEnv,
      JWT_SECRET
    };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('should return 401 when Authorization header is missing', () => {
    const req = createReq({ headers: {} });
    const res = createRes();
    const next = createNext();

    authMiddleware(req, res, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({
      success: false,
      message: 'Not authorized, no token'
    });
    expect(next).not.toHaveBeenCalled();
  });

  it('should return 401 when Authorization header does not start with Bearer', () => {
    const req = createReq({
      headers: { authorization: 'Basic some-token' }
    });
    const res = createRes();
    const next = createNext();

    authMiddleware(req, res, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({
      success: false,
      message: 'Not authorized, no token'
    });
    expect(next).not.toHaveBeenCalled();
  });

  it('should return 401 when token is invalid', () => {
    const req = createReq({
      headers: { authorization: 'Bearer invalid-token' }
    });
    const res = createRes();
    const next = createNext();

    authMiddleware(req, res, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({
      success: false,
      message: 'Not authorized, token failed'
    });
    expect(next).not.toHaveBeenCalled();
  });

  it('should return 401 with specific message when token is expired', () => {
    // Create an expired token
    const expiredToken = jwt.sign(
      { userId: 'admin', email: 'admin@usf.edu' },
      JWT_SECRET,
      { expiresIn: '-1h' } // Already expired
    );

    const req = createReq({
      headers: { authorization: `Bearer ${expiredToken}` }
    });
    const res = createRes();
    const next = createNext();

    authMiddleware(req, res, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({
      success: false,
      message: 'Token expired, please login again'
    });
    expect(next).not.toHaveBeenCalled();
  });

  it('should call next() and set req.user when token is valid', () => {
    const validToken = jwt.sign(
      { userId: 'admin', email: 'admin@usf.edu' },
      JWT_SECRET,
      { expiresIn: '1h' }
    );

    const req = createReq({
      headers: { authorization: `Bearer ${validToken}` }
    });
    const res = createRes();
    const next = createNext();

    authMiddleware(req, res, next);

    expect(next).toHaveBeenCalled();
    expect(req.user).toEqual({
      userId: 'admin',
      email: 'admin@usf.edu'
    });
    expect(res.status).not.toHaveBeenCalled();
    expect(res.json).not.toHaveBeenCalled();
  });

  it('should handle token with extra spaces after Bearer', () => {
    const validToken = jwt.sign(
      { userId: 'admin', email: 'admin@usf.edu' },
      JWT_SECRET,
      { expiresIn: '1h' }
    );

    const req = createReq({
      headers: { authorization: `Bearer  ${validToken}` } // Extra space
    });
    const res = createRes();
    const next = createNext();

    authMiddleware(req, res, next);

    // The split will create empty string at index 1, so this should fail
    expect(res.status).toHaveBeenCalledWith(401);
    expect(next).not.toHaveBeenCalled();
  });
});

