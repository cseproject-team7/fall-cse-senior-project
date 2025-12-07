/**
 * Mock HTTP request and response objects for unit testing controllers
 */

/**
 * Create a mock Express response object
 */
exports.createRes = () => {
  const res = {};
  res.statusCode = 200;
  res.status = jest.fn((code) => {
    res.statusCode = code;
    return res;
  });
  res.json = jest.fn((body) => {
    res.body = body;
    return res;
  });
  res.send = jest.fn((body) => {
    res.body = body;
    return res;
  });
  return res;
};

/**
 * Create a mock Express request object
 */
exports.createReq = (overrides = {}) => {
  return {
    params: {},
    query: {},
    body: {},
    headers: {},
    ...overrides
  };
};

/**
 * Create a mock Express next function
 */
exports.createNext = () => jest.fn();

