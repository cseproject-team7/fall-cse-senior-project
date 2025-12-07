# Backend Test Suite

This directory contains unit and integration tests for the USF Authentication Prediction backend server.

## Test Structure

```
__tests__/
├── controllers/           # Unit tests for controllers
│   ├── authController.test.js
│   ├── logsController.test.js
│   └── predictionController.test.js
├── middleware/            # Unit tests for middleware
│   └── authMiddleware.test.js
├── services/              # Unit tests for services
│   ├── eventHubService.test.js
│   ├── kafkaService.test.js
│   └── localMLService.test.js
├── integration/           # Integration tests for API routes
│   ├── api.public.test.js
│   └── api.protected.test.js
└── utils/                 # Test utilities and helpers
    ├── httpMocks.js
    └── testApp.js
```

## Test Types

### Unit Tests
Unit tests verify individual components (controllers, middleware, services) in isolation using mocks for dependencies.

**Coverage includes:**
- `authController`: Login logic, credential validation, JWT generation
- `authMiddleware`: Token verification, auth flow
- `logsController`: Persona and log retrieval, Azure Storage integration
- `predictionController`: ML predictions, feedback recording
- `localMLService`: ML server communication, payload formatting
- `kafkaService`: Feedback storage and statistics
- `eventHubService`: Mock log data retrieval

### Integration Tests
Integration tests verify the complete API contract using Supertest to make actual HTTP requests to the Express app.

**Coverage includes:**
- Public endpoints: `/api/health`, `/api/logs/personas`, `/api/predict`, etc.
- Protected endpoints: `/api/logs`, `/api/patterns`, `/api/pattern-chains`
- Authentication flow: Token validation, expiration handling
- Error handling: 400, 401, 500 responses

## Running Tests

### Prerequisites
```bash
# Install dependencies (from server directory)
npm install
```

### Run All Tests
```bash
npm test
```

### Run Tests in Watch Mode
```bash
npm run test:watch
```

### Run Tests with Coverage
```bash
npm run test:coverage
```

### Run Specific Test File
```bash
npm test -- authController.test.js
```

### Run Tests Matching Pattern
```bash
npm test -- --testNamePattern="should return 401"
```

## Environment Variables

Tests use mock environment variables. The following are set automatically in test files:

- `JWT_SECRET`: test-jwt-secret-key
- `ADMIN_USER`: admin@usf.edu
- `ADMIN_HASH`: bcrypt hash of test password
- `ML_SERVER_URL`: http://localhost:5001

## Test Coverage Goals

- **Controllers**: 85% lines, 85% branches
- **Middleware**: 90% lines, 90% branches
- **Services**: 80% lines, 80% branches
- **Overall**: 80% lines, 75% branches

View coverage report after running `npm run test:coverage`:
```bash
open coverage/lcov-report/index.html
```

## Mocking Strategy

### External Services
- **Azure Blob Storage**: Mocked with `@azure/storage-blob` mock
- **ML Server**: Mocked using `axios` mock with `nock` for specific cases
- **File System**: Mocked using `fs` mock for `kafkaService`
- **Event Hub**: Uses mock data from `eventHubService`

### Authentication
- Valid JWT tokens generated for protected route tests
- Expired tokens tested for proper expiration handling
- Invalid tokens tested for error responses

## Common Issues

### Tests failing with "Cannot find module"
Ensure all dependencies are installed:
```bash
cd server
npm install
```

### Tests timing out
Increase Jest timeout in individual tests:
```javascript
jest.setTimeout(10000); // 10 seconds
```

### Mock not resetting between tests
Ensure `jest.clearAllMocks()` is called in `beforeEach`:
```javascript
beforeEach(() => {
  jest.clearAllMocks();
});
```

## Adding New Tests

### Unit Test Template
```javascript
const { createReq, createRes } = require('../utils/httpMocks');
const myController = require('../../controllers/myController');

// Mock dependencies
jest.mock('../../services/myService');
const myService = require('../../services/myService');

describe('myController', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should do something', async () => {
    myService.doSomething.mockResolvedValue({ data: 'test' });
    
    const req = createReq({ body: { test: 'data' } });
    const res = createRes();

    await myController.myMethod(req, res);

    expect(res.json).toHaveBeenCalledWith({ success: true });
  });
});
```

### Integration Test Template
```javascript
const request = require('supertest');
const createTestApp = require('../utils/testApp');

describe('My API Endpoint', () => {
  let app;

  beforeAll(() => {
    app = createTestApp();
  });

  it('should return expected response', async () => {
    const response = await request(app)
      .get('/api/my-endpoint')
      .expect(200);

    expect(response.body.success).toBe(true);
  });
});
```

## CI/CD Integration

These tests are designed to run in CI/CD pipelines. Add to your pipeline:

```yaml
- name: Install dependencies
  run: cd server && npm install

- name: Run tests
  run: cd server && npm test

- name: Generate coverage
  run: cd server && npm run test:coverage

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./server/coverage/lcov.info
```

## Questions?

For questions or issues with tests, contact the Team 7 development team.

