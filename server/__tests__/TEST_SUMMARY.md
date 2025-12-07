# Test Suite Summary

## Overview

This test suite contains **both unit and integration tests** for the USF Authentication Prediction backend server. All tests follow Jest conventions and use appropriate mocking strategies.

---

## Unit Tests

Unit tests verify individual components in **isolation** by mocking all external dependencies. They test the internal logic of a single module without making real network calls or database operations.

### 1. Controllers (Unit Tests)

#### `authController.test.js` - **UNIT TEST**
Tests the authentication controller's business logic:
- ✅ Returns 400 when email/password missing
- ✅ Returns 401 for invalid credentials
- ✅ Returns 200 with JWT token for valid credentials
- ✅ Handles case-insensitive email matching
- ✅ Returns 500 on server errors

**Mocking:** Uses mocked `bcrypt` and `jwt` modules

---

#### `logsController.test.js` - **UNIT TEST**
Tests log retrieval controller logic:
- ✅ Returns personas from service
- ✅ Returns logs filtered by persona
- ✅ Respects pagination parameters
- ✅ Returns 503 when Azure Storage not configured
- ✅ Handles service errors gracefully

**Mocking:** Mocks `eventHubService` and `@azure/storage-blob`

---

#### `predictionController.test.js` - **UNIT TEST**
Tests prediction and feedback controller logic:
- ✅ Handles prediction requests with various payload formats
- ✅ Validates required fields for recording app access
- ✅ Stores feedback with correct structure
- ✅ Returns feedback statistics
- ✅ Uses 'anonymous' userId when not provided

**Mocking:** Mocks `localMLService` and `kafkaService`

---

### 2. Middleware (Unit Tests)

#### `authMiddleware.test.js` - **UNIT TEST**
Tests JWT authentication middleware:
- ✅ Returns 401 when Authorization header missing
- ✅ Returns 401 when token format is invalid
- ✅ Returns 401 with specific message for expired tokens
- ✅ Calls `next()` and sets `req.user` for valid tokens
- ✅ Rejects tokens with invalid signatures

**Mocking:** Uses real `jwt` module with test secrets

---

### 3. Services (Unit Tests)

#### `localMLService.test.js` - **UNIT TEST**
Tests ML service communication logic:
- ✅ Formats payload correctly for local vs Azure ML endpoints
- ✅ Parses JSON string responses from Azure ML
- ✅ Handles `predictWithNewAccess` with new app access field
- ✅ Validates inputs for `predictPersona`
- ✅ Throws errors when ML response indicates failure

**Mocking:** Mocks `axios` for HTTP requests

---

#### `kafkaService.test.js` - **UNIT TEST**
Tests feedback storage service logic:
- ✅ Stores feedback with generated ID and timestamp
- ✅ Appends to existing feedback entries
- ✅ Returns all feedback entries
- ✅ Calculates statistics (total, by pattern, recent)
- ✅ Handles file read/write errors gracefully

**Mocking:** Mocks `fs` module

---

#### `eventHubService.test.js` - **UNIT TEST**
Tests mock event hub data retrieval:
- ✅ Returns consistent array of personas
- ✅ Returns logs with valid structure (appDisplayName, hour, weekday)
- ✅ Respects limit parameter
- ✅ Returns empty array for invalid persona
- ✅ Validates data integrity (hour 0-23, weekday 0-6)

**Mocking:** Uses built-in mock data, no external mocks needed

---

## Integration Tests

Integration tests verify the **complete API contract** by making real HTTP requests through the Express app. They test the full request-response cycle including routing, middleware, and controller integration.

### 1. Public API Routes

#### `api.public.test.js` - **INTEGRATION TEST**
Tests publicly accessible endpoints without authentication:

**GET /api/health**
- ✅ Returns 200 with status 'ok'
- ✅ Includes valid ISO timestamp

**GET /api/logs/personas**
- ✅ Returns list of personas
- ✅ Returns 500 when service fails

**GET /api/logs/:persona**
- ✅ Returns logs for specified persona
- ✅ Respects limit query parameter
- ✅ Uses default limit of 50 when not specified

**POST /api/predict**
- ✅ Accepts `{ data: [...] }` format
- ✅ Accepts direct array format `[...]`
- ✅ Returns prediction with timestamp
- ✅ Returns 500 when ML service fails

**POST /api/record-app-access**
- ✅ Returns 400 when logs array missing
- ✅ Returns 400 when appDisplayName missing
- ✅ Records app access and returns updated prediction

**POST /api/feedback**
- ✅ Stores feedback with feedbackId
- ✅ Returns 400 when required fields missing

**GET /api/feedback/stats**
- ✅ Returns feedback statistics

**Mocking:** Services are mocked, but routing and middleware are real

---

### 2. Protected API Routes

#### `api.protected.test.js` - **INTEGRATION TEST**
Tests endpoints requiring JWT authentication:

**GET /api/logs** (Protected)
- ✅ Returns 401 without token
- ✅ Returns 401 with invalid token
- ✅ Returns 401 with malformed Authorization header
- ✅ Returns 503 with valid token (auth succeeds, storage unavailable in tests)

**GET /api/patterns** (Protected)
- ✅ Returns 401 without token
- ✅ Accepts valid token

**GET /api/pattern-chains** (Protected)
- ✅ Returns 401 without token
- ✅ Returns 500 with valid token when ML server unavailable

**Token Expiration**
- ✅ Returns 401 with specific message for expired tokens

**Public Routes Should Not Require Auth**
- ✅ `/api/health` works without token
- ✅ `/api/logs/personas` works without token
- ✅ `/api/predict` works without token

**Mocking:** Services are mocked, but full Express app routing and auth middleware are real

---

## Test Classification Summary

| Test File | Type | What It Tests |
|-----------|------|---------------|
| `authController.test.js` | **Unit** | Controller logic in isolation |
| `authMiddleware.test.js` | **Unit** | Middleware logic in isolation |
| `logsController.test.js` | **Unit** | Controller logic in isolation |
| `predictionController.test.js` | **Unit** | Controller logic in isolation |
| `localMLService.test.js` | **Unit** | Service logic in isolation |
| `kafkaService.test.js` | **Unit** | Service logic in isolation |
| `eventHubService.test.js` | **Unit** | Service logic in isolation |
| `api.public.test.js` | **Integration** | Full HTTP API contract (public routes) |
| `api.protected.test.js` | **Integration** | Full HTTP API contract (auth-protected routes) |

---

## Key Differences

### Unit Tests
- ✅ Test **single module** in isolation
- ✅ Mock **all dependencies** (services, middleware, external APIs)
- ✅ Use `httpMocks.js` utility to create mock req/res objects
- ✅ Fast execution (no HTTP overhead)
- ✅ Focused on internal logic and edge cases

### Integration Tests
- ✅ Test **complete request-response cycle**
- ✅ Use **Supertest** to make real HTTP requests
- ✅ Test **routing + middleware + controllers together**
- ✅ Mock only external services (Azure, ML server)
- ✅ Verify API contract and status codes
- ✅ Test authentication flow end-to-end

---

## Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| Controllers | 85% lines, 85% branches |
| Middleware | 90% lines, 90% branches |
| Services | 80% lines, 80% branches |
| **Overall** | **80% lines, 75% branches** |

---

## Running Tests

```bash
# Run all tests (unit + integration)
npm test

# Run only unit tests
npm test -- controllers services middleware

# Run only integration tests
npm test -- integration

# Run with coverage
npm run test:coverage
```

---

## Answer to Your Question

**Q: Are the above tests unit or integration, or both?**

**A: BOTH!**

- **Unit Tests (7 files)**: Test individual controllers, middleware, and services in isolation with all dependencies mocked
- **Integration Tests (2 files)**: Test the complete Express API using Supertest to verify the full HTTP request-response cycle

The test suite provides comprehensive coverage at both the **component level** (unit) and **system level** (integration).

