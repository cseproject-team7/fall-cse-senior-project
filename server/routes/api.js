/*
const express = require('express');
const router = express.Router();
const predictionController = require('../controllers/predictionController');
const logsController = require('../controllers/logsController');
const mlPredictionController = require('../controllers/mlPredictionController');

// Health check
router.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// === Existing Logs endpoints (for ML/Persona) ===
router.get('/logs/personas', logsController.getPersonas);
router.get('/logs/:persona', logsController.getLogsByPersona);

// === NEW Analytics Dashboard endpoints ===
router.get('/logs', logsController.getAllLogs); // Fetches all logs
router.get('/patterns', logsController.getLogPatterns); // Fetches sequence patterns

// Prediction endpoint (Azure ML)
router.post('/predict', predictionController.predict);

// === ML Prediction endpoints (local models) ===
router.post('/predict-next-app', mlPredictionController.predictNextApp);
router.post('/predict-persona', mlPredictionController.predictPersona);

module.exports = router;
*/

const express = require('express');
const router = express.Router();

// --- Import all our controllers and middleware ---
const predictionController = require('../controllers/predictionController');
const logsController = require('../controllers/logsController');
const authController = require('../controllers/authController'); // <-- NEW
const authMiddleware = require('../middleware/authMiddleware'); // <-- NEW

// --- Auth Route ---
// This route is PUBLIC. Anyone can try to log in.
router.post('/auth/login', authController.login);

// --- Health check (Public) ---
router.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// --- ML/Persona Routes ---
// Temporarily removed authMiddleware for demo/testing purposes
router.get('/logs/personas', logsController.getPersonas);
router.get('/logs/:persona', logsController.getLogsByPersona);
router.get('/logs-grouped', logsController.getAllLogsGroupedByUser); // NEW: Get all logs grouped by user for predictions page
router.post('/predict', predictionController.predict);
router.post('/record-app-access', predictionController.recordAppAccess); // NEW: Record app access
router.post('/feedback', predictionController.submitFeedback); // NEW: Submit incorrect prediction feedback
router.get('/feedback/stats', predictionController.getFeedbackStats); // NEW: Get feedback stats

// --- Analytics Dashboard Routes ---
// These are now also protected by 'authMiddleware'
router.get('/logs', authMiddleware, logsController.getAllLogs);
router.get('/patterns', authMiddleware, logsController.getLogPatterns);

module.exports = router;