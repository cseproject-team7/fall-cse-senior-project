/*
const express = require('express');
const router = express.Router();
const predictionController = require('../controllers/predictionController');
const logsController = require('../controllers/logsController');

// Health check
router.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Logs endpoints
router.get('/logs/personas', logsController.getPersonas);
router.get('/logs/:persona', logsController.getLogsByPersona);

// Prediction endpoint
router.post('/predict', predictionController.predict);

module.exports = router;
*/

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