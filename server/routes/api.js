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

