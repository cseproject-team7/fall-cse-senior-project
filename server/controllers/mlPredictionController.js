const localMLService = require('../services/localMLService');

// Next App Predictor endpoint
exports.predictNextApp = async (req, res) => {
  try {
    const { data } = req.body;

    if (!data || !Array.isArray(data)) {
      return res.status(400).json({
        error: 'Invalid input',
        message: 'Expected "data" field with array of app access records'
      });
    }

    // Validate data structure
    for (const record of data) {
      if (!record.appDisplayName || record.hour === undefined || record.weekday === undefined) {
        return res.status(400).json({
          error: 'Invalid record format',
          message: 'Each record must have appDisplayName, hour, and weekday'
        });
      }
    }

    const result = await localMLService.predictNextApp(data);

    res.json({
      success: true,
      ...result,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
};

// Persona Classifier endpoint
exports.predictPersona = async (req, res) => {
  try {
    const { userId, records } = req.body;

    if (!userId || !records || !Array.isArray(records)) {
      return res.status(400).json({
        error: 'Invalid input',
        message: 'Expected "userId" and "records" (array of app access records)'
      });
    }

    if (records.length < 5) {
      return res.status(400).json({
        error: 'Insufficient data',
        message: 'Persona classification requires at least 5 app access records'
      });
    }

    // Validate record structure
    for (const record of records) {
      if (!record.appDisplayName || !record.createdDateTime) {
        return res.status(400).json({
          error: 'Invalid record format',
          message: 'Each record must have appDisplayName and createdDateTime'
        });
      }
    }

    res.json({
      success: true,
      ...result,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
};

// Evaluate model accuracy endpoint
exports.evaluateAccuracy = async (req, res) => {
  try {
    const { userId } = req.body;

    if (!userId) {
      return res.status(400).json({
        error: 'Invalid input',
        message: 'Expected "userId" to evaluate accuracy'
      });
    }

    const result = await localMLService.evaluateAccuracy(userId);

    res.json({
      success: true,
      ...result,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Accuracy evaluation error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
};
