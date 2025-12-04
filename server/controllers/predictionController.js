const localMLService = require('../services/localMLService');
const kafkaService = require('../services/kafkaService');

exports.predict = async (req, res) => {
  try {
    // Extract data array from request body
    const requestData = Array.isArray(req.body) 
      ? req.body 
      : req.body.data || req.body;

    // Use local ML service instead of Azure ML
    const prediction = await localMLService.predict(requestData);

    res.json({
      success: true,
      prediction,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Prediction error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

exports.recordAppAccess = async (req, res) => {
  try {
    const { logs, appDisplayName } = req.body;

    if (!logs || !Array.isArray(logs)) {
      return res.status(400).json({
        success: false,
        error: 'Logs array is required'
      });
    }

    if (!appDisplayName) {
      return res.status(400).json({
        success: false,
        error: 'appDisplayName is required'
      });
    }

    // Record the app access and get updated predictions
    const prediction = await localMLService.predictWithNewAccess(logs, appDisplayName);

    res.json({
      success: true,
      prediction,
      recorded_app: appDisplayName,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Record app access error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

exports.submitFeedback = async (req, res) => {
  try {
    const { logs, prediction, actualApp, userId } = req.body;

    if (!logs || !prediction || !actualApp) {
      return res.status(400).json({
        success: false,
        error: 'logs, prediction, and actualApp are required'
      });
    }

    // Store feedback for model retraining
    const feedbackId = await kafkaService.storeFeedback({
      logs,
      prediction,
      actualApp,
      userId: userId || 'anonymous',
      timestamp: new Date().toISOString()
    });

    res.json({
      success: true,
      feedbackId,
      message: 'Feedback stored for model retraining'
    });

  } catch (error) {
    console.error('Submit feedback error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

exports.getFeedbackStats = async (req, res) => {
  try {
    const stats = await kafkaService.getFeedbackStats();
    
    res.json({
      success: true,
      stats
    });

  } catch (error) {
    console.error('Get feedback stats error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};
