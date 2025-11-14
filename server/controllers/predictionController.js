const localMLService = require('../services/localMLService');

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

