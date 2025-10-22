const azureMLService = require('../services/azureMLService');

exports.predict = async (req, res) => {
  try {
    if (!process.env.AZURE_ML_API_KEY) {
      return res.status(500).json({ 
        success: false,
        error: 'API key not configured'
      });
    }

    // Extract data array from request body
    const requestData = Array.isArray(req.body) 
      ? req.body 
      : req.body.data || req.body;

    const prediction = await azureMLService.predict(requestData);

    res.json({
      success: true,
      prediction,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Prediction error:', error.message);
    if (error.response) {
      console.error('Azure ML error response:', error.response.data);
    }
    res.status(error.response?.status || 500).json({
      success: false,
      error: error.response?.data || error.message,
      details: error.response?.data
    });
  }
};

