const axios = require('axios');

// Main prediction function - calls dual-head LSTM server
exports.predict = async (data) => {
  try {
    // Read configuration dynamically (for testing)
    const ML_SERVER_URL = process.env.ML_SERVER_URL || 'http://localhost:5001';
    const AZURE_ML_KEY = process.env.AZURE_ML_KEY || '';
    const isAzureML = ML_SERVER_URL.includes('inference.ml.azure.com');

    // Ensure data is an array
    let payload = data;
    if (!Array.isArray(data)) {
      payload = data.data || [data];
    }

    console.log('🔵 Sending to Dual-Head LSTM server:', JSON.stringify(payload.slice(0, 3), null, 2), '...');
    
    // Determine endpoint URL (Azure ML uses /score, local uses /predict)
    const endpointUrl = isAzureML ? ML_SERVER_URL : `${ML_SERVER_URL}/predict`;
    console.log('🔵 ML Server URL:', endpointUrl);

    // Convert logs to format expected by ml_server_dual
    const logs = payload.map(log => ({
      appDisplayName: log.appDisplayName,
      createdDateTime: log.createdDateTime
    }));

    const startTime = Date.now();

    // Prepare headers
    const headers = { 'Content-Type': 'application/json' };
    if (isAzureML && AZURE_ML_KEY) {
      headers['Authorization'] = `Bearer ${AZURE_ML_KEY}`;
    }

    // Call the dual-head LSTM server
    const response = await axios.post(
      endpointUrl,
      { 
        logs: logs,
        num_predictions: 10  // Get 10 future pattern predictions
      },
      {
        headers,
        timeout: 30000
      }
    );

    const elapsed = Date.now() - startTime;
    console.log(`⏱️  ML server responded in ${elapsed}ms`);
    console.log('✅ ML response:', JSON.stringify(response.data, null, 2));

    // Parse response if it's a string (Azure ML returns JSON string)
    let result = response.data;
    if (typeof result === 'string') {
      result = JSON.parse(result);
    }

    if (!result.success) {
      throw new Error(result.error || 'Prediction failed');
    }

    // Format response for frontend
    // Return both the multi-step predictions and the immediate next prediction
    return {
      success: true,
      next_pattern: result.predictions[0].pattern,
      pattern_confidence: result.predictions[0].confidence,
      next_apps: result.predictions[0].top_apps,
      predictions: result.predictions,  // All 10 predictions
      sessions_analyzed: result.sessions_analyzed
    };

  } catch (error) {
    console.error('❌ ML server error:', error.message);
    if (error.response) {
      console.error('❌ ML server response:', error.response.data);
    }
    throw error;
  }
};

// Legacy function for backward compatibility
exports.predictNextApp = exports.predict;

// Predict with new app access - adds app to history and re-predicts
exports.predictWithNewAccess = async (logs, appDisplayName) => {
  try {
    // Read configuration dynamically (for testing)
    const ML_SERVER_URL = process.env.ML_SERVER_URL || 'http://localhost:5001';
    const AZURE_ML_KEY = process.env.AZURE_ML_KEY || '';
    const isAzureML = ML_SERVER_URL.includes('inference.ml.azure.com');

    console.log('📝 Recording new app access:', appDisplayName);
    
    // Convert logs to proper format
    let logData = Array.isArray(logs) ? logs : [logs];
    
    // Prepare headers
    const headers = { 'Content-Type': 'application/json' };
    if (isAzureML && AZURE_ML_KEY) {
      headers['Authorization'] = `Bearer ${AZURE_ML_KEY}`;
    }

    // Determine endpoint URL
    const endpointUrl = isAzureML ? ML_SERVER_URL : `${ML_SERVER_URL}/predict`;

    // Call ML server with new app access
    const response = await axios.post(
      endpointUrl,
      { 
        logs: logData.map(log => ({
          appDisplayName: log.appDisplayName,
          createdDateTime: log.createdDateTime
        })),
        num_predictions: 10,
        new_app_access: {
          appDisplayName: appDisplayName,
          createdDateTime: new Date().toISOString()
        }
      },
      {
        headers,
        timeout: 30000
      }
    );

    // Parse response if it's a string (Azure ML returns JSON string)
    let result = response.data;
    if (typeof result === 'string') {
      result = JSON.parse(result);
    }

    if (!result.success) {
      throw new Error(result.error || 'Prediction failed');
    }

    console.log('✅ Updated predictions after recording:', result.predictions[0].pattern);

    return {
      success: true,
      next_pattern: result.predictions[0].pattern,
      pattern_confidence: result.predictions[0].confidence,
      next_apps: result.predictions[0].top_apps,
      predictions: result.predictions,
      sessions_analyzed: result.sessions_analyzed,
      logs_used: result.logs_used,
      recorded_app: appDisplayName
    };

  } catch (error) {
    console.error('❌ Error recording app access:', error.message);
    if (error.response) {
      console.error('❌ ML server response:', error.response.data);
    }
    throw error;
  }
};

// Persona Classifier - direct call to ML server
exports.predictPersona = async (userId, records) => {
  try {
    // Read configuration dynamically (for testing)
    const ML_SERVER_URL = process.env.ML_SERVER_URL || 'http://localhost:5001';
    const AZURE_ML_KEY = process.env.AZURE_ML_KEY || '';
    const isAzureML = ML_SERVER_URL.includes('inference.ml.azure.com');

    if (!userId || !records || !Array.isArray(records)) {
      throw new Error('Expected userId and records (array of app access records)');
    }

    if (records.length < 5) {
      throw new Error('Persona classification requires at least 5 app access records');
    }

    // Validate record structure
    for (const record of records) {
      if (!record.appDisplayName || !record.createdDateTime) {
        throw new Error('Each record must have appDisplayName and createdDateTime');
      }
    }

    // Prepare headers
    const headers = { 'Content-Type': 'application/json' };
    if (isAzureML && AZURE_ML_KEY) {
      headers['Authorization'] = `Bearer ${AZURE_ML_KEY}`;
    }

    // Note: Persona prediction not yet supported in Azure ML endpoint
    const endpointUrl = isAzureML ? ML_SERVER_URL : `${ML_SERVER_URL}/api/predict-persona`;

    const response = await axios.post(
      endpointUrl,
      { userId, records },
      {
        headers,
        timeout: 30000
      }
    );

    // Parse response if it's a string (Azure ML returns JSON string)
    let result = response.data;
    if (typeof result === 'string') {
      result = JSON.parse(result);
    }

    return result;

  } catch (error) {
    console.error('ML server error:', error.message);
    throw error;
  }
};

// Evaluate accuracy on user logs
exports.evaluateAccuracy = async (userId) => {
  try {
    // Read configuration dynamically (for testing)
    const ML_SERVER_URL = process.env.ML_SERVER_URL || 'http://localhost:5001';

    // Fetch user logs
    const logsController = require('../controllers/logsController');
    const logs = await new Promise((resolve, reject) => {
      const req = { query: {} };
      const res = {
        json: (data) => resolve(data.logs || []),
        status: () => ({ json: reject })
      };
      logsController.getAllLogs(req, res);
    });

    // Filter logs for this user
    const userLogs = logs.filter(log => log.userId === userId);

    if (userLogs.length < 20) {
      throw new Error('Need at least 20 log entries for accuracy evaluation');
    }

    const response = await axios.post(
      `${ML_SERVER_URL}/evaluate-accuracy`,
      { 
        logs: userLogs,
        test_size: Math.min(10, Math.floor(userLogs.length * 0.2))
      },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 60000
      }
    );

    return response.data;

  } catch (error) {
    console.error('Accuracy evaluation error:', error.message);
    throw error;
  }
};
