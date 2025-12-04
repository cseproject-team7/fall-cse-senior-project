const axios = require('axios');

const axios = require('axios');

// ML Server Configuration
// Use environment variable for production (separate ML server)
// Default to localhost for development
const ML_SERVER_URL = process.env.ML_SERVER_URL || 'http://localhost:5001';

// Main prediction function - calls dual-head LSTM server
exports.predict = async (data) => {
  try {
    // Ensure data is an array
    let payload = data;
    if (!Array.isArray(data)) {
      payload = data.data || [data];
    }

    console.log('🔵 Sending to Dual-Head LSTM server:', JSON.stringify(payload.slice(0, 3), null, 2), '...');
    console.log('🔵 ML Server URL:', `${ML_SERVER_URL}/predict`);

    // Convert logs to format expected by ml_server_dual
    const logs = payload.map(log => ({
      appDisplayName: log.appDisplayName,
      createdDateTime: log.createdDateTime
    }));

    const startTime = Date.now();

    // Call the dual-head LSTM server
    const response = await axios.post(
      `${ML_SERVER_URL}/predict`,
      { 
        logs: logs,
        num_predictions: 10  // Get 10 future pattern predictions
      },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );

    const elapsed = Date.now() - startTime;
    console.log(`⏱️  ML server responded in ${elapsed}ms`);
    console.log('✅ ML response:', JSON.stringify(response.data, null, 2));

    const result = response.data;

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
    console.log('📝 Recording new app access:', appDisplayName);
    
    // Convert logs to proper format
    let logData = Array.isArray(logs) ? logs : [logs];
    
    // Call ML server with new app access
    const response = await axios.post(
      `${ML_SERVER_URL}/predict`,
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
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );

    const result = response.data;

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

    const response = await axios.post(
      `${ML_SERVER_URL}/api/predict-persona`,
      { userId, records },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );

    return response.data;

  } catch (error) {
    console.error('ML server error:', error.message);
    throw error;
  }
};
