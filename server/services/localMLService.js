const axios = require('axios');

// ML Server Configuration
const ML_SERVER_URL = process.env.ML_SERVER_URL || 'http://localhost:3000';

// Main prediction function - calls separate ML server
exports.predict = async (data) => {
  try {
    // Ensure data is an array
    let payload = data;
    if (!Array.isArray(data)) {
      payload = data.data || [data];
    }

    console.log('🔵 Sending to ML server:', JSON.stringify(payload.slice(0, 3), null, 2), '...');
    console.log('🔵 ML Server URL:', `${ML_SERVER_URL}/api/predict-next-app`);

    const startTime = Date.now();

    const response = await axios.post(
      `${ML_SERVER_URL}/api/predict-next-app`,
      { data: payload },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );

    const elapsed = Date.now() - startTime;
    console.log(`⏱️  ML server responded in ${elapsed}ms`);
    console.log('✅ ML response:', JSON.stringify(response.data, null, 2));

    const result = response.data;

    const azureMLFormat = {
      pred_app: result.top_prediction,
      predictions: result.predictions,
      confidence: result.predictions?.[0]?.confidence || null
    };

    return azureMLFormat;

  } catch (error) {
    console.error('❌ ML server error:', error.message);
    if (error.response) {
      console.error('❌ ML server response:', error.response.data);
    }
    throw error;
  }
};

exports.predictNextApp = async (data) => {
  try {
    if (!data || !Array.isArray(data)) {
      throw new Error('Expected data to be an array of app access records');
    }

    for (const record of data) {
      if (!record.appDisplayName || record.hour === undefined || record.weekday === undefined) {
        throw new Error('Each record must have appDisplayName, hour, and weekday');
      }
    }

    const response = await axios.post(
      `${ML_SERVER_URL}/api/predict-next-app`,
      { data },
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

exports.predictPersona = async (userId, records) => {
  try {
    if (!userId || !records || !Array.isArray(records)) {
      throw new Error('Expected userId and records (array of app access records)');
    }

    if (records.length < 5) {
      throw new Error('Persona classification requires at least 5 app access records');
    }

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
