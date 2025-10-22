const axios = require('axios');

const AZURE_ML_ENDPOINT = process.env.AZURE_ML_ENDPOINT || 
  'https://process-predicting-model-yaxap.eastus.inference.ml.azure.com/score';
const AZURE_ML_API_KEY = process.env.AZURE_ML_API_KEY;
const AZURE_ML_DEPLOYMENT = process.env.AZURE_ML_DEPLOYMENT || 'predict-model-4';

exports.predict = async (data) => {
  // Ensure data is an array
  let payload = data;
  if (!Array.isArray(data)) {
    // If data is wrapped in an object, try to extract the array
    payload = data.data || [data];
  }
  
  console.log('Sending to Azure ML:', JSON.stringify(payload, null, 2));
  
  const response = await axios.post(
    AZURE_ML_ENDPOINT,
    payload,  // Send the array directly
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${AZURE_ML_API_KEY}`,
        'azureml-model-deployment': AZURE_ML_DEPLOYMENT
      },
      timeout: 30000
    }
  );

  console.log('Azure ML response:', JSON.stringify(response.data, null, 2));
  return response.data;
};

