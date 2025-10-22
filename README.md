# USF Authentication Prediction App

A web application that predicts user authentication patterns using Azure Machine Learning.

## Features

- 🎯 Real-time prediction of next application usage
- 👥 Persona-based activity log viewing
- 📊 Side-by-side comparison of input logs and ML predictions
- 🔄 Event Hub integration for live authentication data

## Setup

### Prerequisites
- Node.js (v18+)
- Azure ML API key
- Azure Event Hub (for production logs)

### Installation

1. **Install backend dependencies:**
```bash
cd server
npm install
```

2. **Install frontend dependencies:**
```bash
cd client
npm install
```

3. **Configure environment variables:**

Create `server/.env`:
```
AZURE_ML_API_KEY=your_api_key_here
AZURE_ML_ENDPOINT=https://process-predicting-model-yaxap.eastus.inference.ml.azure.com/score
AZURE_ML_DEPLOYMENT=predict-model-4

# Optional: Event Hub configuration (for production)
EVENT_HUB_CONNECTION_STRING=your_connection_string
EVENT_HUB_NAME=your_event_hub_name
```

## Running Locally

**Start backend (Terminal 1):**
```bash
cd server
npm start
```
Server runs on http://localhost:8080

**Start frontend (Terminal 2):**
```bash
cd client
npm start
```
React app runs on http://localhost:3000

## Project Structure

```
├── client/                    # React frontend
│   └── src/
│       ├── App.jsx           # Main app with persona selector & side-by-side view
│       └── App.css           # Styling
├── server/                    # Node.js backend
│   ├── server.js             # Express server entry point
│   ├── routes/
│   │   └── api.js           # API route definitions
│   ├── controllers/
│   │   ├── logsController.js      # Handles log retrieval
│   │   └── predictionController.js # Handles predictions
│   └── services/
│       ├── eventHubService.js     # Event Hub integration
│       └── azureMLService.js      # Azure ML integration
└── .github/workflows/pipeline.yml  # CI/CD
```

## API Endpoints

### Get Available Personas
```
GET /api/logs/personas
```
Returns list of personas (engineering_junior, admin_employee, etc.)

### Get Logs by Persona
```
GET /api/logs/:persona?limit=50
```
Returns activity logs for specified persona

### Make Prediction
```
POST /api/predict
Content-Type: application/json

Body:
{
  "data": [
    {
      "appDisplayName": "Canvas",
      "hour": 14,
      "weekday": 1
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "pred_app": "Canvas"
  },
  "timestamp": "2025-10-22T..."
}
```

## Adding New Endpoints

The backend is structured for easy collaboration:

1. **Create a new controller** in `server/controllers/`:
```javascript
// server/controllers/analyticsController.js
exports.getAnalytics = async (req, res) => {
  // Your logic here
  res.json({ success: true, data: {} });
};
```

2. **Add route** in `server/routes/api.js`:
```javascript
const analyticsController = require('../controllers/analyticsController');
router.get('/analytics', analyticsController.getAnalytics);
```

3. **Done!** Your new endpoint is available at `/api/analytics`

## Event Hub Integration

The `eventHubService.js` currently uses mock data. To connect to real Event Hub:

1. Install Azure Event Hubs SDK:
```bash
cd server
npm install @azure/event-hubs
```

2. Update `server/services/eventHubService.js` with Event Hub consumer logic
3. Add Event Hub credentials to `.env`

## Deployment

Push to `master` branch triggers automatic deployment to Azure Web App via GitHub Actions.

**Configure Azure:**
1. Azure Portal → App Service → Configuration
2. Add Application Settings:
   - `AZURE_ML_API_KEY`
   - `EVENT_HUB_CONNECTION_STRING` (if using Event Hub)
   - `EVENT_HUB_NAME`
3. Save and restart

## ML Model

The ML model processes authentication logs to predict next application usage based on:
- Application display name
- Hour of day (0-23)
- Day of week (0=Monday through 6=Sunday)

Returns predicted application name in `pred_app` field.

---

**Team 7 - University of South Florida**
