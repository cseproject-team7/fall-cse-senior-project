# USF Authentication Prediction App

A web application that predicts user authentication patterns using Azure Machine Learning.

## Features

- 🎯 Real-time prediction of next application usage and behavioral patterns
- 👥 Persona-based activity log viewing
- 📊 Pattern chain visualization with ML predictions
- 🔄 Event Hub integration for live authentication data
- 🧪 Dual ML pipelines (LSTM & Random Forest)

---

## Quick Start Guide

This guide walks you through the complete workflow: generating logs → training models → testing infrastructure → running the application.

### Prerequisites

- **Node.js** (v18+)
- **Python** (3.8+)
- **Azure Account** (for production testing)
- **Ollama** (optional, for LLM-based log generation)

### Step 1: Install Dependencies

```bash
# Backend dependencies
cd server
npm install

# Frontend dependencies
cd ../client
npm install

# Python dependencies (for ML pipelines)
cd ..
pip install -r ml_requirements.txt
```

### Step 2: Generate Synthetic Logs

Choose one of the log generation methods:

**Option A: Weighted Log Generation (Recommended for beginners)**
```bash
cd scripts/model_pipeline/reverse_model_disstilation/create_logs
python weighted_logs.py
# Creates: raw_logs/logs.json
```

**Option B: LLM-Based Log Generation (Requires Ollama)**
```bash
cd scripts/model_pipeline/llm_pattern_model/create_logs
python generate_cohort.py
# Creates: raw_logs/cohort_visible_semester.json
```

### Step 3: Train ML Models

**Option A: Reverse Model Distillation Pipeline (LSTM)**
```bash
cd scripts/model_pipeline/reverse_model_disstilation

# Extract sessions
python step1_extract_sessions.py

# Label with LLM (use Ollama for local, or OpenAI)
python step2_llm_labeling.py --backend ollama --model llama3 --sample-size 1000

# Train Random Forest classifier
python step3_train_and_apply.py

# Train Seq2Seq LSTM model
python step4_train_final_model.py

# Test the model
python step5_test_model.py
```

**Option B: LLM Pattern Model Pipeline (Random Forest)**
```bash
cd scripts/model_pipeline/llm_pattern_model

# Extract patterns
python extract_patterns.py

# Train classifiers
python step1_train_classifier.py
python step2_train_forecaster.py

# Test predictions
python step3_predict_next.py
```

### Step 4: Azure Infrastructure Testing (Optional)

For testing the full production pipeline with Azure Event Hub:

```bash
cd scripts/azure_scripts

# 1. Create virtual test users in Azure Entra ID
# Configure credentials in create_vusers.py first
python create_vusers.py

# 2. Send logs to Event Hub (triggers AVRO → JSON pipeline)
python send_logs_to_eventhub.py

# Alternative: Direct blob upload (testing only, skips Event Hub)
python upload_logs_to_blob.py
```

### Step 5: Configure Environment

Copy the example environment file and rename it to `.env`, then fill in your credentials:

```bash
cd server
cp .env.example .env
# Now edit server/.env with your actual values
```

Edit `server/.env` and replace the placeholder values with your actual credentials. See `server/.env.example` for all available configuration options including:
- Azure ML endpoint and API key
- Local ML server URL
- Azure Storage connection string
- Event Hub credentials (for production)
- JWT secret for authentication

### Step 6: Run the Application

**Terminal 1 - Backend:**
```bash
cd server
npm start
```
Server runs on http://localhost:8080

**Terminal 2 - Frontend:**
```bash
cd client
npm start
```
React app runs on http://localhost:3000

**Terminal 3 - Local ML Server (Optional):**
```bash
cd azure_ml_deployment
python local_test_server.py
```
ML server runs on http://localhost:5001

### Step 7: Test the Application

1. Open http://localhost:3000
2. Log in (use test credentials if authentication is enabled)
3. Navigate to **Log Dashboard** to view activity logs
4. Navigate to **Predictions** to test ML pattern predictions
5. Use **Test Lab** to experiment with different input sequences

---

## Development Workflow

```bash
# 1. Generate new logs
cd scripts/model_pipeline/reverse_model_disstilation/create_logs
python weighted_logs.py

# 2. Retrain model
cd ../  # Now in reverse_model_disstilation/
python step1_extract_sessions.py
python step2_llm_labeling.py --backend ollama --model llama3 --sample-size 1000
python step3_train_and_apply.py
python step4_train_final_model.py

# 3. Test locally
cd ../../../azure_ml_deployment  # From reverse_model_disstilation/ to azure_ml_deployment/
python local_test_server.py

# 4. Update backend to use local ML server
# In server/.env, set: ML_SERVER_URL=http://localhost:5001

# 5. Restart backend
cd ../server
npm start
```

---

## Application Architecture

### Frontend (`client/`)

React application with Redux Toolkit for state management and modern feature-based organization:

```
client/
├── src/
│   ├── api/                   # API client layer
│   │   ├── client.js          # Axios base configuration
│   │   ├── logsApi.js         # Logs endpoints
│   │   ├── predictionsApi.js  # ML prediction endpoints
│   │   └── feedbackApi.js     # User feedback endpoints
│   ├── components/            # Reusable UI components
│   │   ├── Layout.jsx         # App layout wrapper
│   │   ├── Sidebar.jsx        # Navigation sidebar
│   │   ├── PatternChainViewer.jsx  # Pattern visualization
│   │   └── ProtectedRoute.jsx # Auth guard
│   ├── features/              # Feature modules (co-located logic)
│   │   ├── activityLogs/      # Activity log viewing
│   │   ├── predictions/       # ML predictions display
│   │   ├── feedback/          # User feedback collection
│   │   └── personaSelector/   # Persona switching
│   ├── pages/                 # Route-level pages
│   │   ├── LogDashboard.jsx   # Main dashboard
│   │   ├── PredictionsPage.jsx # Predictions interface
│   │   ├── TestLabPage.jsx    # ML testing playground
│   │   └── LoginPage.jsx      # Authentication
│   ├── store/                 # Redux slices
│   │   ├── store.js           # Store configuration
│   │   ├── logsSlice.js       # Logs state
│   │   ├── personaSlice.js    # Persona state
│   │   └── predictionsSlice.js # Predictions state
│   ├── hooks/                 # Custom React hooks
│   ├── context/               # React Context providers
│   └── utils/                 # Helper functions
└── package.json
```

**Key Features:**
- Feature-based architecture for scalability
- Centralized API layer with Axios
- Redux Toolkit for predictable state management
- Protected routes with authentication context
- Real-time pattern chain visualization

### Backend (`server/`)

Express.js API server with service-oriented architecture:

```
server/
├── server.js                  # Express app entry point
├── routes/
│   └── api.js                 # Route definitions
├── controllers/               # Request handlers
│   ├── authController.js      # Authentication logic
│   ├── logsController.js      # Log retrieval & filtering
│   ├── predictionController.js # ML prediction orchestration
│   └── mlPredictionController.js # Advanced ML endpoints
├── middleware/
│   └── authMiddleware.js      # JWT authentication
├── services/                  # Business logic layer
│   ├── azureMLService.js      # Azure ML API client
│   ├── localMLService.js      # Local ML server client
│   ├── eventHubService.js     # Event Hub consumer
│   └── kafkaService.js        # Kafka streaming (optional)
├── __tests__/                 # Jest test suites
│   ├── controllers/           # Controller tests
│   ├── middleware/            # Middleware tests
│   ├── services/              # Service tests
│   └── integration/           # E2E API tests
└── package.json
```

**Key Features:**
- RESTful API design with Express
- Service layer for external integrations (Azure ML, Event Hub)
- JWT-based authentication middleware
- Comprehensive test coverage (Jest)
- Environment-based configuration (.env)

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
    "predicted_pattern": "Academic Work",
    "predicted_apps": ["Canvas", "Word Online", "OneDrive"],
    "pattern_chain": [
      {
        "pattern": "Academic Work",
        "probability": 0.85,
        "next_apps": ["Canvas", "Word Online", "OneDrive"]
      },
      {
        "pattern": "Research",
        "probability": 0.72,
        "next_apps": ["Library Database", "Google Scholar"]
      }
    ]
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

## Machine Learning Pipelines

This project includes two distinct ML approaches for pattern prediction, each with different strengths:

### 1. Reverse Model Distillation Pipeline (`reverse_model_disstilation/`)

This pipeline uses an LLM-supervised approach to train an LSTM model that predicts both behavioral patterns and specific application sequences. The process begins by generating synthetic logs, extracting sessions, using an LLM to label a subset of patterns, training a Random Forest classifier on those labels, applying it to all patterns, and finally training a Seq2Seq LSTM model to predict complete app sequences.

**Features:**
- Encoder-Decoder LSTM architecture for variable-length predictions
- LLM-supervised labeling (GPT-4o-mini or Ollama) for semantic pattern understanding
- Memory-efficient streaming and batch processing for large datasets
- Predicts complete pattern sequences (not just next app)

**Pipeline Steps:**
1. **Generate Training Data** - Use `weighted_logs.py` or `model_based_logs.py` to create synthetic logs
2. **Extract Sessions** - Parse logs into discrete sessions based on sign-in/sign-out events
3. **LLM Labeling** - Sample 10k sessions and label with LLM (GPT-4o-mini or Ollama)
4. **Train Classifier** - Train Random Forest on labeled samples
5. **Apply to All** - Use classifier to label remaining sessions
6. **Train LSTM** - Train Seq2Seq LSTM model on all labeled patterns
7. **Test & Predict** - Interactive testing and rolling predictions

**Key Files:**
- `create_logs/weighted_logs.py` - Rule-based log generation
- `create_logs/model_based_logs.py` - LSTM-based log generation
- `step1_extract_sessions.py` - Session extraction
- `step2_llm_labeling.py` - LLM-supervised labeling
- `step3_train_and_apply.py` - Random Forest training
- `step4_train_final_model.py` - LSTM training
- `step5_test_model.py` - Interactive testing
- `step6_rolling_prediction.py` - Multi-step forecasting

**Outputs:**
- `prepared_data/patterns.jsonl` - Extracted sessions
- `prepared_data/labeled_sessions.jsonl` - LLM-labeled samples
- `prepared_data/all_labeled_patterns.jsonl` - All patterns with RF labels
- `model_output/seq2seq_pattern_model.pth` - Final LSTM model

---

### 2. LLM Pattern Model Pipeline (`llm_pattern_model/`)

This pipeline uses a fully supervised approach with pre-labeled cohort data to train Random Forest models for context classification and sequential pattern forecasting. It's designed for scenarios where ground-truth labels are available and focuses on predicting high-level behavioral contexts rather than individual apps.

**Features:**
- Weekly behavior aggregation for context classification
- N-gram sequence modeling for temporal pattern forecasting
- Blueprint-based app recommendations
- Persona-aware predictions (student major, work ethic)

**Pipeline Steps:**
1. **Generate Labeled Data** - Use `generate_cohort.py` or `local_llm_logs.py` with Ollama for contextual labels
2. **Extract Patterns** - Sessionize logs into behavioral patterns
3. **Train Classifier** - Weekly context classifier (e.g., "MIDTERMS", "PROJECT")
4. **Train Forecaster** - Sequential pattern forecaster using N-grams
5. **Test Predictions** - Single-step and rolling predictions with blueprint recommendations

**Key Files:**
- `create_logs/generate_cohort.py` - Cohort generation
- `create_logs/local_llm_logs.py` - Single student semester with LLM context labels
- `extract_patterns.py` - Pattern extraction
- `step1_train_classifier.py` - Weekly context classifier
- `step2_train_forecaster.py` - Sequential forecaster
- `step3_predict_next.py` - Single-step prediction
- `step4_rolling_prediction.py` - Multi-step trajectory
- `train_persona_encoder.py` - Persona inference (optional)
- `train_aware_forecaster.py` - Persona-aware predictions (optional)

**Outputs:**
- `shadow_simulation/student_patterns.json` - Sessionized patterns
- `shadow_simulation/weekly_classifier.pkl` - Context classifier
- `shadow_simulation/pattern_forecaster.pkl` - Sequential forecaster
- `shadow_simulation/blueprints.json` - Context-to-app mappings

---

### Pipeline Comparison

| Feature | Reverse Model Distillation | LLM Pattern Model |
|---------|---------------------------|-------------------|
| **Supervision** | LLM-labeled (10k samples) | Fully pre-labeled |
| **Model Type** | Seq2Seq LSTM | Random Forest |
| **Predictions** | Complete app sequences | Behavioral contexts + apps |
| **Data Requirements** | Raw logs only | Labeled cohort data |
| **Best For** | Large-scale unlabeled data | Small labeled datasets |

---

## Azure Testing & Deployment Scripts

The `scripts/azure_scripts/` folder contains utilities for testing the full Azure infrastructure pipeline with synthetic data.

### Azure Infrastructure Flow

```
Virtual Users → Event Hub → AVRO Capture → ADF Pipeline → JSON Blob Storage → Application
```

### Available Scripts

**1. Create Virtual Users** (`create_vusers.py`)
Creates test user accounts in Azure Entra ID (formerly Azure AD) for simulating authentication patterns. Requires app registration with `User.ReadWrite.All` permissions.

**2. Anonymize and Inject Logs** (`inject_anonymized_logs.py`)
Injects anonymized authentication logs for specific users. Uses SHA-256 hashing to protect user identity while maintaining consistency for pattern analysis.

**3. Send Logs to Event Hub** (`send_logs_to_eventhub.py`)
Streams generated logs to Azure Event Hub in batches. Event Hub captures data in AVRO format, triggering an Azure Data Factory (ADF) pipeline that converts AVRO to JSON and stores in Blob Storage.

Features:
- Batch processing (100 logs per batch)
- Progress tracking and error handling
- Test message functionality
- Automatic throttling prevention

**4. Direct Blob Upload** (`upload_logs_to_blob.py`)
**(Testing Only)** Bypasses Event Hub and directly uploads logs to Azure Blob Storage. Useful for rapid iteration during development.

Features:
- Deletes existing logs in the container
- Uploads new synthetic logs
- Creates Spark-style success markers

**⚠️ Note:** Direct blob upload skips the Event Hub → AVRO → JSON pipeline. For production testing, use `send_logs_to_eventhub.py` to validate the full infrastructure.

---

**Team 7 - University of South Florida**
