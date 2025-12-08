"""
Pytest fixtures and configuration for script tests
"""
import pytest
import sys
import os
from datetime import datetime
from unittest.mock import Mock, MagicMock
import torch
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_azure_blob_client():
    """Mock Azure Blob Storage client"""
    mock_client = Mock()
    mock_container = Mock()
    mock_blob = Mock()
    
    # Mock blob listing
    mock_container.list_blobs.return_value = [
        Mock(name='json-output/logs_001.json'),
        Mock(name='json-output/logs_002.json')
    ]
    
    # Mock blob operations
    mock_container.delete_blob.return_value = None
    mock_container.upload_blob.return_value = None
    
    mock_client.get_container_client.return_value = mock_container
    mock_client.get_blob_client.return_value = mock_blob
    
    return mock_client


@pytest.fixture
def sample_log_entry():
    """Sample log entry for testing"""
    return {
        "id": "12345-67890",
        "createdDateTime": "2025-01-01T09:00:00Z",
        "userPrincipalName": "test@usf.edu",
        "userId": "user-123",
        "appDisplayName": "Canvas",
        "ipAddress": "131.247.34.100",
        "location": {
            "city": "Tampa",
            "state": "Florida",
            "countryOrRegion": "US"
        },
        "status": {
            "errorCode": 0,
            "failureReason": "Success"
        }
    }


@pytest.fixture
def sample_logs_sequence():
    """Sample sequence of logs for ML testing"""
    return [
        {
            "appDisplayName": "Canvas",
            "createdDateTime": "2025-01-01T09:00:00Z"
        },
        {
            "appDisplayName": "Outlook",
            "createdDateTime": "2025-01-01T09:15:00Z"
        },
        {
            "appDisplayName": "Teams",
            "createdDateTime": "2025-01-01T09:30:00Z"
        }
    ]


@pytest.fixture
def mock_lstm_model():
    """Mock PyTorch LSTM model"""
    class MockDualHeadLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, pattern_ids, app_ids):
            # Return dummy logits
            batch_size = pattern_ids.size(0)
            pattern_logits = torch.randn(batch_size, 10)
            app_logits = torch.randn(batch_size, 20)
            return pattern_logits, app_logits
    
    model = MockDualHeadLSTM()
    model.eval()
    return model


@pytest.fixture
def mock_encoders():
    """Mock LabelEncoders for patterns and apps"""
    pattern_encoder = Mock()
    pattern_encoder.classes_ = ['Canvas → Outlook', 'Outlook → Teams', 'Teams → Canvas']
    pattern_encoder.transform.return_value = [0, 1, 2]
    
    app_encoder = Mock()
    app_encoder.classes_ = ['Canvas', 'Outlook', 'Teams', 'GitHub', 'MATLAB']
    app_encoder.transform.return_value = [0, 1, 2, 3, 4]
    
    return {'pattern': pattern_encoder, 'app': app_encoder}


@pytest.fixture
def mock_model_config():
    """Mock model configuration"""
    return {
        'num_patterns': 10,
        'num_apps': 20,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2,
        'sequence_length': 5
    }


@pytest.fixture
def mock_event_hub_producer():
    """Mock Azure Event Hub producer"""
    mock_producer = Mock()
    mock_producer.send_batch.return_value = None
    mock_producer.close.return_value = None
    return mock_producer


@pytest.fixture
def sample_user():
    """Sample user for testing"""
    return {
        "userId": "user-123",
        "userPrincipalName": "test.student@usf.edu",
        "displayName": "Test Student",
        "persona": "engineering_junior"
    }


@pytest.fixture
def mock_flask_app():
    """Mock Flask app for testing ML servers"""
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test"""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)

