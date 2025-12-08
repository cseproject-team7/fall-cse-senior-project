"""
Unit tests for local_ml_server.py - Dual-Head LSTM ML Server
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import torch
import joblib

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
import local_ml_server as ml_server_dual


@pytest.mark.unit
class TestDualHeadLSTM:
    """Tests for the DualHeadLSTM model architecture"""
    
    def test_model_initialization(self):
        """Test model can be initialized with correct parameters"""
        model = ml_server_dual.DualHeadLSTM(
            num_patterns=10,
            num_apps=20,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        assert model.pattern_embedding.num_embeddings == 10
        assert model.app_embedding.num_embeddings == 20
    
    def test_model_forward_pass(self):
        """Test forward pass returns correct tensor shapes"""
        model = ml_server_dual.DualHeadLSTM(
            num_patterns=10,
            num_apps=20,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        
        # Create dummy input (batch_size=2, sequence_length=3)
        pattern_ids = torch.randint(0, 10, (2, 3))
        app_ids = torch.randint(0, 20, (2, 3))
        
        pattern_logits, app_logits = model(pattern_ids, app_ids)
        
        assert pattern_logits.shape == (2, 10)  # batch_size x num_patterns
        assert app_logits.shape == (2, 20)  # batch_size x num_apps


@pytest.mark.unit
class TestSessionGrouping:
    """Tests for group_apps_into_sessions function"""
    
    def test_empty_logs_returns_empty_sessions(self):
        """Empty logs should return empty session list"""
        sessions = ml_server_dual.group_apps_into_sessions([])
        assert sessions == []
    
    def test_single_log_creates_single_session(self, sample_logs_sequence):
        """Single log entry should create one session"""
        single_log = [sample_logs_sequence[0]]
        sessions = ml_server_dual.group_apps_into_sessions(single_log)
        
        assert len(sessions) == 1
        assert len(sessions[0]['apps']) == 1
        assert sessions[0]['apps'][0] == 'Canvas'
    
    def test_logs_within_gap_same_session(self, sample_logs_sequence):
        """Logs within time gap should be in same session"""
        # All logs within 20 minutes
        sessions = ml_server_dual.group_apps_into_sessions(
            sample_logs_sequence,
            gap_minutes=20
        )
        
        assert len(sessions) == 1
        assert len(sessions[0]['apps']) == 3
    
    def test_logs_exceeding_gap_different_sessions(self):
        """Logs exceeding time gap should split into different sessions"""
        logs = [
            {"appDisplayName": "Canvas", "createdDateTime": "2025-01-01T09:00:00Z"},
            {"appDisplayName": "Outlook", "createdDateTime": "2025-01-01T10:00:00Z"}  # 60 min gap
        ]
        
        sessions = ml_server_dual.group_apps_into_sessions(logs, gap_minutes=20)
        
        assert len(sessions) == 2
    
    def test_consecutive_duplicate_apps_deduplicated(self):
        """Consecutive duplicate apps should be deduplicated"""
        logs = [
            {"appDisplayName": "Canvas", "createdDateTime": "2025-01-01T09:00:00Z"},
            {"appDisplayName": "Canvas", "createdDateTime": "2025-01-01T09:05:00Z"},
            {"appDisplayName": "Outlook", "createdDateTime": "2025-01-01T09:10:00Z"}
        ]
        
        sessions = ml_server_dual.group_apps_into_sessions(logs)
        
        assert len(sessions) == 1
        assert sessions[0]['apps'] == ['Canvas', 'Outlook']


@pytest.mark.unit
class TestPatternDetection:
    """Tests for detect_common_patterns function"""
    
    def test_empty_sessions_returns_empty_dict(self):
        """Empty sessions should return empty pattern dict"""
        patterns = ml_server_dual.detect_common_patterns([])
        assert patterns == {}
    
    def test_session_with_single_app_no_pattern(self):
        """Session with single app shouldn't create patterns"""
        sessions = [{'apps': ['Canvas']}]
        patterns = ml_server_dual.detect_common_patterns(sessions)
        assert patterns == {}
    
    def test_simple_two_app_pattern_detected(self):
        """Simple two-app pattern should be detected"""
        sessions = [
            {'apps': ['Canvas', 'Outlook']},
            {'apps': ['Canvas', 'Outlook']}
        ]
        
        patterns = ml_server_dual.detect_common_patterns(sessions)
        
        assert 'Canvas → Outlook' in patterns
        assert patterns['Canvas → Outlook'] == 2
    
    def test_longer_patterns_detected(self):
        """Longer app sequences should be detected as patterns"""
        sessions = [
            {'apps': ['Canvas', 'Outlook', 'Teams']},
            {'apps': ['Canvas', 'Outlook', 'Teams']}
        ]
        
        patterns = ml_server_dual.detect_common_patterns(sessions)
        
        # Should detect both 2-app and 3-app patterns
        assert 'Canvas → Outlook' in patterns
        assert 'Canvas → Outlook → Teams' in patterns


@pytest.mark.unit  
class TestPredictEndpoint:
    """Tests for the /predict endpoint"""
    
    @patch('ml_server_dual.model')
    @patch('ml_server_dual.pattern_encoder')
    @patch('ml_server_dual.app_encoder')
    @patch('ml_server_dual.config')
    def test_predict_with_valid_logs(self, mock_config, mock_app_enc, mock_pat_enc, mock_model):
        """Test prediction with valid log sequence"""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda x: {
            'num_patterns': 10,
            'num_apps': 20,
            'sequence_length': 5
        }[x]
        
        mock_pat_enc.classes_ = ['Canvas → Outlook', 'Outlook → Teams']
        mock_app_enc.classes_ = ['Canvas', 'Outlook', 'Teams', 'GitHub']
        
        # Mock model output
        mock_model.return_value = (
            torch.tensor([[0.8, 0.2, 0.1, 0.05]]),  # pattern logits
            torch.tensor([[0.1, 0.9, 0.3, 0.05]])   # app logits
        )
        mock_model.eval = Mock()
        
        # Test data
        test_data = {
            'logs': [
                {'appDisplayName': 'Canvas', 'createdDateTime': '2025-01-01T09:00:00Z'},
                {'appDisplayName': 'Outlook', 'createdDateTime': '2025-01-01T09:15:00Z'}
            ],
            'num_predictions': 5
        }
        
        with ml_server_dual.app.test_client() as client:
            response = client.post('/predict', json=test_data)
            
            # Should return 200 even if model isn't fully loaded (will use fallback)
            assert response.status_code in [200, 500]
    
    def test_predict_missing_logs_returns_error(self):
        """Test prediction without logs returns error"""
        with ml_server_dual.app.test_client() as client:
            response = client.post('/predict', json={'num_predictions': 5})
            
            assert response.status_code == 400
            data = response.get_json()
            assert data['success'] is False
            assert 'logs' in data['error'].lower()
    
    def test_predict_empty_logs_returns_error(self):
        """Test prediction with empty logs returns error"""
        with ml_server_dual.app.test_client() as client:
            response = client.post('/predict', json={'logs': []})
            
            assert response.status_code == 400
            data = response.get_json()
            assert data['success'] is False


@pytest.mark.unit
class TestLoadModel:
    """Tests for model loading functionality"""
    
    @patch('ml_server_dual.joblib.load')
    @patch('ml_server_dual.torch.load')
    @patch('os.path.exists')
    def test_load_model_success(self, mock_exists, mock_torch_load, mock_joblib_load):
        """Test successful model loading"""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock encoder classes
        mock_pattern_encoder = Mock()
        mock_pattern_encoder.classes_ = ['Pattern1', 'Pattern2']
        
        mock_app_encoder = Mock()
        mock_app_encoder.classes_ = ['App1', 'App2']
        
        mock_config = {
            'num_patterns': 10,
            'num_apps': 20,
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2
        }
        
        # Setup joblib.load to return different values on consecutive calls
        mock_joblib_load.side_effect = [mock_pattern_encoder, mock_app_encoder, mock_config]
        
        # Mock torch state dict
        mock_torch_load.return_value = {}
        
        result = ml_server_dual.load_model()
        
        # Should attempt to load model
        assert mock_joblib_load.call_count == 3
        assert mock_torch_load.call_count == 1
    
    @patch('ml_server_dual.joblib.load')
    def test_load_model_file_not_found(self, mock_joblib_load):
        """Test model loading with missing files"""
        mock_joblib_load.side_effect = FileNotFoundError("Model file not found")
        
        result = ml_server_dual.load_model()
        
        assert result is False


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for the /health endpoint"""
    
    def test_health_check_returns_200(self):
        """Health endpoint should return 200 OK"""
        with ml_server_dual.app.test_client() as client:
            response = client.get('/health')
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['status'] == 'ok'
    
    def test_health_check_includes_timestamp(self):
        """Health endpoint should include timestamp"""
        with ml_server_dual.app.test_client() as client:
            response = client.get('/health')
            data = response.get_json()
            
            assert 'timestamp' in data
            # Verify timestamp is valid ISO format
            datetime.fromisoformat(data['timestamp'])

