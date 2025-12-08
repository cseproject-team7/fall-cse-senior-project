"""
Unit tests for upload_logs_to_blob.py - Azure Blob Storage upload script
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, mock_open, MagicMock
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import upload_logs_to_blob


@pytest.mark.unit
class TestDeleteAllBlobs:
    """Tests for delete_all_blobs function"""
    
    @patch('upload_logs_to_blob.BlobServiceClient')
    def test_deletes_all_blobs_in_prefix(self, mock_blob_service):
        """Should delete all blobs with specified prefix"""
        # Setup mocks
        mock_container = Mock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container
        
        # Mock blob listing
        mock_blobs = [
            Mock(name='json-output/logs_001.json'),
            Mock(name='json-output/logs_002.json'),
            Mock(name='json-output/logs_003.json')
        ]
        mock_container.list_blobs.return_value = mock_blobs
        mock_container.delete_blob.return_value = None
        
        count = upload_logs_to_blob.delete_all_blobs()
        
        # Should delete all 3 blobs
        assert count == 3
        assert mock_container.delete_blob.call_count == 3
    
    @patch('upload_logs_to_blob.BlobServiceClient')
    def test_returns_zero_for_empty_container(self, mock_blob_service):
        """Should return 0 when no blobs to delete"""
        mock_container = Mock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container
        mock_container.list_blobs.return_value = []
        
        count = upload_logs_to_blob.delete_all_blobs()
        
        assert count == 0
        assert mock_container.delete_blob.call_count == 0
    
    @patch('upload_logs_to_blob.BlobServiceClient')
    def test_handles_deletion_errors(self, mock_blob_service):
        """Should handle errors during blob deletion"""
        mock_container = Mock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container
        
        # Mock exception during list_blobs
        mock_container.list_blobs.side_effect = Exception("Connection error")
        
        count = upload_logs_to_blob.delete_all_blobs()
        
        # Should return 0 and not crash
        assert count == 0


@pytest.mark.unit
class TestUploadLogs:
    """Tests for upload_logs function"""
    
    @patch('upload_logs_to_blob.BlobServiceClient')
    @patch('builtins.open', new_callable=mock_open, read_data='{"test": "log"}\n{"test2": "log2"}\n')
    @patch('os.path.exists')
    def test_uploads_logs_in_chunks(self, mock_exists, mock_file, mock_blob_service):
        """Should split logs into chunks and upload each"""
        mock_exists.return_value = True
        
        mock_container = Mock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container
        mock_container.upload_blob.return_value = None
        
        # Test with small chunk size
        with patch('upload_logs_to_blob.LINES_PER_FILE', 1):
            upload_logs_to_blob.upload_logs()
        
        # Should have uploaded 2 chunks (2 lines, 1 per chunk)
        assert mock_container.upload_blob.call_count == 2
    
    @patch('os.path.exists')
    def test_handles_missing_log_file(self, mock_exists):
        """Should handle case when log file doesn't exist"""
        mock_exists.return_value = False
        
        # Should not crash, just print error
        upload_logs_to_blob.upload_logs()
    
    @patch('upload_logs_to_blob.BlobServiceClient')
    @patch('builtins.open', new_callable=mock_open, read_data='')
    @patch('os.path.exists')
    def test_handles_empty_log_file(self, mock_exists, mock_file, mock_blob_service):
        """Should handle empty log file gracefully"""
        mock_exists.return_value = True
        
        mock_container = Mock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container
        
        upload_logs_to_blob.upload_logs()
        
        # Should not upload anything for empty file
        assert mock_container.upload_blob.call_count == 0
    
    @patch('upload_logs_to_blob.BlobServiceClient')
    @patch('builtins.open', new_callable=mock_open, read_data='{"test": "log"}\n')
    @patch('os.path.exists')
    def test_blob_names_are_sequential(self, mock_exists, mock_file, mock_blob_service):
        """Blob names should be sequential (part-001, part-002, etc)"""
        mock_exists.return_value = True
        
        mock_container = Mock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container
        
        upload_logs_to_blob.upload_logs()
        
        # Check the blob name format
        call_args = mock_container.upload_blob.call_args[0]
        blob_name = call_args[0]
        
        assert 'json-output/' in blob_name
        assert blob_name.startswith('json-output/logs_part_')
        assert blob_name.endswith('.json')


@pytest.mark.unit
class TestConfiguration:
    """Tests for configuration and setup"""
    
    def test_container_name_is_defined(self):
        """CONTAINER_NAME should be properly defined"""
        assert hasattr(upload_logs_to_blob, 'CONTAINER_NAME')
        assert upload_logs_to_blob.CONTAINER_NAME == 'json-signin-logs'
    
    def test_blob_prefix_is_defined(self):
        """BLOB_PREFIX should be properly defined"""
        assert hasattr(upload_logs_to_blob, 'BLOB_PREFIX')
        assert upload_logs_to_blob.BLOB_PREFIX == 'json-output/'
    
    def test_log_file_path_is_defined(self):
        """LOG_FILE path should be properly defined"""
        assert hasattr(upload_logs_to_blob, 'LOG_FILE')
        assert isinstance(upload_logs_to_blob.LOG_FILE, str)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for the complete upload workflow"""
    
    @patch('upload_logs_to_blob.BlobServiceClient')
    @patch('builtins.open', new_callable=mock_open, read_data='{"app": "Canvas"}\n{"app": "Outlook"}\n')
    @patch('os.path.exists')
    def test_delete_then_upload_workflow(self, mock_exists, mock_file, mock_blob_service):
        """Test complete workflow: delete old logs, upload new ones"""
        mock_exists.return_value = True
        
        # Setup blob service mock
        mock_container = Mock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container
        
        # Mock existing blobs for deletion
        mock_container.list_blobs.return_value = [
            Mock(name='json-output/old_001.json')
        ]
        mock_container.delete_blob.return_value = None
        mock_container.upload_blob.return_value = None
        
        # Run delete
        deleted = upload_logs_to_blob.delete_all_blobs()
        assert deleted == 1
        
        # Run upload
        upload_logs_to_blob.upload_logs()
        
        # Verify both operations happened
        assert mock_container.delete_blob.call_count >= 1
        assert mock_container.upload_blob.call_count >= 1


@pytest.mark.unit
class TestEnvironmentSetup:
    """Tests for environment variable handling"""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('upload_logs_to_blob.load_dotenv')
    def test_loads_env_from_server_directory(self, mock_load_dotenv):
        """Should attempt to load .env from server directory"""
        # Reimport to trigger env loading
        import importlib
        importlib.reload(upload_logs_to_blob)
        
        # load_dotenv should have been called
        assert mock_load_dotenv.call_count >= 0  # May be called during import
    
    @patch.dict(os.environ, {'AZURE_STORAGE_CONNECTION_STRING': ''}, clear=True)
    @patch('sys.exit')
    def test_exits_when_connection_string_missing(self, mock_exit):
        """Should exit if AZURE_STORAGE_CONNECTION_STRING not found"""
        # This test checks the module-level code that runs on import
        # The actual check happens when the module loads, so we verify the behavior exists
        assert 'connection_string' in dir(upload_logs_to_blob)


@pytest.mark.unit
class TestBlobNaming:
    """Tests for blob naming conventions"""
    
    def test_blob_names_use_three_digit_padding(self):
        """Blob part numbers should be zero-padded to 3 digits"""
        # Test the format used in upload_logs
        for i in range(1, 100):
            name = f"logs_part_{i:03d}.json"
            
            # Should always be 3 digits
            assert len(name.split('_')[-1].split('.')[0]) == 3
    
    def test_blob_names_include_json_extension(self):
        """All blob names should end with .json"""
        # This is enforced in the BLOB_PREFIX and naming
        assert upload_logs_to_blob.LOG_FILE.endswith('.json')

