"""
Unit tests for create_logs.py - Log generation script
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import create_logs


@pytest.mark.unit
class TestGenerateUsers:
    """Tests for generate_users function"""
    
    def test_generate_correct_number_of_users(self):
        """Should generate exactly the requested number of users"""
        users = create_logs.generate_users(5)
        assert len(users) == 5
    
    def test_generated_users_have_required_fields(self):
        """Each user should have all required fields"""
        users = create_logs.generate_users(3)
        
        for user in users:
            assert 'userId' in user
            assert 'userPrincipalName' in user
            assert 'displayName' in user
            assert 'persona' in user
    
    def test_user_emails_are_usf_domain(self):
        """All generated emails should be in usf.edu domain"""
        users = create_logs.generate_users(5)
        
        for user in users:
            assert user['userPrincipalName'].endswith('@usf.edu')
    
    def test_user_personas_are_valid(self):
        """All personas should be from the defined PERSONAS dict"""
        users = create_logs.generate_users(10)
        valid_personas = set(create_logs.PERSONAS.keys())
        
        for user in users:
            assert user['persona'] in valid_personas
    
    def test_user_ids_are_unique(self):
        """All user IDs should be unique"""
        users = create_logs.generate_users(20)
        user_ids = [u['userId'] for u in users]
        
        assert len(user_ids) == len(set(user_ids))


@pytest.mark.unit
class TestCreateSigninLog:
    """Tests for create_signin_log function"""
    
    def test_creates_log_with_all_required_fields(self, sample_user):
        """Log should have all required Microsoft Graph API fields"""
        timestamp = datetime.now()
        log = create_logs.create_signin_log(sample_user, timestamp)
        
        required_fields = [
            'id', 'createdDateTime', 'userPrincipalName',
            'userId', 'appDisplayName', 'ipAddress',
            'location', 'status'
        ]
        
        for field in required_fields:
            assert field in log
    
    def test_log_timestamp_is_iso_format(self, sample_user):
        """Timestamp should be in ISO format with Z suffix"""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        log = create_logs.create_signin_log(sample_user, timestamp)
        
        assert log['createdDateTime'].endswith('Z')
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(log['createdDateTime'].replace('Z', '+00:00'))
        assert parsed.year == 2025
    
    def test_app_selection_respects_persona(self, sample_user):
        """App selection should respect persona preferences"""
        # Engineering junior persona should use engineering apps
        log = create_logs.create_signin_log(sample_user, datetime.now())
        
        engineering_apps = list(create_logs.PERSONAS['engineering_junior']['apps'].keys())
        assert log['appDisplayName'] in engineering_apps
    
    def test_ip_address_matches_location(self, sample_user):
        """IP address should start with location-specific prefix"""
        log = create_logs.create_signin_log(sample_user, datetime.now())
        
        # Get valid IP prefixes for this persona
        persona_profile = create_logs.PERSONAS[sample_user['persona']]
        valid_prefixes = [
            create_logs.LOCATIONS[loc]
            for loc in persona_profile['locations'].keys()
        ]
        
        ip_prefix = '.'.join(log['ipAddress'].split('.')[:3]) + '.'
        assert ip_prefix in valid_prefixes
    
    def test_location_is_tampa_florida(self, sample_user):
        """All logs should be from Tampa, Florida"""
        log = create_logs.create_signin_log(sample_user, datetime.now())
        
        assert log['location']['city'] == 'Tampa'
        assert log['location']['state'] == 'Florida'
        assert log['location']['countryOrRegion'] == 'US'
    
    def test_success_status_most_common(self, sample_user):
        """Most logs (~95%) should have success status"""
        success_count = 0
        total_logs = 100
        
        for _ in range(total_logs):
            log = create_logs.create_signin_log(sample_user, datetime.now())
            if log['status']['errorCode'] == 0:
                success_count += 1
        
        # Should be roughly 95% success (allow some variance)
        assert success_count >= 85  # At least 85%
        assert success_count <= 100


@pytest.mark.unit
class TestPersonaDefinitions:
    """Tests for PERSONAS configuration"""
    
    def test_all_personas_have_required_fields(self):
        """Each persona should have apps, locations, and activity_schedule"""
        for persona_name, persona_data in create_logs.PERSONAS.items():
            assert 'apps' in persona_data
            assert 'locations' in persona_data
            assert 'activity_schedule' in persona_data
            assert callable(persona_data['activity_schedule'])
    
    def test_app_probabilities_sum_to_one(self):
        """App probabilities for each persona should sum to ~1.0"""
        for persona_name, persona_data in create_logs.PERSONAS.items():
            prob_sum = sum(persona_data['apps'].values())
            assert abs(prob_sum - 1.0) < 0.01, f"{persona_name} apps don't sum to 1.0"
    
    def test_location_probabilities_sum_to_one(self):
        """Location probabilities for each persona should sum to ~1.0"""
        for persona_name, persona_data in create_logs.PERSONAS.items():
            prob_sum = sum(persona_data['locations'].values())
            assert abs(prob_sum - 1.0) < 0.01, f"{persona_name} locations don't sum to 1.0"
    
    def test_engineering_junior_has_technical_apps(self):
        """Engineering junior persona should have technical apps"""
        apps = create_logs.PERSONAS['engineering_junior']['apps']
        
        technical_apps = ['MATLAB', 'SolidWorks', 'GitHub', 'IEEE Xplore']
        for app in technical_apps:
            assert app in apps
    
    def test_admin_employee_office_hours(self):
        """Admin employee should be active during office hours only"""
        schedule = create_logs.PERSONAS['admin_employee']['activity_schedule']
        
        # Should be active during weekday office hours
        weekday_morning = datetime(2025, 1, 6, 9, 0, 0)  # Monday 9 AM
        assert schedule(weekday_morning) == 1.0
        
        # Should not be active on weekends
        weekend = datetime(2025, 1, 4, 9, 0, 0)  # Saturday
        assert schedule(weekend) == 0.0


@pytest.mark.unit
class TestLocations:
    """Tests for LOCATIONS configuration"""
    
    def test_all_locations_have_valid_ip_prefixes(self):
        """All locations should have valid IP address prefixes"""
        for location_name, ip_prefix in create_logs.LOCATIONS.items():
            parts = ip_prefix.split('.')
            
            # Should have 3 octets (missing last one for randomization)
            assert len(parts) == 4
            assert parts[3] == ''  # Last part should be empty
            
            # First 3 parts should be valid IP octets
            for i in range(3):
                assert parts[i].isdigit()
                assert 0 <= int(parts[i]) <= 255
    
    def test_usf_campus_locations_have_usf_ip_range(self):
        """USF campus locations should use 131.247.x.x IP range"""
        campus_locations = [
            'ENG_Building', 'Admin_Building_ADM', 'Library_LIB',
            'Fine_Arts_FAH', 'USF_Health_MDC', 'Juniper_Poplar_Hall'
        ]
        
        for location in campus_locations:
            ip_prefix = create_logs.LOCATIONS[location]
            assert ip_prefix.startswith('131.247.'), f"{location} doesn't use USF IP range"


@pytest.mark.integration
@patch('create_logs.EventHubProducerClient')
class TestMainFunction:
    """Integration tests for main function"""
    
    def test_main_requires_event_hub_config(self, mock_producer_class):
        """Main should exit if Event Hub not configured"""
        # Test with default placeholder value
        original_conn = create_logs.EVENT_HUB_CONNECTION_STRING
        create_logs.EVENT_HUB_CONNECTION_STRING = "YOUR_EVENT_HUB_CONNECTION_STRING"
        
        # Should print error and return early
        create_logs.main()
        
        # Should not create producer with invalid config
        assert mock_producer_class.from_connection_string.call_count == 0
        
        # Restore original
        create_logs.EVENT_HUB_CONNECTION_STRING = original_conn
    
    def test_main_creates_producer_with_valid_config(self, mock_producer_class):
        """Main should create Event Hub producer with valid config"""
        # Mock valid configuration
        create_logs.EVENT_HUB_CONNECTION_STRING = "Endpoint=sb://test.servicebus.windows.net/..."
        create_logs.EVENT_HUB_NAME = "test-hub"
        
        # Mock producer
        mock_producer = Mock()
        mock_producer_class.from_connection_string.return_value = mock_producer
        
        # Patch time.sleep to avoid delays in test
        with patch('time.sleep'):
            with patch('create_logs.SIMULATION_DURATION_MINUTES', 0):  # Run for 0 minutes
                create_logs.main()
        
        # Should have created producer
        mock_producer_class.from_connection_string.assert_called_once()

