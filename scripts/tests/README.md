# Python Scripts Test Suite

Comprehensive unit and integration tests for the ML server and data processing scripts.

## Test Structure

```
tests/
├── conftest.py                      # Pytest fixtures and configuration
├── test_ml_server_dual.py           # ML server dual-head LSTM tests
├── test_create_logs.py              # Log generation tests
├── test_upload_logs_to_blob.py      # Azure Blob upload tests
└── README.md                        # This file
```

## Setup

### Install Test Dependencies

```bash
cd scripts
pip install -r test_requirements.txt
```

### Required Dependencies

- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `pytest-asyncio` - Async test support
- Mocking libraries for Azure and ML components

## Running Tests

### Run All Tests

```bash
cd scripts
pytest
```

### Run Specific Test File

```bash
pytest tests/test_ml_server_dual.py
pytest tests/test_create_logs.py
pytest tests/test_upload_logs_to_blob.py
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
```

View HTML report:
```bash
open htmlcov/index.html   # macOS
start htmlcov/index.html  # Windows
```

### Run with Verbose Output

```bash
pytest -v
```

### Run Specific Test

```bash
pytest tests/test_ml_server_dual.py::TestDualHeadLSTM::test_model_initialization
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)

Test individual functions and classes in isolation:

- **ML Server Tests**
  - Model architecture initialization
  - Session grouping logic
  - Pattern detection algorithms
  - Prediction endpoint validation
  - Model loading functionality

- **Log Generation Tests**
  - User generation
  - Sign-in log creation
  - Persona configuration
  - Location mapping
  - Probability distributions

- **Upload Tests**
  - Blob deletion
  - Log file chunking
  - Upload operations
  - Error handling
  - Configuration validation

### Integration Tests (`@pytest.mark.integration`)

Test complete workflows:

- End-to-end log generation and upload
- ML server prediction pipeline
- Azure service integration

## Test Fixtures

### Available Fixtures (from `conftest.py`)

- `mock_azure_blob_client` - Mock Azure Blob Storage client
- `sample_log_entry` - Sample log entry for testing
- `sample_logs_sequence` - Sequence of logs for ML testing
- `mock_lstm_model` - Mock PyTorch LSTM model
- `mock_encoders` - Mock LabelEncoders
- `mock_model_config` - Mock model configuration
- `mock_event_hub_producer` - Mock Azure Event Hub producer
- `sample_user` - Sample user data
- `mock_flask_app` - Mock Flask app for API testing
- `reset_environment` - Auto-reset environment variables

## Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| ML Server | 80%+ lines |
| Log Generation | 85%+ lines |
| Upload Scripts | 80%+ lines |
| **Overall** | **80%+ lines** |

## Current Test Coverage

Run `pytest --cov=. --cov-report=term-missing` to see detailed coverage.

## Writing New Tests

### Test Naming Convention

```python
class TestComponentName:
    """Tests for ComponentName"""
    
    def test_specific_behavior(self):
        """Should do X when Y happens"""
        # Arrange
        # Act
        # Assert
```

### Using Fixtures

```python
def test_with_fixture(sample_log_entry):
    """Test using a fixture"""
    assert sample_log_entry['appDisplayName'] == 'Canvas'
```

### Mocking External Dependencies

```python
@patch('module.external_dependency')
def test_with_mock(mock_dependency):
    """Test with mocked dependency"""
    mock_dependency.return_value = expected_value
    result = function_under_test()
    assert result == expected_output
```

## Common Test Patterns

### Testing Flask Endpoints

```python
def test_endpoint(self):
    with ml_server_dual.app.test_client() as client:
        response = client.post('/predict', json={'logs': []})
        assert response.status_code == 400
```

### Testing Azure Operations

```python
@patch('module.BlobServiceClient')
def test_azure_operation(self, mock_blob_service):
    mock_container = Mock()
    mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container
    # Test logic
```

### Testing Time-Dependent Code

```python
@patch('module.datetime')
def test_with_time(self, mock_datetime):
    mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, 0)
    # Test logic
```

## Troubleshooting

### Import Errors

If you get import errors, ensure the parent directory is in the Python path:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Mock Not Working

Ensure you're patching at the point of use, not definition:

```python
# Wrong: @patch('torch.nn.Module')
# Right: @patch('ml_server_dual.torch.nn.Module')
```

### Environment Variables

Tests automatically reset environment variables between runs via the `reset_environment` fixture.

## CI/CD Integration

Tests can be integrated into GitHub Actions:

```yaml
- name: Install test dependencies
  run: |
    cd scripts
    pip install -r test_requirements.txt

- name: Run tests
  run: |
    cd scripts
    pytest --cov=. --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./scripts/coverage.xml
```

## Best Practices

1. ✅ **Test Isolation** - Each test should be independent
2. ✅ **Mock External Services** - Don't call real Azure APIs in tests
3. ✅ **Use Fixtures** - Reuse common test data
4. ✅ **Clear Test Names** - Describe what's being tested
5. ✅ **Arrange-Act-Assert** - Structure tests clearly
6. ✅ **One Assertion Per Test** - Focus on one behavior
7. ✅ **Fast Tests** - Unit tests should run in milliseconds
8. ✅ **Deterministic** - Tests should always pass or always fail

## Questions?

For questions about the test suite, contact the Team 7 development team.

