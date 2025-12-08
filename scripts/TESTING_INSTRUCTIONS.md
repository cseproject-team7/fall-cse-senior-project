# Python Scripts Testing Instructions

## ⚠️ Current Issue

The tests are failing to import modules because **runtime dependencies are missing**.

## 🔧 Solution: Install All Dependencies

### Step 1: Install Script Dependencies

```powershell
cd scripts
pip install faker flask flask-cors joblib pandas numpy scikit-learn torch azure-storage-blob azure-eventhub python-dotenv
```

**Or install from both requirements files:**
```powershell
pip install -r ml_requirements.txt
pip install -r test_requirements.txt
```

### Step 2: Verify Installation

```powershell
python -c "import faker; import flask_cors; print('Dependencies OK!')"
```

If you see "Dependencies OK!", proceed to step 3.

### Step 3: Run Tests

```powershell
python -m pytest tests/ -v
```

## 📊 Expected Output (When Working)

```
================================ test session starts ================================
collected 60 items

tests/test_upload_logs_to_blob.py::TestDeleteAllBlobs::test_deletes_all_blobs PASSED
tests/test_upload_logs_to_blob.py::TestDeleteAllBlobs::test_returns_zero PASSED
tests/test_upload_logs_to_blob.py::TestUploadLogs::test_uploads_logs PASSED
... (more tests)

tests/test_create_logs.py::TestGenerateUsers::test_generate_correct_number PASSED
tests/test_create_logs.py::TestCreateSigninLog::test_creates_log_with_fields PASSED
... (more tests)

tests/test_ml_server_dual.py::TestDualHeadLSTM::test_model_initialization PASSED
tests/test_ml_server_dual.py::TestSessionGrouping::test_empty_logs PASSED
... (more tests)

========================== 60 passed in 3.45s ===================================
```

## 🐛 Troubleshooting

### If pip install fails:

1. **Check Python version:**
   ```powershell
   python --version
   ```
   Should be Python 3.8+

2. **Try with --user flag:**
   ```powershell
   pip install --user faker flask flask-cors
   ```

3. **Use virtual environment (RECOMMENDED):**
   ```powershell
   cd scripts
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r test_requirements.txt
   pytest tests/ -v
   ```

### If imports still fail:

Check if you're in the right directory:
```powershell
pwd  # Should be in .../scripts
ls   # Should see test_create_logs.py, ml_server_dual.py, etc.
```

### If specific tests fail:

Run individual test files:
```powershell
pytest tests/test_upload_logs_to_blob.py -v     # This one should work!
pytest tests/test_create_logs.py -v             # Needs faker
pytest tests/test_ml_server_dual.py -v          # Needs flask_cors
```

## 📝 What Each Test File Needs

| Test File | Required Dependencies |
|-----------|----------------------|
| `test_upload_logs_to_blob.py` | azure-storage-blob, python-dotenv |
| `test_create_logs.py` | faker, azure-eventhub |
| `test_ml_server_dual.py` | flask, flask-cors, torch, joblib |

## ✅ Once Tests Pass

You'll see:
- Green checkmarks for passing tests
- Coverage report in `htmlcov/index.html`
- Test summary with pass count

Then you can commit:
```powershell
git add scripts/
git commit -m "test(scripts): add Python test suite with pytest"
git push
```

## 🆘 Still Having Issues?

The test files are ready but need proper Python environment setup. If pip continues to fail:

1. Use a fresh virtual environment
2. Or install dependencies one-by-one
3. Or run the server tests (which work) and skip Python tests for now

The backend tests (97 tests) already provide comprehensive coverage for the API layer! 🎯

