---
description: How to run the log generation and pattern extraction pipeline
---

# Run Pattern Extraction Pipeline

This workflow guides you through generating synthetic logs and extracting patterns using Model Distillation (with Ollama).

## 1. Generate Logs
First, generate the raw Entra ID logs. This simulates 50,000 students.

```bash
# This will create raw_logs/logs.json
python3 model_training_pipeline/generate_logs.py
```

## 2. Setup Ollama (Local LLM)
If you haven't already, install and run Ollama.

1.  **Install**: Download from [ollama.com](https://ollama.com).
2.  **Run**: Start the application.
3.  **Pull Model**: Open a terminal and run:
    ```bash
    ollama pull llama3
    ```

## 3. Install Python Dependencies
Install the required libraries for the distillation script.

```bash
pip install ollama pandas scikit-learn
```

## 4. Run Distillation Pipeline
Run the `colab_distillation.py` script. This will:
1.  Sample 10,000 sessions.
2.  Use Ollama (Llama 3) to label them.
3.  Train a Random Forest classifier.
4.  Label the remaining millions of sessions.

```bash
python3 model_training_pipeline/colab_distillation.py --backend ollama --sample_size 10000
```

## 5. Output
The final dataset for training your LSTM/Transformer will be saved to:
`prepared_data/distilled_sequences.csv`
