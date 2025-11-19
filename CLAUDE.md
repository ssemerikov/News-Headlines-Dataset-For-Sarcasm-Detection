# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sarcasm detection project using the News Headlines Dataset. The dataset contains 28,619 headlines from TheOnion (sarcastic) and HuffPost (non-sarcastic). The implementation compares three RNN architectures: SimpleRNN, LSTM, and BiLSTM for binary sarcasm classification.

## Core Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Training Models
```bash
# Train all three models (SimpleRNN, LSTM, BiLSTM) and generate comparison visualizations
python sarcasm_detection_rnn.py
```

This script:
- Loads and preprocesses `Sarcasm_Headlines_Dataset.json`
- Trains three models with 80/20 train/test split
- Generates `rnn_models_comparison.png` visualization
- Saves models as `model_{simplernn,lstm,bilstm}_final.h5` and `tokenizer.pickle`

### Making Predictions
```bash
# Single prediction with default BiLSTM model
python predict_sarcasm.py "Your headline here"

# Interactive mode for continuous predictions
python predict_sarcasm.py --interactive

# Compare predictions from all three models
python predict_sarcasm.py --compare "Your headline here"

# Use specific model
python predict_sarcasm.py --model model_lstm_final.h5 "Your headline here"
```

## Architecture

### Data Pipeline
1. **Dataset**: `Sarcasm_Headlines_Dataset.json` - Each line is a JSON object with `is_sarcastic` (0/1), `headline` (text), and `article_link` (URL)
2. **Preprocessing**: Tokenization (10K vocab, 100 max length) → Padding → 80/20 train/test split (stratified)
3. **Training**: Three parallel model training pipelines with early stopping and model checkpoints

### Model Architectures
All three models share the same structure except for the RNN layer:

```
Embedding(10000, 128) → RNN_LAYER → Dropout(0.5) → Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid)
```

- **SimpleRNN**: Basic RNN(64 units) - fastest, baseline performance
- **LSTM**: LSTM(64 units) - better long-term dependencies
- **BiLSTM**: Bidirectional(LSTM(64 units)) - best performance, processes both directions

### Key Hyperparameters
```python
VOCAB_SIZE = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 128
RNN_UNITS = 64
DROPOUT_RATE = 0.5
BATCH_SIZE = 128
EPOCHS = 20 (with early stopping, patience=3)
```

### Inference Module
`predict_sarcasm.py` contains `SarcasmDetector` class with:
- `predict(headline)`: Single prediction returning dict with probability, label, confidence
- `predict_batch(headlines)`: Batch predictions
- `interactive_mode()`: REPL for continuous testing

## Generated Artifacts

After training, the following files are created:
- `model_simplernn_final.h5` / `model_simplernn_best.h5`
- `model_lstm_final.h5` / `model_lstm_best.h5`
- `model_bilstm_final.h5` / `model_bilstm_best.h5`
- `tokenizer.pickle` - Required for all predictions
- `rnn_models_comparison.png` - Comprehensive visualization with accuracy/loss curves, metrics comparison, and confusion matrices

These files are gitignored (see `.gitignore` for model and output patterns).

## Dataset Structure

The dataset is balanced with 47.6% sarcastic and 52.4% non-sarcastic headlines. Headlines are professional and formal (from news sites), resulting in cleaner data compared to Twitter datasets - only 23.35% of words are not available in pre-trained embeddings.

## Important Implementation Details

### Reproducibility
Both scripts set random seeds:
```python
np.random.seed(42)
tf.random.set_seed(42)
```

### Prediction Format
All predictions return:
```python
{
    'headline': str,
    'probability': float,  # 0-1, raw sigmoid output
    'is_sarcastic': bool,  # probability > 0.5
    'label': str,          # "Sarcastic 😏" or "Non-Sarcastic 📰"
    'confidence': float    # max(prob, 1-prob), range 0.5-1.0
}
```

### Training Callbacks
- **EarlyStopping**: Monitors `val_loss`, patience=3, restores best weights
- **ModelCheckpoint**: Monitors `val_accuracy`, saves best only

## Working with This Codebase

### To add a new model architecture:
1. Add model creation function in `sarcasm_detection_rnn.py` (section 4)
2. Add to `models` dict around line 166-170
3. The training loop (section 5) will automatically train your new model

### To modify hyperparameters:
All hyperparameters are defined at the top of their respective sections in `sarcasm_detection_rnn.py`:
- Vocabulary/padding config: lines 66-71
- Model config: lines 106-109

### To use models programmatically:
```python
from predict_sarcasm import SarcasmDetector

detector = SarcasmDetector(model_path='model_bilstm_final.h5')
result = detector.predict("Your headline")
```

The tokenizer must be in the same directory as the script or provide `tokenizer_path` parameter.

## Expected Performance

Based on the dataset characteristics and architecture:
- SimpleRNN: ~78-82% accuracy, fastest training
- LSTM: ~82-86% accuracy, medium training time
- BiLSTM: ~85-89% accuracy, slowest but best performance

BiLSTM typically achieves the best F1-score and is recommended for production use.
