# Sarcasm Detection with RNN Architectures

## Overview

This implementation provides a comprehensive comparison of three different Recurrent Neural Network (RNN) architectures for detecting sarcasm in news headlines:

1. **SimpleRNN** - Basic recurrent neural network
2. **LSTM** - Long Short-Term Memory network
3. **BiLSTM** - Bidirectional LSTM network

## Architecture Details

### 1. SimpleRNN Model

**Architecture:**
```
Embedding(10000, 128) → SimpleRNN(64) → Dropout(0.5) → Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid)
```

**Characteristics:**
- Simplest RNN architecture
- Processes sequences unidirectionally (left to right)
- Prone to vanishing gradient problem with long sequences
- Fastest to train
- Good baseline for comparison

**Advantages:**
- Simple and fast
- Fewer parameters to train
- Good for short sequences

**Disadvantages:**
- Limited memory of past information
- Struggles with long-term dependencies
- Gradient vanishing issues

### 2. LSTM Model

**Architecture:**
```
Embedding(10000, 128) → LSTM(64) → Dropout(0.5) → Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid)
```

**Characteristics:**
- Uses cell state and three gates (input, forget, output)
- Better at capturing long-term dependencies
- Solves vanishing gradient problem
- Moderate training time

**Advantages:**
- Excellent memory retention
- Handles long-term dependencies well
- More stable training than SimpleRNN
- Industry standard for sequence tasks

**Disadvantages:**
- More parameters than SimpleRNN
- Slower training
- Only processes sequences in one direction

### 3. BiLSTM Model

**Architecture:**
```
Embedding(10000, 128) → Bidirectional(LSTM(64)) → Dropout(0.5) → Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid)
```

**Characteristics:**
- Processes sequences in both directions (forward and backward)
- Double the LSTM units (forward + backward)
- Captures context from both past and future
- Slowest to train but often best performance

**Advantages:**
- Captures bidirectional context
- Best for understanding sentence meaning
- Superior performance on many NLP tasks
- Understands word relationships better

**Disadvantages:**
- Most parameters (double LSTM)
- Longest training time
- Higher memory requirements

## Hyperparameters

```python
# Vocabulary
VOCAB_SIZE = 10000          # Top 10,000 most frequent words
MAX_LENGTH = 100            # Maximum sequence length
EMBEDDING_DIM = 128         # Word embedding dimension

# Model
RNN_UNITS = 64             # Number of RNN units
DROPOUT_RATE = 0.5         # Dropout rate for regularization

# Training
EPOCHS = 20                # Maximum epochs
BATCH_SIZE = 128           # Batch size
LEARNING_RATE = 0.001      # Adam optimizer learning rate
```

## Training Strategy

### Data Split
- **Train Set:** 80% (22,895 samples)
- **Test Set:** 20% (5,724 samples)
- **Stratified Split:** Maintains class balance

### Callbacks
1. **EarlyStopping**
   - Monitors: validation loss
   - Patience: 3 epochs
   - Restores best weights

2. **ModelCheckpoint**
   - Monitors: validation accuracy
   - Saves best model only

### Optimization
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Binary crossentropy
- **Metrics:** Accuracy

## Expected Results

Based on similar implementations and the dataset characteristics, you can expect:

### Performance Metrics

| Model      | Accuracy | Precision | Recall | F1-Score | Training Time |
|------------|----------|-----------|--------|----------|---------------|
| SimpleRNN  | ~78-82%  | ~0.78-0.82| ~0.75-0.80| ~0.76-0.81| Fastest      |
| LSTM       | ~82-86%  | ~0.82-0.86| ~0.80-0.84| ~0.81-0.85| Medium       |
| BiLSTM     | ~85-89%  | ~0.85-0.89| ~0.83-0.87| ~0.84-0.88| Slowest      |

**Note:** Actual results may vary based on random initialization and hardware.

### Performance Analysis

**SimpleRNN:**
- Good baseline performance
- Fast inference
- May struggle with subtle sarcasm requiring context

**LSTM:**
- Significant improvement over SimpleRNN
- Better at understanding context
- Good balance of speed and accuracy

**BiLSTM:**
- Best overall performance
- Captures bidirectional context
- Best for production deployment

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Usage

### 1. Training Models

```bash
python sarcasm_detection_rnn.py
```

**Output:**
- `model_simplernn_final.h5` - Trained SimpleRNN model
- `model_lstm_final.h5` - Trained LSTM model
- `model_bilstm_final.h5` - Trained BiLSTM model
- `tokenizer.pickle` - Fitted tokenizer
- `rnn_models_comparison.png` - Visualization of results

### 2. Making Predictions

**Single Prediction:**
```bash
python predict_sarcasm.py "Area Man Knows All The Shortcut Keys"
```

**Interactive Mode:**
```bash
python predict_sarcasm.py --interactive
```

**Compare All Models:**
```bash
python predict_sarcasm.py --compare "Nation Demands New Season Of Black Mirror"
```

**Use Specific Model:**
```bash
python predict_sarcasm.py --model model_lstm_final.h5 "Your headline here"
```

### 3. Using in Python

```python
from predict_sarcasm import SarcasmDetector

# Initialize detector
detector = SarcasmDetector(model_path='model_bilstm_final.h5')

# Single prediction
result = detector.predict("Area Man Knows All The Shortcut Keys")
print(f"Sarcastic: {result['is_sarcastic']} (confidence: {result['confidence']:.2%})")

# Batch predictions
headlines = [
    "Scientists Discover Water On Mars",
    "Local Idiot To Post Comment On Internet"
]
results = detector.predict_batch(headlines)
```

## Understanding the Output

### Prediction Dictionary

```python
{
    'headline': 'Area Man Knows All The Shortcut Keys',
    'probability': 0.9234,  # Raw probability (0-1)
    'is_sarcastic': True,   # Boolean prediction
    'label': 'Sarcastic 😏', # Human-readable label
    'confidence': 0.9234    # Confidence in prediction (0.5-1.0)
}
```

### Interpreting Probabilities

- **0.0 - 0.3:** Strongly non-sarcastic
- **0.3 - 0.4:** Likely non-sarcastic
- **0.4 - 0.6:** Uncertain (around decision boundary)
- **0.6 - 0.7:** Likely sarcastic
- **0.7 - 1.0:** Strongly sarcastic

## Key Implementation Details

### 1. Data Preprocessing

```python
# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# Sequence conversion
sequences = tokenizer.texts_to_sequences(sentences)

# Padding
padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
```

### 2. Embedding Layer

- Converts integer tokens to dense vectors
- Dimension: 128
- Learned during training
- Captures semantic relationships

### 3. Dropout Regularization

- Rate: 0.5 (50% dropout)
- Prevents overfitting
- Applied after RNN and Dense layers

### 4. Output Layer

- Single neuron with sigmoid activation
- Outputs probability [0, 1]
- Threshold: 0.5 for classification

## Evaluation Metrics

### 1. Accuracy
- Overall correctness
- (TP + TN) / (TP + TN + FP + FN)

### 2. Precision
- Accuracy of positive predictions
- TP / (TP + FP)
- "Of all predicted sarcastic, how many were actually sarcastic?"

### 3. Recall
- Coverage of actual positives
- TP / (TP + FN)
- "Of all actual sarcastic headlines, how many did we catch?"

### 4. F1-Score
- Harmonic mean of precision and recall
- 2 × (Precision × Recall) / (Precision + Recall)
- **Best metric for imbalanced datasets**

## Visualization

The script generates a comprehensive visualization (`rnn_models_comparison.png`) with:

1. **Training Accuracy:** Shows learning curves for all models
2. **Training Loss:** Shows loss reduction over epochs
3. **Metrics Comparison:** Bar chart of all performance metrics
4. **Confusion Matrices:** Visual representation of predictions for each model

## Why These Architectures?

### SimpleRNN
- **Educational:** Understanding basic RNN mechanics
- **Baseline:** Comparison point for improvements
- **Speed:** Quick experimentation

### LSTM
- **Industry Standard:** Proven for NLP tasks
- **Long-term Dependencies:** Better memory than SimpleRNN
- **Reliable:** Stable and well-understood

### BiLSTM
- **State-of-the-Art:** Best for text understanding
- **Bidirectional Context:** Sees whole sentence
- **Production Ready:** Often best performance

## Common Issues and Solutions

### 1. Out of Memory
**Solution:** Reduce batch size or model size
```python
BATCH_SIZE = 64  # Reduce from 128
RNN_UNITS = 32   # Reduce from 64
```

### 2. Overfitting
**Symptoms:** High train accuracy, low validation accuracy
**Solutions:**
- Increase dropout rate
- Add L2 regularization
- Reduce model complexity
- Get more training data

### 3. Underfitting
**Symptoms:** Low accuracy on both train and validation
**Solutions:**
- Increase model complexity (more units/layers)
- Train for more epochs
- Reduce dropout rate
- Increase vocabulary size

### 4. Slow Training
**Solutions:**
- Reduce max length: `MAX_LENGTH = 50`
- Use smaller vocab: `VOCAB_SIZE = 5000`
- Reduce batch size may help with memory but slower
- Use GPU if available

## Further Improvements

### 1. Pre-trained Embeddings
Use GloVe or Word2Vec embeddings:
```python
# Load pre-trained embeddings
embedding_matrix = load_glove_embeddings()
Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)
```

### 2. Attention Mechanism
Add attention to focus on important words:
```python
from tensorflow.keras.layers import Attention
# Add after LSTM layer
```

### 3. Data Augmentation
- Back-translation
- Synonym replacement
- Random insertion/deletion

### 4. Ensemble Methods
Combine predictions from all three models:
```python
ensemble_pred = (pred_rnn + pred_lstm + pred_bilstm) / 3
```

### 5. Hyperparameter Tuning
Use Keras Tuner or similar:
- Learning rate
- RNN units
- Dropout rate
- Embedding dimension

## References

### Papers
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
- Schuster & Paliwal (1997): "Bidirectional Recurrent Neural Networks"

### Resources
- [Keras Documentation](https://keras.io/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## License

This implementation is for educational purposes. The dataset license applies to the data files.

## Author

Created as part of the News Headlines Dataset for Sarcasm Detection project.


