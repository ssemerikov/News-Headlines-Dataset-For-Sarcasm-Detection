# Model Conversion Guide for Web Deployment

This guide explains how to convert Keras/TensorFlow models to TensorFlow.js format for running in the browser.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Best Practices for Model Creation](#best-practices-for-model-creation)
4. [Understanding the Problem](#understanding-the-problem)
5. [Solution: Step-by-Step Conversion](#solution-step-by-step-conversion)
6. [Testing Your Models](#testing-your-models)
7. [Troubleshooting](#troubleshooting)

---

## Overview

**Goal**: Convert your trained Keras models (`.h5` files) to TensorFlow.js format so they can run in a web browser.

**What you need**:
- Trained Keras models (`.h5` files)
- Tokenizer (`.pickle` file)
- Python environment with TensorFlow/Keras
- Optionally: Node.js for testing

**What you'll get**:
- `model.json` files (model architecture)
- `.bin` files (model weights)
- `word_index.json` (vocabulary)
- `config.json` (preprocessing parameters)

---

## Prerequisites

### Required Software

```bash
# Check Python version (need 3.8+)
python3 --version

# Check if TensorFlow is installed
python3 -c "import tensorflow as tf; print(tf.__version__)"

# Check if Keras is installed
python3 -c "import keras; print(keras.__version__)"
```

### File Structure

Before starting, your directory should look like:

```
your-project/
├── model_bilstm_best.h5      # Trained BiLSTM model
├── model_lstm_best.h5         # Trained LSTM model
├── model_simplernn_best.h5    # Trained SimpleRNN model
├── tokenizer.pickle           # Tokenizer with vocabulary
└── docs/                      # Output directory for web files
    └── models/                # Will contain converted models
```

---

## Best Practices for Model Creation

To make model conversion easier, follow these guidelines when creating and saving models in Python:

### 1. Use Compatible TensorFlow/Keras Versions

**Recommended Setup:**
```bash
# For easiest conversion (Keras 2.x)
pip install tensorflow==2.15.0

# Or current versions (requires more conversion work)
pip install tensorflow>=2.16.0
```

**Why it matters:** Keras 3 (included in TensorFlow 2.16+) uses a different model format that requires additional conversion steps.

### 2. Save Models in Multiple Formats

When training, save your model in multiple formats for flexibility:

```python
# Train your model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save in H5 format (easiest to convert)
model.save('model_best.h5')

# Also save in SavedModel format
model.save('saved_models/my_model')

# Export tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)
```

### 3. Use Simple, Web-Compatible Architectures

**Preferred layer types** (well-supported in TensorFlow.js):
- Embedding
- LSTM, GRU, SimpleRNN
- Bidirectional (with LSTM/GRU)
- Dense
- Dropout
- Conv1D, Conv2D
- MaxPooling1D, MaxPooling2D
- Flatten

**Avoid or use with caution:**
- Custom layers (won't convert)
- Lambda layers (won't convert)
- Very deep bidirectional layers (slow in browser)
- Batch normalization after RNN layers (can cause issues)

### 4. Keep Models Browser-Friendly

**Size considerations:**
```python
# Smaller models load faster in browsers
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128),  # 128 instead of 256
    tf.keras.layers.LSTM(64),                     # 64 instead of 128
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

**Target sizes:**
- Vocabulary: 5,000-10,000 words (not 50,000+)
- Embedding dim: 64-128 (not 300+)
- LSTM units: 32-128 (not 512+)
- Total model size: <15 MB ideally

### 5. Export Configuration with Models

Always export preprocessing parameters:

```python
import json

# Save tokenizer configuration
config = {
    "vocabSize": vocab_size,
    "maxLength": max_length,
    "embeddingDim": embedding_dim,
    "paddingType": "post",
    "truncationType": "post",
    "oovToken": "<OOV>",
    "oovIndex": 1
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Export word index
word_index = {word: idx for word, idx in tokenizer.word_index.items()
              if idx < vocab_size}

with open('word_index.json', 'w') as f:
    json.dump(word_index, f)
```

### 6. Test Before Converting

Before spending time on conversion, verify your model:

```python
# Test prediction
test_text = "Example headline"
sequence = tokenizer.texts_to_sequences([test_text])
padded = tf.keras.preprocessing.sequence.pad_sequences(
    sequence, maxlen=max_length, padding='post')
prediction = model.predict(padded)

print(f"Prediction: {prediction[0][0]:.4f}")

# Check model summary
model.summary()

# Verify model size
import os
size_mb = os.path.getsize('model_best.h5') / (1024 * 1024)
print(f"Model size: {size_mb:.2f} MB")
```

### 7. Use Sequential Models When Possible

Sequential models are easier to convert than Functional API models:

```python
# GOOD - Sequential (easier to convert)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=100),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# OKAY - Functional API (more complex)
inputs = tf.keras.Input(shape=(100,))
x = tf.keras.layers.Embedding(10000, 128)(inputs)
x = tf.keras.layers.LSTM(64)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

### 8. Document Your Model

Create a model card:

```python
model_info = {
    "name": "sarcasm_detector",
    "version": "1.0",
    "architecture": "BiLSTM",
    "vocab_size": 10000,
    "max_length": 100,
    "embedding_dim": 128,
    "training_accuracy": 0.95,
    "validation_accuracy": 0.89,
    "input_format": "padded sequence of integers",
    "output_format": "probability (0-1)"
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
```

---

## Understanding the Problem

### Why Conversion is Needed

Keras/TensorFlow models use Python and can't run directly in browsers. TensorFlow.js allows JavaScript to load and run these models client-side.

### Common Issues

**Keras 3 Compatibility**: Keras 3.x models have a different config format than TensorFlow.js expects. You may see:
- "className' and 'config' must be set" errors
- "Improper config format" errors
- Input tensor count mismatch errors

**Solution**: Use compatibility methods to convert properly.

---

## Solution: Step-by-Step Conversion

### Method 1: Using Official TensorFlow.js Converter (Recommended if it works)

#### Step 1: Install TensorFlow.js Converter

```bash
pip install tensorflowjs
```

**Note**: If you get dependency errors, this method may not work with Keras 3. Skip to Method 2.

#### Step 2: Convert Models

```bash
# Convert SavedModel to TensorFlow.js
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_layers_model \
  saved_models/bilstm \
  docs/models/bilstm

# Repeat for other models
tensorflowjs_converter \
  --input_format=tf_saved_model \
  saved_models/lstm \
  docs/models/lstm

tensorflowjs_converter \
  --input_format=tf_saved_model \
  saved_models/simplernn \
  docs/models/simplernn
```

---

### Method 2: Manual Conversion with Compatibility Fix (Works with Keras 3)

This method manually creates TensorFlow.js-compatible model files.

#### Step 1: Export Tokenizer Data

Create `export_tokenizer.py`:

```python
import os
import json
import pickle

# Load tokenizer
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Export word index (limit to vocab size)
VOCAB_SIZE = 10000
word_index = {word: idx for word, idx in tokenizer.word_index.items()
              if idx < VOCAB_SIZE}

# Save configuration
config = {
    "vocabSize": VOCAB_SIZE,
    "maxLength": 100,
    "embeddingDim": 128,
    "paddingType": "post",
    "truncationType": "post",
    "oovToken": "<OOV>",
    "oovIndex": 1,
    "filters": "!\"#$%&()*+,-./:;<=>?@[\\\\]^_`{|}~\\t\\n",
    "lowercase": True
}

# Create output directory
os.makedirs('docs/models', exist_ok=True)

# Save files
with open('docs/models/word_index.json', 'w') as f:
    json.dump(word_index, f)

with open('docs/models/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Exported word index: {len(word_index)} words")
print(f"✓ Saved config.json")
```

Run it:
```bash
python3 export_tokenizer.py
```

#### Step 2: Save Models in Compatible Format

The key is to save models in TensorFlow SavedModel format first:

Create `save_as_savedmodel.py`:

```python
import os
from tensorflow import keras
import tensorflow as tf

def save_model(h5_path, output_dir):
    print(f"Loading {h5_path}...")
    model = keras.models.load_model(h5_path)

    print(f"Saving to {output_dir}...")
    # Use TensorFlow's saved_model.save (not model.export)
    tf.saved_model.save(model, output_dir)

    print(f"✓ Saved {output_dir}\n")

# Convert all models
models = [
    ('model_bilstm_best.h5', 'saved_models/bilstm'),
    ('model_lstm_best.h5', 'saved_models/lstm'),
    ('model_simplernn_best.h5', 'saved_models/simplernn'),
]

for h5, sm_dir in models:
    if os.path.exists(h5):
        save_model(h5, sm_dir)
```

Run it:
```bash
python3 save_as_savedmodel.py
```

#### Step 3: Install TensorFlow.js Node Package

If you have Node.js installed:

```bash
npm install -g @tensorflowjs/tfjs-converter
```

#### Step 4: Convert SavedModels to TensorFlow.js

```bash
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_layers_model \
  saved_models/bilstm \
  docs/models/bilstm
```

Repeat for other models.

---

### Method 3: Workaround Using Pre-trained Model Loading

If conversion continues to fail due to Keras 3 compatibility, you can:

1. Load the models in browser using a proxy endpoint
2. Use ONNX as an intermediate format
3. Retrain models with Keras 2.x (TensorFlow 2.15)

---

## Testing Your Models

### Test in Node.js

Create `test_models.js`:

```javascript
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function testModel() {
    // Load config and word index
    const config = JSON.parse(fs.readFileSync('docs/models/config.json'));
    const wordIndex = JSON.parse(fs.readFileSync('docs/models/word_index.json'));

    // Load model
    const model = await tf.loadLayersModel('file://docs/models/bilstm/model.json');
    console.log('✓ Model loaded');

    // Test with sample input
    const input = tf.tensor2d([[1, 2, 3, ...]], [1, 100]);  // Preprocessed text
    const output = model.predict(input);
    const probability = (await output.data())[0];

    console.log(`Prediction: ${probability > 0.5 ? 'Sarcastic' : 'Non-Sarcastic'}`);
    console.log(`Probability: ${(probability * 100).toFixed(2)}%`);
}

testModel().catch(console.error);
```

Run:
```bash
node test_models.js
```

### Test in Browser

1. Start a local server:
   ```bash
   cd docs
   python3 -m http.server 8888
   ```

2. Open `http://localhost:8888` in your browser

3. Open browser console (F12) and check for:
   - "✓ Loaded bilstm model"
   - "✓ Loaded lstm model"
   - "✓ Loaded simplernn model"

4. Try entering a headline and clicking "Analyze"

---

## Troubleshooting

### Error: "className' and 'config' must be set"

**Cause**: Keras 3 model config format incompatible with TensorFlow.js

**Solution**:
1. Use `tf.saved_model.save()` instead of `model.export()`
2. OR retrain models with TensorFlow 2.15 (has Keras 2.x)
3. OR use Method 3 workarounds

### Error: "No module named 'tensorflow_decision_forests'"

**Cause**: Missing dependency for tensorflowjs

**Solution**:
```bash
pip install tensorflow-decision-forests ydf
```

### Error: "Input tensor count mismatch"

**Cause**: Model was saved as GraphModel instead of LayersModel

**Solution**:
- Use `--output_format=tfjs_layers_model` in converter
- In JavaScript, use `tf.loadLayersModel()` not `tf.loadGraphModel()`

### Error: "[Errno 28] No space left on device"

**Cause**: Not enough disk space for TensorFlow dependencies

**Solution**:
1. Clean up unused files:
   ```bash
   pip cache purge
   rm -rf ~/.cache/pip
   rm -rf saved_models/  # After conversion
   ```

2. Use a machine with more disk space

### Models Load But Give Wrong Predictions

**Cause**: Preprocessing mismatch between Python and JavaScript

**Solution**:
1. Verify `config.json` matches training parameters exactly
2. Check `word_index.json` has correct vocabulary
3. Ensure JavaScript preprocessing matches Python:
   - Same lowercase/filtering
   - Same tokenization
   - Same padding (post-padding to length 100)
   - Same OOV handling (index 1)

---

## Best Practices

### 1. Always Test Locally First

Don't deploy to GitHub Pages until you've tested locally and confirmed:
- Models load without errors
- Predictions make sense
- UI responds correctly

### 2. Check File Sizes

TensorFlow.js models are typically:
- Similar size to original H5 files (~5-15 MB each)
- If much larger or smaller, something went wrong

### 3. Version Compatibility

Record versions that worked:
```bash
# In requirements.txt
tensorflow==2.20.0  # Or whatever version worked
keras==3.12.0
tensorflowjs==4.22.0  # If using Python converter
```

### 4. Keep Original Models

Never delete your original `.h5` files - you may need to reconvert with different settings.

---

## Quick Reference

### File Locations After Conversion

```
docs/
├── models/
│   ├── config.json           # Preprocessing config
│   ├── word_index.json       # Vocabulary (161 KB)
│   ├── bilstm/
│   │   ├── model.json        # Model architecture (~3 KB)
│   │   └── group1-shard1of1.bin  # Weights (~5 MB)
│   ├── lstm/
│   │   ├── model.json
│   │   └── group1-shard1of1.bin
│   └── simplernn/
│       ├── model.json
│       └── group1-shard1of1.bin
└── .nojekyll                 # Prevents Jekyll processing
```

### Loading in JavaScript

```javascript
// Load model
const model = await tf.loadLayersModel('models/bilstm/model.json');

// Make prediction
const inputTensor = tf.tensor2d([[/* preprocessed sequence */]], [1, 100]);
const outputTensor = model.predict(inputTensor);
const probability = (await outputTensor.data())[0];

// Clean up
inputTensor.dispose();
outputTensor.dispose();
```

---

## Need Help?

If you're still having issues:

1. Check TensorFlow.js documentation: https://www.tensorflow.org/js/guide/conversion
2. Review the browser console for specific error messages
3. Verify your TensorFlow/Keras versions match the guide
4. Consider using Docker for a clean environment

## Summary Checklist

- [ ] Installed tensorflowjs or @tensorflowjs/tfjs-converter
- [ ] Exported tokenizer data (word_index.json, config.json)
- [ ] Converted models to SavedModel format
- [ ] Converted SavedModels to TensorFlow.js format
- [ ] Verified model.json and .bin files exist
- [ ] Tested models load in Node.js
- [ ] Tested models work in browser locally
- [ ] Ready to deploy to GitHub Pages!
