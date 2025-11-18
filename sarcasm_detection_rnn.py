"""
Sarcasm Detection with RNN Architectures
=========================================
This script implements three different RNN architectures for detecting sarcasm in news headlines:
1. SimpleRNN - Basic recurrent neural network
2. LSTM - Long Short-Term Memory network
3. BiLSTM - Bidirectional LSTM network

Dataset: News Headlines Dataset for Sarcasm Detection
Train/Test Split: 80:20
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, SimpleRNN, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("SARCASM DETECTION WITH RNN ARCHITECTURES")
print("=" * 80)

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================
print("\n[1] Loading and preprocessing data...")

with open("Sarcasm_Headlines_Dataset.json", 'r') as f:
    datastore = [json.loads(line) for line in f]

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Convert to numpy arrays
labels = np.array(labels)

print(f"   Total samples: {len(sentences)}")
print(f"   Sarcastic headlines: {sum(labels)} ({sum(labels)/len(labels)*100:.2f}%)")
print(f"   Non-sarcastic headlines: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.2f}%)")

# =============================================================================
# 2. TOKENIZATION AND PADDING
# =============================================================================
print("\n[2] Tokenizing and padding sequences...")

# Hyperparameters
VOCAB_SIZE = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 128
TRUNCATION_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATION_TYPE)

print(f"   Vocabulary size: {len(word_index)}")
print(f"   Using vocab size: {VOCAB_SIZE}")
print(f"   Max sequence length: {MAX_LENGTH}")
print(f"   Padded shape: {padded.shape}")
print(f"   Sample padded sequence: {padded[0][:20]}...")

# =============================================================================
# 3. TRAIN/TEST SPLIT (80:20)
# =============================================================================
print("\n[3] Splitting data into train/test sets (80:20)...")

X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")
print(f"   Train sarcastic ratio: {sum(y_train)/len(y_train)*100:.2f}%")
print(f"   Test sarcastic ratio: {sum(y_test)/len(y_test)*100:.2f}%")

# =============================================================================
# 4. MODEL ARCHITECTURES
# =============================================================================
print("\n[4] Building model architectures...")

# Training hyperparameters
EPOCHS = 20
BATCH_SIZE = 128
RNN_UNITS = 64
DROPOUT_RATE = 0.5

def create_simple_rnn_model():
    """Create a SimpleRNN model for sarcasm detection"""
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        SimpleRNN(RNN_UNITS, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ], name='SimpleRNN_Model')

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def create_lstm_model():
    """Create an LSTM model for sarcasm detection"""
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        LSTM(RNN_UNITS, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ], name='LSTM_Model')

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def create_bilstm_model():
    """Create a Bidirectional LSTM model for sarcasm detection"""
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        Bidirectional(LSTM(RNN_UNITS, return_sequences=False)),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ], name='BiLSTM_Model')

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

# Create models
models = {
    'SimpleRNN': create_simple_rnn_model(),
    'LSTM': create_lstm_model(),
    'BiLSTM': create_bilstm_model()
}

# Display model architectures
for name, model in models.items():
    print(f"\n   {name} Architecture:")
    print(f"   {'-' * 40}")
    model.summary()
    print()

# =============================================================================
# 5. TRAINING
# =============================================================================
print("\n[5] Training models...")

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

histories = {}
trained_models = {}

for name, model in models.items():
    print(f"\n{'=' * 80}")
    print(f"Training {name}...")
    print('=' * 80)

    # Create model checkpoint
    checkpoint = ModelCheckpoint(
        f'model_{name.lower()}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    histories[name] = history
    trained_models[name] = model

    print(f"\n{name} training completed!")

# =============================================================================
# 6. EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("[6] EVALUATING MODELS")
print("=" * 80)

results = {}

for name, model in trained_models.items():
    print(f"\n{name} Results:")
    print("-" * 40)

    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'probabilities': y_pred_prob
    }

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Sarcastic', 'Sarcastic']))

# =============================================================================
# 7. VISUALIZATION
# =============================================================================
print("\n[7] Generating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 7.1 Training History - Accuracy
ax1 = plt.subplot(2, 3, 1)
for name, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{name} Train', marker='o')
    plt.plot(history.history['val_accuracy'], label=f'{name} Val', marker='s', linestyle='--')
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# 7.2 Training History - Loss
ax2 = plt.subplot(2, 3, 2)
for name, history in histories.items():
    plt.plot(history.history['loss'], label=f'{name} Train', marker='o')
    plt.plot(history.history['val_loss'], label=f'{name} Val', marker='s', linestyle='--')
plt.title('Model Loss Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 7.3 Performance Metrics Comparison
ax3 = plt.subplot(2, 3, 3)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_names))
width = 0.25

for idx, (name, result) in enumerate(results.items()):
    metrics_values = [result['accuracy'], result['precision'], result['recall'], result['f1']]
    plt.bar(x + idx*width, metrics_values, width, label=name)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
plt.xticks(x + width, metrics_names)
plt.legend()
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')

# 7.4-7.6 Confusion Matrices
for idx, (name, result) in enumerate(results.items(), start=4):
    ax = plt.subplot(2, 3, idx)
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0.5, 1.5], ['Non-Sarcastic', 'Sarcastic'])
    plt.yticks([0.5, 1.5], ['Non-Sarcastic', 'Sarcastic'])

plt.tight_layout()
plt.savefig('rnn_models_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved visualization to: rnn_models_comparison.png")

# =============================================================================
# 8. MODEL COMPARISON SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

# Create comparison table
print(f"\n{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 63)
for name, result in results.items():
    print(f"{name:<15} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
          f"{result['recall']:<12.4f} {result['f1']:<12.4f}")

# Find best model
best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
print(f"\n🏆 Best Model: {best_model_name} (based on F1-Score)")

# =============================================================================
# 9. PREDICTION FUNCTION
# =============================================================================
print("\n[9] Setting up prediction function...")

def predict_sarcasm(headline, model_name='BiLSTM'):
    """
    Predict if a headline is sarcastic or not

    Args:
        headline (str): The headline text to analyze
        model_name (str): Which model to use ('SimpleRNN', 'LSTM', or 'BiLSTM')

    Returns:
        dict: Prediction results including probability and label
    """
    model = trained_models[model_name]

    # Preprocess the headline
    sequence = tokenizer.texts_to_sequences([headline])
    padded_seq = pad_sequences(sequence, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATION_TYPE)

    # Predict
    prob = model.predict(padded_seq, verbose=0)[0][0]
    is_sarcastic = prob > 0.5

    return {
        'headline': headline,
        'model': model_name,
        'probability': float(prob),
        'is_sarcastic': bool(is_sarcastic),
        'label': 'Sarcastic' if is_sarcastic else 'Non-Sarcastic',
        'confidence': float(prob if is_sarcastic else 1-prob)
    }

# =============================================================================
# 10. DEMO PREDICTIONS
# =============================================================================
print("\n" + "=" * 80)
print("DEMO PREDICTIONS")
print("=" * 80)

# Test headlines
test_headlines = [
    "Area Man Knows All The Shortcut Keys",
    "Trump Announces New Immigration Policy At Border",
    "Scientists Discover Water On Mars",
    "Local Idiot To Post Comment On Internet",
    "Stock Market Reaches Record High",
    "Nation Demands New Season Of 'Black Mirror' Right Fucking Now",
]

print("\nTesting predictions on sample headlines:\n")

for headline in test_headlines:
    print(f"Headline: \"{headline}\"")
    print("-" * 80)

    for model_name in ['SimpleRNN', 'LSTM', 'BiLSTM']:
        result = predict_sarcasm(headline, model_name)
        print(f"  {model_name:<10} → {result['label']:<15} (confidence: {result['confidence']:.2%})")
    print()

# =============================================================================
# 11. SAVE MODELS
# =============================================================================
print("\n[11] Saving final models...")

for name, model in trained_models.items():
    filename = f'model_{name.lower()}_final.h5'
    model.save(filename)
    print(f"   Saved {name} to: {filename}")

# Save tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("   Saved tokenizer to: tokenizer.pickle")

print("\n" + "=" * 80)
print("SARCASM DETECTION TRAINING COMPLETE!")
print("=" * 80)
print(f"\nBest performing model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.4%}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
print("\nFiles generated:")
print("  - rnn_models_comparison.png (visualization)")
print("  - model_simplernn_final.h5 (SimpleRNN model)")
print("  - model_lstm_final.h5 (LSTM model)")
print("  - model_bilstm_final.h5 (BiLSTM model)")
print("  - tokenizer.pickle (tokenizer)")
print("=" * 80)
