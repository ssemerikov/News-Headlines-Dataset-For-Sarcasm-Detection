# Sarcasm Detection Web Interface

A web-based sarcasm detection application powered by TensorFlow.js, featuring three different RNN architectures trained on the News Headlines Dataset.

## Features

- **Three RNN Models**: Compare SimpleRNN, LSTM, and BiLSTM architectures
- **Real-time Predictions**: Analyze headlines directly in your browser
- **Model Comparison**: View predictions from all three models side-by-side
- **Privacy-First**: All processing happens client-side - your data never leaves your browser
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices
- **Example Headlines**: Pre-loaded examples to test the models

## Architecture

### Models
- **SimpleRNN**: Basic RNN (~5 MB) - Fast baseline performance
- **LSTM**: Long Short-Term Memory (~5 MB) - Better at capturing context
- **BiLSTM**: Bidirectional LSTM (~5 MB) - Best overall performance

### Files Structure
```
gh-pages/
├── index.html              # Main HTML interface
├── css/
│   └── style.css          # Custom styling
├── js/
│   ├── app.js             # Main application logic
│   ├── preprocessor.js    # Text preprocessing
│   └── examples.js        # Example headlines
└── models/
    ├── config.json         # Preprocessing configuration
    ├── word_index.json     # Vocabulary mapping
    ├── bilstm/            # BiLSTM model files
    │   ├── model.json
    │   └── group1-shard1of1.bin
    ├── lstm/              # LSTM model files
    │   ├── model.json
    │   └── group1-shard1of1.bin
    └── simplernn/         # SimpleRNN model files
        ├── model.json
        └── group1-shard1of1.bin
```

## How It Works

1. **Text Preprocessing**: Input headlines are converted to lowercase, filtered, tokenized, and converted to sequences
2. **Padding**: Sequences are padded/truncated to 100 tokens
3. **Model Inference**: TensorFlow.js runs the model directly in the browser
4. **Results**: Displays prediction, confidence, and raw probability

## Dataset

Trained on 28,619 news headlines:
- **Sarcastic** (13,635): from TheOnion
- **Non-Sarcastic** (14,984): from HuffPost

## Performance

Expected accuracy ranges:
- SimpleRNN: ~78-82%
- LSTM: ~82-86%
- BiLSTM: ~85-89%

## Technologies

- **TensorFlow.js 4.15**: ML inference in browser
- **Bootstrap 5.3**: UI framework
- **Vanilla JavaScript**: No additional frameworks

## Usage

Simply open `index.html` in a modern browser. The models will load automatically (first load may take a few seconds).

### Examples:

**Sarcastic Headlines:**
- "Area Man Knows All The Shortcut Keys"
- "Local Idiot To Post Comment On Internet"

**Non-Sarcastic Headlines:**
- "Scientists Discover Water On Mars"
- "Stock Market Reaches Record High"

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

## License

This web interface is part of the News Headlines Dataset for Sarcasm Detection project. See main repository README for dataset license and citation information.

## Credits

- **Dataset**: Rishabh Misra & Prahal Arora
- **Web Interface**: Built with Claude Code
- **Powered by**: TensorFlow.js
