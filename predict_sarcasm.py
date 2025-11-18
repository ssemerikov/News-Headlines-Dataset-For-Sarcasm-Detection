"""
Sarcasm Prediction Utility
===========================
Load trained models and make predictions on new headlines.

Usage:
    python predict_sarcasm.py "Your headline here"
    python predict_sarcasm.py --interactive
"""

import sys
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNCATION_TYPE = 'post'

class SarcasmDetector:
    """Sarcasm detection using trained RNN models"""

    def __init__(self, model_path='model_bilstm_final.h5', tokenizer_path='tokenizer.pickle'):
        """
        Initialize the sarcasm detector

        Args:
            model_path (str): Path to the trained model file
            tokenizer_path (str): Path to the tokenizer pickle file
        """
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)

        print(f"Loading tokenizer from: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        print("Sarcasm detector ready!\n")

    def predict(self, headline):
        """
        Predict if a headline is sarcastic

        Args:
            headline (str): The headline to analyze

        Returns:
            dict: Prediction results
        """
        # Preprocess
        sequence = self.tokenizer.texts_to_sequences([headline])
        padded_seq = pad_sequences(sequence, maxlen=MAX_LENGTH,
                                   padding=PADDING_TYPE, truncating=TRUNCATION_TYPE)

        # Predict
        prob = self.model.predict(padded_seq, verbose=0)[0][0]
        is_sarcastic = prob > 0.5

        return {
            'headline': headline,
            'probability': float(prob),
            'is_sarcastic': bool(is_sarcastic),
            'label': 'Sarcastic 😏' if is_sarcastic else 'Non-Sarcastic 📰',
            'confidence': float(prob if is_sarcastic else 1-prob)
        }

    def predict_batch(self, headlines):
        """
        Predict for multiple headlines

        Args:
            headlines (list): List of headlines to analyze

        Returns:
            list: List of prediction results
        """
        results = []
        for headline in headlines:
            results.append(self.predict(headline))
        return results

    def interactive_mode(self):
        """Run in interactive mode for continuous predictions"""
        print("=" * 80)
        print("INTERACTIVE SARCASM DETECTION MODE")
        print("=" * 80)
        print("Enter headlines to check if they're sarcastic.")
        print("Type 'quit', 'exit', or press Ctrl+C to stop.\n")

        while True:
            try:
                headline = input("Enter headline: ").strip()

                if headline.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if not headline:
                    continue

                result = self.predict(headline)
                print(f"\n  {'─' * 70}")
                print(f"  Prediction:  {result['label']}")
                print(f"  Confidence:  {result['confidence']:.2%}")
                print(f"  Probability: {result['probability']:.4f}")
                print(f"  {'─' * 70}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Predict sarcasm in news headlines')
    parser.add_argument('headline', nargs='*', help='Headline to analyze')
    parser.add_argument('--model', default='model_bilstm_final.h5',
                       help='Path to model file (default: model_bilstm_final.h5)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--compare', action='store_true',
                       help='Compare predictions from all three models')

    args = parser.parse_args()

    # Initialize detector
    try:
        detector = SarcasmDetector(model_path=args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have trained the model first by running:")
        print("  python sarcasm_detection_rnn.py")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        detector.interactive_mode()
        return

    # Compare mode
    if args.compare and args.headline:
        headline = ' '.join(args.headline)
        print(f"Headline: \"{headline}\"\n")
        print("=" * 80)

        models = {
            'SimpleRNN': 'model_simplernn_final.h5',
            'LSTM': 'model_lstm_final.h5',
            'BiLSTM': 'model_bilstm_final.h5'
        }

        for name, model_path in models.items():
            try:
                det = SarcasmDetector(model_path=model_path)
                result = det.predict(headline)
                print(f"{name:<12} → {result['label']:<20} (confidence: {result['confidence']:.2%})")
            except Exception as e:
                print(f"{name:<12} → Error: {e}")

        print("=" * 80)
        return

    # Single prediction
    if args.headline:
        headline = ' '.join(args.headline)
        result = detector.predict(headline)

        print("=" * 80)
        print(f"Headline:    \"{result['headline']}\"")
        print(f"Prediction:  {result['label']}")
        print(f"Confidence:  {result['confidence']:.2%}")
        print(f"Probability: {result['probability']:.4f}")
        print("=" * 80)
    else:
        # No arguments, show usage
        parser.print_help()
        print("\nExamples:")
        print("  python predict_sarcasm.py \"Area Man Knows All The Shortcut Keys\"")
        print("  python predict_sarcasm.py --interactive")
        print("  python predict_sarcasm.py --compare \"Nation Demands New Season Of Black Mirror\"")


if __name__ == '__main__':
    main()
