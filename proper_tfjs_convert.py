"""
Proper conversion to TensorFlow.js using layers model format
This will create a format that TensorFlow.js can actually use
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import tensorflowjs as tfjs
from tensorflow import keras

def convert_model_properly(h5_path, output_dir):
    """Convert Keras model to TensorFlow.js LayersModel format"""
    print(f"\nConverting: {h5_path}")
    print(f"  Output: {output_dir}")

    # Load model
    print("  Loading model...")
    model = keras.models.load_model(h5_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert using tensorflowjs
    print("  Converting to TensorFlow.js format...")
    tfjs.converters.save_keras_model(model, output_dir)

    print(f"  ✓ Conversion complete!\n")

def main():
    print("=" * 80)
    print("PROPER TENSORFLOW.JS CONVERSION")
    print("=" * 80)

    models = [
        ('model_bilstm_best.h5', 'docs/models/bilstm'),
        ('model_lstm_best.h5', 'docs/models/lstm'),
        ('model_simplernn_best.h5', 'docs/models/simplernn'),
    ]

    for h5_file, output_dir in models:
        if os.path.exists(h5_file):
            try:
                convert_model_properly(h5_file, output_dir)
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n⚠ {h5_file} not found\n")

    print("=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
