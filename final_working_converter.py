"""
Final working converter - saves models in TensorFlow SavedModel format (legacy),
then uses tensorflowjs_converter CLI to convert to TFJS
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import subprocess
from tensorflow import keras

def convert_h5_to_tfjs_via_savedmodel(h5_path, output_dir):
    """
    Convert H5 -> SavedModel (legacy format) -> TFJS
    This uses the legacy SavedModel format that TFJS understands
    """
    print(f"\nConverting: {h5_path}")

    # Load model
    model = keras.models.load_model(h5_path)

    # Save as legacy SavedModel (not using Keras 3's new export method)
    temp_sm_dir = output_dir + '_temp_sm'

    # Use legacy_export to get TF2.x compatible SavedModel
    print(f"  Saving as legacy SavedModel...")
    import tensorflow as tf
    tf.saved_model.save(model, temp_sm_dir)

    # Convert using tensorflowjs_converter CLI
    print(f"  Converting to TensorFlow.js...")
    cmd = [
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_layers_model',
        '--signature_name=serving_default',
        '--saved_model_tags=serve',
        temp_sm_dir,
        output_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ Conversion failed:")
        print(result.stderr)
        raise Exception(f"Conversion failed for {h5_path}")

    # Clean up temp
    import shutil
    shutil.rmtree(temp_sm_dir, ignore_errors=True)

    print(f"  ✓ Successfully converted to {output_dir}")

def main():
    print("=" * 80)
    print("FINAL WORKING TENSORFLOW.JS CONVERSION")
    print("=" * 80)

    models = [
        ('model_bilstm_best.h5', 'docs/models/bilstm'),
        ('model_lstm_best.h5', 'docs/models/lstm'),
        ('model_simplernn_best.h5', 'docs/models/simplernn'),
    ]

    for h5_file, output_dir in models:
        if os.path.exists(h5_file):
            try:
                convert_h5_to_tfjs_via_savedmodel(h5_file, output_dir)
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
        else:
            print(f"\n⚠ {h5_file} not found\n")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
