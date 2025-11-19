"""
Final fix: Remove :0 suffix and fix bidirectional weight names to match TensorFlow.js format
"""
import json
import os

def fix_model_weights(model_path, model_type):
    """
    Fix weight names to match TensorFlow.js format exactly
    """
    json_file = os.path.join(model_path, 'model.json')

    if not os.path.exists(json_file):
        print(f"  ✗ {json_file} not found")
        return False

    print(f"\nFixing: {json_file}")

    # Load model.json
    with open(json_file, 'r') as f:
        model_data = json.load(f)

    # Update weight names
    if 'weightsManifest' in model_data:
        for manifest_entry in model_data['weightsManifest']:
            if 'weights' in manifest_entry:
                updated_weights = []

                for weight in manifest_entry['weights']:
                    old_name = weight['name']
                    new_name = old_name

                    # Remove :0 suffix
                    if new_name.endswith(':0'):
                        new_name = new_name[:-2]

                    # Fix bidirectional layer weight names
                    if model_type == 'bilstm' and 'bidirectional_1/' in new_name:
                        # Map our flat names to TensorFlow.js nested names
                        if new_name == 'bidirectional_1/kernel':
                            new_name = 'bidirectional_1/forward_lstm_LSTM1/kernel'
                        elif new_name == 'bidirectional_1/bias':
                            new_name = 'bidirectional_1/forward_lstm_LSTM1/recurrent_kernel'
                        elif new_name == 'bidirectional_1/weight_2':
                            new_name = 'bidirectional_1/forward_lstm_LSTM1/bias'
                        elif new_name == 'bidirectional_1/weight_3':
                            new_name = 'bidirectional_1/backward_lstm_LSTM1/kernel'
                        elif new_name == 'bidirectional_1/weight_4':
                            new_name = 'bidirectional_1/backward_lstm_LSTM1/recurrent_kernel'
                        elif new_name == 'bidirectional_1/weight_5':
                            new_name = 'bidirectional_1/backward_lstm_LSTM1/bias'

                    # Fix LSTM layer weight names
                    if model_type == 'lstm' and 'lstm_1/' in new_name:
                        if new_name == 'lstm_1/kernel':
                            new_name = 'lstm_1/kernel'
                        elif new_name == 'lstm_1/bias':
                            new_name = 'lstm_1/recurrent_kernel'
                        elif new_name == 'lstm_1/weight_2':
                            new_name = 'lstm_1/bias'

                    # Fix SimpleRNN layer weight names
                    if model_type == 'simplernn' and 'simplernn_1/' in new_name:
                        if new_name == 'simplernn_1/kernel':
                            new_name = 'simplernn_1/kernel'
                        elif new_name == 'simplernn_1/bias':
                            new_name = 'simplernn_1/recurrent_kernel'
                        elif new_name == 'simplernn_1/weight_2':
                            new_name = 'simplernn_1/bias'

                    weight['name'] = new_name
                    print(f"    {old_name} -> {new_name}")

                    updated_weights.append(weight)

                manifest_entry['weights'] = updated_weights

    # Save
    with open(json_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"  ✓ Fixed!")
    return True

def main():
    print("=" * 80)
    print("FINAL WEIGHT NAME FIX FOR TENSORFLOW.JS")
    print("=" * 80)

    models = [
        ('docs/models/bilstm', 'bilstm'),
        ('docs/models/lstm', 'lstm'),
        ('docs/models/simplernn', 'simplernn')
    ]

    for model_path, model_type in models:
        fix_model_weights(model_path, model_type)

    print("\n" + "=" * 80)
    print("DONE! Test with: node test_web_models.js")
    print("=" * 80)

if __name__ == '__main__':
    main()
