"""
Fix Bidirectional layer to match TensorFlow.js format
"""
import json
import os

def fix_bilstm(model_path):
    """
    Simplify Bidirectional layer config to match TensorFlow.js format
    """
    json_file = os.path.join(model_path, 'model.json')

    if not os.path.exists(json_file):
        print(f"  ✗ {json_file} not found")
        return False

    print(f"\nFixing: {json_file}")

    # Load model.json
    with open(json_file, 'r') as f:
        model_data = json.load(f)

    # Find and fix Bidirectional layer
    topology = model_data.get('modelTopology', {})
    config = topology.get('config', {})
    layers = config.get('layers', [])

    for i, layer in enumerate(layers):
        if layer.get('className') == 'Bidirectional':
            print("  Found Bidirectional layer, simplifying...")

            # Extract the forward layer config
            forward_layer = layer['config']['layer']

            # Create simplified config (TensorFlow.js will create backward layer automatically)
            new_config = {
                'merge_mode': 'concat',
                'layer': {
                    'class_name': 'LSTM',
                    'config': {
                        'name': 'lstm_LSTM1',  # TensorFlow.js naming convention
                        'trainable': True,
                        'units': 64,
                        'activation': 'tanh',
                        'recurrent_activation': 'hard_sigmoid',
                        'use_bias': True,
                        'kernel_initializer': forward_layer['config']['kernelInitializer'],
                        'recurrent_initializer': forward_layer['config']['recurrentInitializer'],
                        'bias_initializer': forward_layer['config']['biasInitializer'],
                        'unit_forget_bias': True,
                        'kernel_regularizer': None,
                        'recurrent_regularizer': None,
                        'bias_regularizer': None,
                        'activity_regularizer': None,
                        'kernel_constraint': None,
                        'recurrent_constraint': None,
                        'bias_constraint': None,
                        'dropout': 0.0,
                        'recurrent_dropout': 0.0,
                        'implementation': None,
                        'return_sequences': False,
                        'return_state': False,
                        'go_backwards': False,
                        'stateful': False,
                        'unroll': False
                    }
                },
                'name': 'bidirectional_1',
                'trainable': True
            }

            # Update layer
            layer['config'] = new_config

            print("  ✓ Bidirectional layer simplified")
            break

    # Save
    with open(json_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"  ✓ Fixed!")
    return True

def main():
    print("=" * 80)
    print("FIXING BIDIRECTIONAL LAYER FORMAT")
    print("=" * 80)

    fix_bilstm('docs/models/bilstm')

    print("\n" + "=" * 80)
    print("DONE! Test with: node test_bilstm.js")
    print("=" * 80)

if __name__ == '__main__':
    main()
