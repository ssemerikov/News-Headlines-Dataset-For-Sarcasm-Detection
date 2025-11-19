"""
Fix weight names to match Sequential model layer names
"""
import json
import os

def fix_weights(model_path):
    """
    Fix weight names in model.json to match layer names
    """
    json_file = os.path.join(model_path, 'model.json')

    if not os.path.exists(json_file):
        print(f"  ✗ {json_file} not found")
        return False

    print(f"\nFixing weights: {json_file}")

    # Load model.json
    with open(json_file, 'r') as f:
        model_data = json.load(f)

    # Get layer names from topology
    topology = model_data.get('modelTopology', {})
    config = topology.get('config', {})
    layers = config.get('layers', [])

    # Create mapping of old weight names to new layer-based names
    layer_index = 0
    weight_mapping = {}

    for layer in layers:
        layer_name = layer.get('config', {}).get('name', '')
        class_name = layer.get('className', '')

        # For each layer, map its old weight prefix to the layer name
        # Sequential models use layer names directly
        if class_name == 'Embedding':
            # Embedding has 'embeddings' or 'kernel' weight
            old_prefix = layer_name + '/'
            weight_mapping[old_prefix] = layer_name + '/'
        elif class_name in ['LSTM', 'SimpleRNN', 'Bidirectional']:
            old_prefix = layer_name + '/'
            weight_mapping[old_prefix] = layer_name + '/'
        elif class_name == 'Dense':
            old_prefix = layer_name + '/'
            weight_mapping[old_prefix] = layer_name + '/'

    # Update weight names in manifest
    if 'weightsManifest' in model_data:
        for manifest_entry in model_data['weightsManifest']:
            if 'weights' in manifest_entry:
                for weight in manifest_entry['weights']:
                    old_name = weight['name']

                    # The weight names are already using layer names, so they should work
                    # The issue might be that Sequential expects slightly different naming

                    # For Sequential models, TensorFlow.js expects:
                    # layerName/kernel:0, layerName/bias:0, etc.
                    # Our weights already have this format, so they should work

                    # Let me check if the issue is the :0 suffix or embedding naming
                    print(f"    Weight: {old_name}")

    # Save
    with open(json_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"  ✓ Checked weights")
    return True

def main():
    print("=" * 80)
    print("CHECKING WEIGHT NAMES")
    print("=" * 80)

    models = [
        'docs/models/bilstm'
    ]

    for model_path in models:
        fix_weights(model_path)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
