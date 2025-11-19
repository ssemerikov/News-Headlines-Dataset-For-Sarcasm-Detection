"""
Use automatic TensorFlow.js layer naming instead of Keras names
"""
import json
import os

def update_model(model_path):
    """
    Update model to use TensorFlow.js auto-generated layer names
    """
    json_file = os.path.join(model_path, 'model.json')

    if not os.path.exists(json_file):
        print(f"  ✗ {json_file} not found")
        return False

    print(f"\nUpdating: {json_file}")

    # Load model.json
    with open(json_file, 'r') as f:
        model_data = json.load(f)

    # Get layers
    topology = model_data.get('modelTopology', {})
    config = topology.get('config', {})
    layers = config.get('layers', [])

    # Map old names to new TensorFlow.js auto-generated names
    # TensorFlow.js names layers as: className_layerIndex
    # e.g., embedding_1, bidirectional_1, dropout_1, dense_1, dropout_2, dense_2

    name_mapping = {}
    layer_counts = {}

    for i, layer in enumerate(layers):
        class_name = layer.get('className', '').lower()

        # Count layers of each type to generate sequential names
        if class_name not in layer_counts:
            layer_counts[class_name] = 1
        else:
            layer_counts[class_name] += 1

        old_name = layer['config'].get('name', '')

        # TensorFlow.js auto-naming: classname_index (starting from 1)
        new_name = f"{class_name}_{layer_counts[class_name]}"

        name_mapping[old_name] = new_name

        # Update layer name
        layer['config']['name'] = new_name

        print(f"    {old_name} -> {new_name}")

    # Update weight names in manifest
    if 'weightsManifest' in model_data:
        for manifest_entry in model_data['weightsManifest']:
            if 'weights' in manifest_entry:
                for weight in manifest_entry['weights']:
                    old_weight_name = weight['name']

                    # Extract layer name from weight (format: layername/weighttype:0)
                    if '/' in old_weight_name:
                        parts = old_weight_name.split('/')
                        old_layer_name = parts[0]
                        weight_type = '/'.join(parts[1:])

                        if old_layer_name in name_mapping:
                            new_weight_name = f"{name_mapping[old_layer_name]}/{weight_type}"
                            weight['name'] = new_weight_name
                            print(f"    Weight: {old_weight_name} -> {new_weight_name}")

    # Save
    with open(json_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"  ✓ Updated!")
    return True

def main():
    print("=" * 80)
    print("USING TENSORFLOW.JS AUTO-GENERATED LAYER NAMES")
    print("=" * 80)

    models = [
        'docs/models/bilstm',
        'docs/models/lstm',
        'docs/models/simplernn'
    ]

    for model_path in models:
        update_model(model_path)

    print("\n" + "=" * 80)
    print("DONE! Test with: node test_web_models.js")
    print("=" * 80)

if __name__ == '__main__':
    main()
