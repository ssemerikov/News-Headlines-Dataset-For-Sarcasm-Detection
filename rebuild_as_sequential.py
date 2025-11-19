"""
Rebuild models as Sequential format for TensorFlow.js compatibility
"""
import json
import os

def rebuild_as_sequential(model_path):
    """
    Rebuild a Functional model as Sequential for TensorFlow.js
    """
    json_file = os.path.join(model_path, 'model.json')

    if not os.path.exists(json_file):
        print(f"  ✗ {json_file} not found")
        return False

    print(f"\nRebuilding: {json_file}")

    # Load model.json
    with open(json_file, 'r') as f:
        model_data = json.load(f)

    # Get the layers from the functional model
    topology = model_data['modelTopology']
    if 'config' in topology:
        config = topology['config']
    else:
        config = topology

    layers = config.get('layers', [])

    # Remove InputLayer (Sequential doesn't need it explicitly)
    filtered_layers = [layer for layer in layers if layer.get('className') != 'InputLayer']

    # For the first layer (Embedding), we need to add inputShape
    if filtered_layers:
        first_layer = filtered_layers[0]
        if first_layer.get('className') == 'Embedding':
            # Get the input shape from the original InputLayer
            input_layer = next((l for l in layers if l.get('className') == 'InputLayer'), None)
            if input_layer:
                batch_shape = input_layer['config'].get('batchInputShape', [None, 100])
                # Add inputLength to Embedding config
                first_layer['config']['inputLength'] = batch_shape[1] if len(batch_shape) > 1 else 100

    # Create Sequential model topology
    sequential_topology = {
        "className": "Sequential",
        "config": {
            "name": config.get('name', 'sequential_model'),
            "layers": filtered_layers
        }
    }

    # Update model data
    model_data['modelTopology'] = sequential_topology

    # Save
    with open(json_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"  ✓ Rebuilt as Sequential!")
    return True

def main():
    print("=" * 80)
    print("REBUILDING MODELS AS SEQUENTIAL FOR TENSORFLOW.JS")
    print("=" * 80)

    models = [
        'docs/models/bilstm',
        'docs/models/lstm',
        'docs/models/simplernn'
    ]

    for model_path in models:
        rebuild_as_sequential(model_path)

    print("\n" + "=" * 80)
    print("DONE! Test with: node test_web_models.js")
    print("=" * 80)

if __name__ == '__main__':
    main()
