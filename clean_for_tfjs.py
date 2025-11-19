"""
Clean model.json to pure TensorFlow.js format by removing Keras 3 metadata
"""
import json
import os

def clean_layer_config(config):
    """
    Clean a layer config to remove Keras 3-specific fields that TensorFlow.js doesn't understand
    """
    cleaned = {}

    # Fields that TensorFlow.js understands
    tfjs_fields = {
        'name', 'trainable', 'batchInputShape', 'dtype', 'sparse', 'ragged',
        'inputDim', 'outputDim', 'embeddingsInitializer', 'inputLength',
        'units', 'activation', 'useBias', 'kernelInitializer', 'biasInitializer',
        'returnSequences', 'returnState', 'goBackwards', 'stateful', 'unroll',
        'recurrentActivation', 'recurrentInitializer', 'unitForgetBias',
        'dropout', 'recurrentDropout', 'mergeMode', 'layer', 'backwardLayer',
        'rate', 'noiseShape', 'seed', 'zeroOutputForMask', 'maskZero'
    }

    for key, value in config.items():
        # Skip Keras 3 metadata fields
        if key in ['module', 'registeredName', 'buildConfig']:
            continue

        # Handle dtype - if it's an object, extract just the name
        if key == 'dtype' and isinstance(value, dict):
            cleaned[key] = value.get('config', {}).get('name', 'float32')
        # Handle initializers - simplified format
        elif key.endswith('Initializer') and isinstance(value, dict):
            if value is None:
                continue
            cleaned[key] = {
                'className': value.get('className', value.get('class_name', 'GlorotUniform')),
                'config': value.get('config', {})
            }
        # Handle nested layer configs (for Bidirectional)
        elif key in ['layer', 'backwardLayer'] and isinstance(value, dict):
            layer_config = value.get('config', {})
            cleaned[key] = {
                'className': value.get('className', value.get('class_name', '')),
                'config': clean_layer_config(layer_config)
            }
        # Copy other recognized fields
        elif key in tfjs_fields:
            cleaned[key] = value

    return cleaned

def clean_layer(layer):
    """
    Clean a complete layer definition
    """
    return {
        'className': layer.get('className', layer.get('class_name', '')),
        'config': clean_layer_config(layer.get('config', {}))
    }

def clean_model(model_path):
    """
    Clean a model.json file
    """
    json_file = os.path.join(model_path, 'model.json')

    if not os.path.exists(json_file):
        print(f"  ✗ {json_file} not found")
        return False

    print(f"\nCleaning: {json_file}")

    # Load model.json
    with open(json_file, 'r') as f:
        model_data = json.load(f)

    # Clean the model topology
    topology = model_data.get('modelTopology', {})
    config = topology.get('config', {})
    layers = config.get('layers', [])

    # Clean each layer
    cleaned_layers = [clean_layer(layer) for layer in layers]

    # Create clean topology
    clean_topology = {
        'className': 'Sequential',
        'config': {
            'name': config.get('name', 'model'),
            'layers': cleaned_layers
        }
    }

    model_data['modelTopology'] = clean_topology

    # Save
    with open(json_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"  ✓ Cleaned!")
    return True

def main():
    print("=" * 80)
    print("CLEANING MODELS FOR TENSORFLOW.JS")
    print("=" * 80)

    models = [
        'docs/models/bilstm',
        'docs/models/lstm',
        'docs/models/simplernn'
    ]

    for model_path in models:
        clean_model(model_path)

    print("\n" + "=" * 80)
    print("DONE! Test with: node test_web_models.js")
    print("=" * 80)

if __name__ == '__main__':
    main()
