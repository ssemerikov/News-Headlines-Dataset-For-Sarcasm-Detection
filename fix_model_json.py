"""
Fix model.json files to be compatible with TensorFlow.js
Transforms Keras 3 format to TensorFlow.js-compatible format
"""
import json
import os

def transform_keras3_to_tfjs(config):
    """
    Transform Keras 3 config format to TensorFlow.js format
    """
    if isinstance(config, dict):
        new_config = {}
        for key, value in config.items():
            # Convert snake_case to camelCase for top-level keys
            if '_' in key:
                parts = key.split('_')
                new_key = parts[0] + ''.join(word.capitalize() for word in parts[1:])
            else:
                new_key = key

            # Recursively transform nested configs
            new_config[new_key] = transform_keras3_to_tfjs(value)

        return new_config
    elif isinstance(config, list):
        return [transform_keras3_to_tfjs(item) for item in config]
    else:
        return config

def fix_model_json(model_path):
    """
    Fix a model.json file to be compatible with TensorFlow.js
    """
    json_file = os.path.join(model_path, 'model.json')

    if not os.path.exists(json_file):
        print(f"  ✗ {json_file} not found")
        return False

    print(f"\nFixing: {json_file}")

    # Load existing model.json
    with open(json_file, 'r') as f:
        model_data = json.load(f)

    # The issue is that modelTopology itself needs to be wrapped
    # TensorFlow.js expects the format:
    # {
    #   "className": "Sequential" or "Functional",
    #   "config": {...}
    # }

    topology = model_data.get('modelTopology', {})

    # Create TensorFlow.js compatible format
    tfjs_topology = {
        "class_name": "Functional",  # Keras 3 models are functional
        "config": topology
    }

    # Update model data
    model_data['modelTopology'] = tfjs_topology

    # Save fixed model.json
    with open(json_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"  ✓ Fixed {json_file}")
    return True

def main():
    print("=" * 80)
    print("FIXING MODEL.JSON FILES FOR TENSORFLOW.JS COMPATIBILITY")
    print("=" * 80)

    models = [
        'docs/models/bilstm',
        'docs/models/lstm',
        'docs/models/simplernn'
    ]

    for model_path in models:
        fix_model_json(model_path)

    print("\n" + "=" * 80)
    print("DONE! Now test with: node test_web_models.js")
    print("=" * 80)

if __name__ == '__main__':
    main()
