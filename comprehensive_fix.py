"""
Comprehensive fix to convert Keras 3 model.json to TensorFlow.js format
"""
import json
import os

def snake_to_camel(snake_str):
    """Convert snake_case to camelCase"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def transform_config(config, is_top_level=False):
    """
    Recursively transform Keras 3 config to TensorFlow.js format
    """
    if isinstance(config, dict):
        new_config = {}
        for key, value in config.items():
            # Special handling for specific keys
            if key == 'batch_shape':
                new_config['batchInputShape'] = value
            elif key == 'class_name':
                new_config['className'] = value
            elif key == 'registered_name':
                new_config['registeredName'] = value
            elif key == 'build_config':
                new_config['buildConfig'] = transform_config(value)
            elif '_' in key:
                # Convert snake_case to camelCase
                new_config[snake_to_camel(key)] = transform_config(value)
            else:
                new_config[key] = transform_config(value)

        return new_config
    elif isinstance(config, list):
        return [transform_config(item) for item in config]
    else:
        return config

def fix_model(model_path):
    """
    Fix a model.json file
    """
    json_file = os.path.join(model_path, 'model.json')

    if not os.path.exists(json_file):
        print(f"  ✗ {json_file} not found")
        return False

    print(f"\nFixing: {json_file}")

    # Load model.json
    with open(json_file, 'r') as f:
        model_data = json.load(f)

    # Transform the modelTopology
    if 'modelTopology' in model_data:
        topology = model_data['modelTopology']

        # If it's already wrapped in class_name/config, unwrap it
        if 'class_name' in topology and 'config' in topology:
            topology = topology['config']

        # Transform all keys
        transformed_topology = transform_config(topology)

        # Wrap in the format TensorFlow.js expects
        model_data['modelTopology'] = {
            "className": "Functional",
            "config": transformed_topology
        }

    # Save
    with open(json_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"  ✓ Fixed!")
    return True

def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL.JSON FIX FOR TENSORFLOW.JS")
    print("=" * 80)

    models = [
        'docs/models/bilstm',
        'docs/models/lstm',
        'docs/models/simplernn'
    ]

    for model_path in models:
        fix_model(model_path)

    print("\n" + "=" * 80)
    print("DONE! Test with: node test_web_models.js")
    print("=" * 80)

if __name__ == '__main__':
    main()
