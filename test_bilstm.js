const tf = require('@tensorflow/tfjs-node');
const path = require('path');

async function test() {
    try {
        const modelPath = path.resolve(__dirname, 'docs/models/bilstm/model.json');
        console.log('Loading from:', modelPath);

        const model = await tf.loadLayersModel('file://' + modelPath);
        console.log('Model loaded successfully!');

        // Print layer info
        model.layers.forEach((layer, i) => {
            console.log(`Layer ${i}: ${layer.name}`);
        });

    } catch (error) {
        console.error('Error:', error.message);
        // Print full stack for debugging
        if (error.stack) {
            console.error(error.stack.split('\n').slice(0, 10).join('\n'));
        }
    }
}

test();
