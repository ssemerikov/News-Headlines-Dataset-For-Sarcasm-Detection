/**
 * Rebuild models in TensorFlow.js and manually load weights
 */
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

async function rebuildAndSaveModel(modelName, createModelFn, modelPath) {
    console.log(`\nRebuilding ${modelName}...`);

    try {
        // Create the model using TensorFlow.js
        const model = createModelFn();

        console.log(`  Model architecture created`);
        console.log(`  Layers: ${model.layers.map(l => l.name).join(', ')}`);

        // Load the old model.json to get weights data
        const oldModelPath = path.join(modelPath, 'model.json');
        const oldModelData = JSON.parse(fs.readFileSync(oldModelPath, 'utf8'));

        // We'll need to manually set weights
        // For now, just save the new architecture
        const savePath = `file://${path.join(modelPath, '../', modelName + '_rebuilt')}`;
        await model.save(savePath);

        console.log(`  ✓ Saved architecture to ${savePath}`);

        return true;
    } catch (error) {
        console.error(`  ✗ Error: ${error.message}`);
        return false;
    }
}

function createBiLSTMModel() {
    const model = tf.sequential();

    model.add(tf.layers.embedding({
        inputDim: 10000,
        outputDim: 128,
        inputLength: 100,
        name: 'embedding_1'
    }));

    model.add(tf.layers.bidirectional({
        layer: tf.layers.lstm({units: 64}),
        name: 'bidirectional_1'
    }));

    model.add(tf.layers.dropout({rate: 0.5, name: 'dropout_1'}));
    model.add(tf.layers.dense({units: 64, activation: 'relu', name: 'dense_1'}));
    model.add(tf.layers.dropout({rate: 0.5, name: 'dropout_2'}));
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid', name: 'dense_2'}));

    return model;
}

function createLSTMModel() {
    const model = tf.sequential();

    model.add(tf.layers.embedding({
        inputDim: 10000,
        outputDim: 128,
        inputLength: 100,
        name: 'embedding_1'
    }));

    model.add(tf.layers.lstm({units: 64, name: 'lstm_1'}));
    model.add(tf.layers.dropout({rate: 0.5, name: 'dropout_1'}));
    model.add(tf.layers.dense({units: 64, activation: 'relu', name: 'dense_1'}));
    model.add(tf.layers.dropout({rate: 0.5, name: 'dropout_2'}));
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid', name: 'dense_2'}));

    return model;
}

function createSimpleRNNModel() {
    const model = tf.sequential();

    model.add(tf.layers.embedding({
        inputDim: 10000,
        outputDim: 128,
        inputLength: 100,
        name: 'embedding_1'
    }));

    model.add(tf.layers.simpleRNN({units: 64, name: 'simplernn_1'}));
    model.add(tf.layers.dropout({rate: 0.5, name: 'dropout_1'}));
    model.add(tf.layers.dense({units: 64, activation: 'relu', name: 'dense_1'}));
    model.add(tf.layers.dropout({rate: 0.5, name: 'dropout_2'}));
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid', name: 'dense_2'}));

    return model;
}

async function main() {
    console.log('='.repeat(80));
    console.log('REBUILDING MODELS IN TENSORFLOW.JS');
    console.log('='.repeat(80));

    const models = [
        ['bilstm', createBiLSTMModel, 'docs/models/bilstm'],
        ['lstm', createLSTMModel, 'docs/models/lstm'],
        ['simplernn', createSimpleRNNModel, 'docs/models/simplernn']
    ];

    for (const [name, createFn, modelPath] of models) {
        await rebuildAndSaveModel(name, createFn, modelPath);
    }

    console.log('\n' + '='.repeat(80));
    console.log('Models rebuilt! Now we need to copy weights...');
    console.log('='.repeat(80));
}

main().catch(console.error);
