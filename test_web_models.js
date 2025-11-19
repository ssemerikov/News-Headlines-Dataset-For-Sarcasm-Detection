/**
 * Test script to verify TensorFlow.js models load and work correctly
 * Run with: node test_web_models.js
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Load configuration and word index
const config = JSON.parse(fs.readFileSync('docs/models/config.json', 'utf8'));
const wordIndex = JSON.parse(fs.readFileSync('docs/models/word_index.json', 'utf8'));

console.log('='.repeat(80));
console.log('TESTING TENSORFLOW.JS MODELS');
console.log('='.repeat(80));
console.log(`\nConfig: vocab=${config.vocabSize}, maxLength=${config.maxLength}`);
console.log(`Word index: ${Object.keys(wordIndex).length} words\n`);

// Simple text preprocessor
function preprocessText(text) {
    // Lowercase
    text = text.toLowerCase();

    // Remove filters
    const filters = /[!"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~\t\n]/g;
    text = text.replace(filters, ' ');

    // Tokenize
    const words = text.trim().split(/\s+/);

    // Convert to indices
    const sequence = words.map(word => wordIndex[word] || config.oovIndex);

    // Truncate/pad
    const padded = sequence.slice(0, config.maxLength);
    while (padded.length < config.maxLength) {
        padded.push(0);
    }

    return tf.tensor2d([padded], [1, config.maxLength]);
}

async function testModel(modelName, modelPath) {
    console.log(`\nTesting ${modelName.toUpperCase()} model...`);
    console.log('-'.repeat(60));

    try {
        // Load model
        const model = await tf.loadLayersModel(`file://${modelPath}`);
        console.log(`  ✓ Model loaded successfully`);
        console.log(`    Input shape: ${model.inputs[0].shape}`);
        console.log(`    Output shape: ${model.outputs[0].shape}`);

        // Test prediction with example headlines
        const testHeadlines = [
            "Area Man Knows All The Shortcut Keys",  // Sarcastic
            "Scientists Discover Water On Mars"      // Non-sarcastic
        ];

        for (const headline of testHeadlines) {
            const inputTensor = preprocessText(headline);
            const output = model.predict(inputTensor);
            const probability = (await output.data())[0];
            const isSarcastic = probability > 0.5;
            const label = isSarcastic ? 'Sarcastic' : 'Non-Sarcastic';

            console.log(`\n  Headline: "${headline}"`);
            console.log(`    Prediction: ${label}`);
            console.log(`    Probability: ${(probability * 100).toFixed(2)}%`);
            console.log(`    Confidence: ${((isSarcastic ? probability : 1-probability) * 100).toFixed(2)}%`);

            inputTensor.dispose();
            output.dispose();
        }

        console.log(`\n  ✓ ${modelName.toUpperCase()} test passed!`);
        return true;

    } catch (error) {
        console.error(`  ✗ Error testing ${modelName}:`, error.message);
        return false;
    }
}

async function main() {
    const models = [
        ['bilstm', path.resolve(__dirname, 'docs/models/bilstm/model.json')],
        ['lstm', path.resolve(__dirname, 'docs/models/lstm/model.json')],
        ['simplernn', path.resolve(__dirname, 'docs/models/simplernn/model.json')]
    ];

    let allPassed = true;

    for (const [name, modelPath] of models) {
        const passed = await testModel(name, modelPath);
        if (!passed) allPassed = false;
    }

    console.log('\n' + '='.repeat(80));
    if (allPassed) {
        console.log('✅ ALL TESTS PASSED!');
        console.log('\nModels are ready for deployment.');
        console.log('You can now:');
        console.log('  1. Open http://localhost:8888 in your browser');
        console.log('  2. Test the web interface');
        console.log('  3. Push to GitHub and deploy to GitHub Pages');
    } else {
        console.log('❌ SOME TESTS FAILED');
        console.log('\nPlease review the errors above.');
    }
    console.log('='.repeat(80));
}

main().catch(console.error);
