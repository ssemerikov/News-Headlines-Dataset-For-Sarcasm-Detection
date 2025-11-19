/**
 * Convert SavedModel to TensorFlow.js format using Node.js
 */
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

async function convertModel(savedModelPath, outputPath) {
    console.log(`\nConverting: ${savedModelPath} -> ${outputPath}`);

    try {
        // Load the SavedModel
        console.log('  Loading SavedModel...');
        const model = await tf.node.loadSavedModel(savedModelPath);

        // Save as TensorFlow.js LayersModel
        console.log('  Saving as TensorFlow.js LayersModel...');
        await model.save(`file://${outputPath}`);

        console.log(`  ✓ Successfully converted to ${outputPath}`);
        return true;

    } catch (error) {
        console.error(`  ✗ Error: ${error.message}`);
        return false;
    }
}

async function main() {
    console.log('=' .repeat(80));
    console.log('CONVERTING MODELS TO TENSORFLOW.JS FORMAT');
    console.log('='.repeat(80));

    const models = [
        ['docs/models/bilstm_temp_sm', 'docs/models/bilstm'],
        ['docs/models/lstm_temp_sm', 'docs/models/lstm'],
        ['docs/models/simplernn_temp_sm', 'docs/models/simplernn']
    ];

    let allSuccess = true;

    for (const [inputPath, outputPath] of models) {
        const fullInputPath = path.resolve(__dirname, inputPath);
        const fullOutputPath = path.resolve(__dirname, outputPath);

        // Create output directory if it doesn't exist
        if (!fs.existsSync(fullOutputPath)) {
            fs.mkdirSync(fullOutputPath, { recursive: true });
        }

        const success = await convertModel(fullInputPath, fullOutputPath);
        if (!success) allSuccess = false;
    }

    console.log('\n' + '='.repeat(80));
    if (allSuccess) {
        console.log('✅ ALL MODELS CONVERTED SUCCESSFULLY!');
        console.log('\nNext steps:');
        console.log('  1. Run: node test_web_models.js');
        console.log('  2. Open: http://localhost:8888');
        console.log('  3. Test the web interface');
    } else {
        console.log('❌ SOME CONVERSIONS FAILED');
    }
    console.log('='.repeat(80));
}

main().catch(console.error);
