/**
 * Main Application Logic for Sarcasm Detection
 */

// Global state
const app = {
    models: {},
    wordIndex: null,
    config: null,
    preprocessor: null,
    currentModel: 'bilstm',
    isLoading: true
};

/**
 * Initialize the application
 */
async function init() {
    try {
        updateLoadingStatus('Loading configuration...', 10);

        // Load config
        app.config = await loadJSON('models/config.json');

        updateLoadingStatus('Loading vocabulary...', 20);

        // Load word index
        app.wordIndex = await loadJSON('models/word_index.json');

        updateLoadingStatus('Initializing preprocessor...', 30);

        // Initialize preprocessor
        app.preprocessor = new TextPreprocessor(app.wordIndex, app.config);

        // Load models
        await loadModels();

        updateLoadingStatus('Ready!', 100);

        // Hide loading screen, show app
        setTimeout(() => {
            document.getElementById('loadingScreen').style.display = 'none';
            document.getElementById('mainApp').style.display = 'block';
            document.getElementById('mainApp').classList.add('fade-in');
        }, 500);

        app.isLoading = false;

        // Set up event listeners
        setupEventListeners();

    } catch (error) {
        console.error('Initialization error:', error);
        updateLoadingStatus(`Error: ${error.message}`, 0);
        document.getElementById('loadingScreen').innerHTML += `
            <div class="alert alert-danger mt-4" role="alert">
                <strong>Failed to load the application.</strong><br>
                ${error.message}<br>
                <small>Please refresh the page or check the console for details.</small>
            </div>
        `;
    }
}

/**
 * Load all models
 */
async function loadModels() {
    const modelNames = ['bilstm', 'lstm', 'simplernn'];
    const loadPromises = [];

    for (let i = 0; i < modelNames.length; i++) {
        const modelName = modelNames[i];
        const progress = 30 + (i / modelNames.length) * 60;

        updateLoadingStatus(`Loading ${modelName.toUpperCase()} model...`, progress);

        loadPromises.push(
            tf.loadGraphModel(`models/${modelName}/model.json`)
                .then(model => {
                    app.models[modelName] = model;
                    console.log(`✓ Loaded ${modelName} model`);
                })
                .catch(err => {
                    console.error(`Failed to load ${modelName}:`, err);
                    throw new Error(`Failed to load ${modelName} model`);
                })
        );
    }

    await Promise.all(loadPromises);
}

/**
 * Update loading screen status
 */
function updateLoadingStatus(message, progress) {
    const statusEl = document.getElementById('loadingStatus');
    const progressBar = document.getElementById('loadingProgress');

    if (statusEl) statusEl.textContent = message;
    if (progressBar) progressBar.style.width = `${progress}%`;
}

/**
 * Load JSON file
 */
async function loadJSON(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load ${url}: ${response.statusText}`);
    }
    return await response.json();
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Model selection
    document.querySelectorAll('input[name="modelSelect"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            app.currentModel = e.target.value;
        });
    });

    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', analyzeSingle);

    // Compare button
    document.getElementById('compareBtn').addEventListener('click', compareModels);

    // Enter key in textarea
    document.getElementById('headlineInput').addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeSingle();
        }
    });
}

/**
 * Analyze headline with single model
 */
async function analyzeSingle() {
    const input = document.getElementById('headlineInput').value.trim();

    if (!input) {
        alert('Please enter a headline to analyze.');
        return;
    }

    // Show loading state
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalText = analyzeBtn.innerHTML;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';

    try {
        const result = await predict(input, app.currentModel);
        displaySingleResult(input, result, app.currentModel);
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = originalText;
    }
}

/**
 * Compare all models
 */
async function compareModels() {
    const input = document.getElementById('headlineInput').value.trim();

    if (!input) {
        alert('Please enter a headline to analyze.');
        return;
    }

    // Show loading state
    const compareBtn = document.getElementById('compareBtn');
    const originalText = compareBtn.innerHTML;
    compareBtn.disabled = true;
    compareBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Comparing...';

    try {
        const results = {};
        for (const modelName of ['simplernn', 'lstm', 'bilstm']) {
            results[modelName] = await predict(input, modelName);
        }
        displayComparisonResult(input, results);
    } catch (error) {
        console.error('Comparison error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        compareBtn.disabled = false;
        compareBtn.innerHTML = originalText;
    }
}

/**
 * Make prediction with a model
 */
async function predict(text, modelName) {
    const model = app.models[modelName];
    if (!model) {
        throw new Error(`Model ${modelName} not loaded`);
    }

    // Preprocess
    const inputTensor = app.preprocessor.preprocess(text);

    // Predict
    const outputTensor = model.predict(inputTensor);
    const probability = (await outputTensor.data())[0];

    // Clean up tensors
    inputTensor.dispose();
    outputTensor.dispose();

    const isSarcastic = probability > 0.5;
    const confidence = isSarcastic ? probability : (1 - probability);

    return {
        probability,
        isSarcastic,
        confidence,
        label: isSarcastic ? 'Sarcastic' : 'Non-Sarcastic'
    };
}

/**
 * Display single model result
 */
function displaySingleResult(headline, result, modelName) {
    const resultsSection = document.getElementById('resultsSection');
    const singleResult = document.getElementById('singleResult');
    const comparisonResult = document.getElementById('comparisonResult');
    const resultContent = document.getElementById('resultContent');

    // Show results section
    resultsSection.style.display = 'block';
    singleResult.style.display = 'block';
    comparisonResult.style.display = 'none';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Create result HTML
    const confidencePercent = (result.confidence * 100).toFixed(1);
    const confidenceClass = result.confidence > 0.8 ? 'high' : (result.confidence > 0.6 ? 'medium' : 'low');
    const badgeClass = result.isSarcastic ? 'sarcastic' : 'non-sarcastic';

    resultContent.innerHTML = `
        <div class="mb-4">
            <h6 class="text-muted">Headline:</h6>
            <p class="fs-5">"${headline}"</p>
        </div>

        <div class="mb-4">
            <h6 class="text-muted">Model:</h6>
            <span class="badge bg-primary">${modelName.toUpperCase()}</span>
        </div>

        <div class="mb-4">
            <div class="result-badge ${badgeClass}">
                ${result.isSarcastic ? '😏' : '📰'} ${result.label}
            </div>
        </div>

        <div class="confidence-container">
            <div class="confidence-label">
                <span>Confidence</span>
                <span><strong>${confidencePercent}%</strong></span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill ${confidenceClass}" style="width: ${confidencePercent}%">
                    ${confidencePercent}%
                </div>
            </div>
        </div>

        <div class="mt-4">
            <small class="text-muted">
                <strong>Raw probability:</strong> ${(result.probability * 100).toFixed(2)}% sarcastic
            </small>
        </div>
    `;

    resultContent.classList.add('fade-in');
}

/**
 * Display model comparison result
 */
function displayComparisonResult(headline, results) {
    const resultsSection = document.getElementById('resultsSection');
    const singleResult = document.getElementById('singleResult');
    const comparisonResult = document.getElementById('comparisonResult');
    const comparisonContent = document.getElementById('comparisonContent');

    // Show results section
    resultsSection.style.display = 'block';
    singleResult.style.display = 'none';
    comparisonResult.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Create comparison table
    const tableRows = Object.entries(results).map(([modelName, result]) => {
        const confidencePercent = (result.confidence * 100).toFixed(1);
        const predictionClass = result.isSarcastic ? 'sarcastic' : 'non-sarcastic';

        return `
            <tr>
                <td class="model-name">${modelName.toUpperCase()}</td>
                <td class="prediction-cell ${predictionClass}">
                    ${result.isSarcastic ? '😏' : '📰'} ${result.label}
                </td>
                <td>${confidencePercent}%</td>
                <td>${(result.probability * 100).toFixed(2)}%</td>
            </tr>
        `;
    }).join('');

    comparisonContent.innerHTML = `
        <div class="mb-4">
            <h6 class="text-muted">Headline:</h6>
            <p class="fs-5">"${headline}"</p>
        </div>

        <div class="comparison-table">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Probability</th>
                    </tr>
                </thead>
                <tbody>
                    ${tableRows}
                </tbody>
            </table>
        </div>

        <div class="alert alert-info">
            <small>
                <strong>Note:</strong> Confidence represents how sure the model is about its prediction.
                Probability shows the raw likelihood of the headline being sarcastic.
            </small>
        </div>
    `;

    comparisonContent.classList.add('fade-in');
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
