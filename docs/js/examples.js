/**
 * Example headlines from TheOnion (sarcastic) and HuffPost (non-sarcastic)
 */

const EXAMPLE_HEADLINES = [
    {
        text: "Area Man Knows All The Shortcut Keys",
        label: "Sarcastic",
        source: "TheOnion"
    },
    {
        text: "Local Idiot To Post Comment On Internet",
        label: "Sarcastic",
        source: "TheOnion"
    },
    {
        text: "Mother Comes Pretty Close To Using Word 'Streaming' Correctly",
        label: "Sarcastic",
        source: "TheOnion"
    },
    {
        text: "Nation Demands New Season Of 'Black Mirror' Right Fucking Now",
        label: "Sarcastic",
        source: "TheOnion"
    },
    {
        text: "Scientists Discover Water On Mars",
        label: "Non-Sarcastic",
        source: "HuffPost"
    },
    {
        text: "Trump Announces New Immigration Policy At Border",
        label: "Non-Sarcastic",
        source: "HuffPost"
    },
    {
        text: "Stock Market Reaches Record High",
        label: "Non-Sarcastic",
        source: "HuffPost"
    },
    {
        text: "Study Finds Exercise Benefits Mental Health",
        label: "Non-Sarcastic",
        source: "HuffPost"
    }
];

/**
 * Render example headlines to the page
 */
function renderExamples() {
    const container = document.getElementById('examplesContainer');
    if (!container) return;

    container.innerHTML = EXAMPLE_HEADLINES.map((example, index) => {
        const labelClass = example.label === 'Sarcastic' ? 'sarcastic' : 'non-sarcastic';
        const badgeClass = example.label === 'Sarcastic' ? 'bg-danger' : 'bg-success';

        return `
            <div class="col-md-6">
                <div class="example-headline ${labelClass}" onclick="loadExample(${index})">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <span class="badge ${badgeClass}">${example.label}</span>
                        <small class="text-muted">${example.source}</small>
                    </div>
                    <div>${example.text}</div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Load an example into the input field
 */
function loadExample(index) {
    const example = EXAMPLE_HEADLINES[index];
    const input = document.getElementById('headlineInput');

    if (input) {
        input.value = example.text;
        input.focus();

        // Scroll to input
        input.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Initialize examples when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderExamples);
} else {
    renderExamples();
}
