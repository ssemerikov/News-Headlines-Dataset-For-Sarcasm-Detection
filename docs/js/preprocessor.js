/**
 * Text preprocessing for sarcasm detection
 * Mirrors the Python preprocessing logic from the training script
 */

class TextPreprocessor {
    constructor(wordIndex, config) {
        this.wordIndex = wordIndex;
        this.config = config;
    }

    /**
     * Preprocess text input to match training data format
     * @param {string} text - The input text to preprocess
     * @returns {tf.Tensor} - Preprocessed tensor ready for model input
     */
    preprocess(text) {
        // 1. Lowercase
        text = text.toLowerCase();

        // 2. Remove filter characters
        text = this.removeFilters(text);

        // 3. Tokenize (split on whitespace)
        const words = text.trim().split(/\s+/);

        // 4. Convert to indices
        const sequence = words.map(word => {
            return this.wordIndex[word] || this.config.oovIndex;
        });

        // 5. Truncate if needed
        const truncated = sequence.slice(0, this.config.maxLength);

        // 6. Pad to maxLength
        const padded = this.padSequence(truncated, this.config.maxLength);

        // 7. Convert to tensor
        const tensor = tf.tensor2d([padded], [1, this.config.maxLength]);

        return tensor;
    }

    /**
     * Remove filter characters from text
     */
    removeFilters(text) {
        // Match Python's filter string
        const filters = /[!"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~\t\n]/g;
        return text.replace(filters, ' ');
    }

    /**
     * Pad sequence to target length
     */
    padSequence(sequence, maxLength) {
        if (sequence.length >= maxLength) {
            return sequence.slice(0, maxLength);
        }

        // Post-padding (pad at the end)
        const padded = [...sequence];
        while (padded.length < maxLength) {
            padded.push(0);
        }

        return padded;
    }

    /**
     * Get debugging info for a text
     */
    debug(text) {
        const lowercased = text.toLowerCase();
        const filtered = this.removeFilters(lowercased);
        const words = filtered.trim().split(/\s+/);
        const indices = words.map(word => ({
            word,
            index: this.wordIndex[word] || this.config.oovIndex,
            found: this.wordIndex.hasOwnProperty(word)
        }));

        return {
            original: text,
            lowercased,
            filtered,
            words,
            indices,
            oovCount: indices.filter(i => !i.found).length
        };
    }
}
