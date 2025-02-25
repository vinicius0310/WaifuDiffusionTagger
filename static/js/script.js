document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');
            
            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show active tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabName}-tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
    
    // Single Image Tab Functionality
    const imagePreview = document.getElementById('image-preview');
    const imageUpload = document.getElementById('image-upload');
    const previewImage = document.getElementById('preview-image');
    const uploadPlaceholder = document.querySelector('.upload-placeholder');
    
    // Image upload via click
    imagePreview.addEventListener('click', () => {
        imageUpload.click();
    });
    
    // Image upload via drag and drop
    imagePreview.addEventListener('dragover', (e) => {
        e.preventDefault();
        imagePreview.style.borderColor = 'var(--primary-color)';
        imagePreview.style.backgroundColor = 'rgba(106, 90, 205, 0.1)';
    });
    
    imagePreview.addEventListener('dragleave', () => {
        imagePreview.style.borderColor = 'var(--border-color)';
        imagePreview.style.backgroundColor = 'var(--card-bg)';
    });
    
    imagePreview.addEventListener('drop', (e) => {
        e.preventDefault();
        imagePreview.style.borderColor = 'var(--border-color)';
        imagePreview.style.backgroundColor = 'var(--card-bg)';
        
        if (e.dataTransfer.files.length) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });
    
    // Handle file selection
    imageUpload.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleImageFile(e.target.files[0]);
        }
    });
    
    function handleImageFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
    
    // Slider value display
    const generalThreshold = document.getElementById('general-threshold');
    const generalThresholdValue = document.getElementById('general-threshold-value');
    const characterThreshold = document.getElementById('character-threshold');
    const characterThresholdValue = document.getElementById('character-threshold-value');
    
    generalThreshold.addEventListener('input', () => {
        generalThresholdValue.textContent = generalThreshold.value;
    });
    
    characterThreshold.addEventListener('input', () => {
        characterThresholdValue.textContent = characterThreshold.value;
    });
    
    // Batch tab sliders
    const batchGeneralThreshold = document.getElementById('batch-general-threshold');
    const batchGeneralThresholdValue = document.getElementById('batch-general-threshold-value');
    const batchCharacterThreshold = document.getElementById('batch-character-threshold');
    const batchCharacterThresholdValue = document.getElementById('batch-character-threshold-value');
    
    batchGeneralThreshold.addEventListener('input', () => {
        batchGeneralThresholdValue.textContent = batchGeneralThreshold.value;
    });
    
    batchCharacterThreshold.addEventListener('input', () => {
        batchCharacterThresholdValue.textContent = batchCharacterThreshold.value;
    });
    
    // Clear button functionality
    const clearBtn = document.getElementById('clear-btn');
    clearBtn.addEventListener('click', () => {
        // Reset image
        previewImage.src = '';
        previewImage.style.display = 'none';
        uploadPlaceholder.style.display = 'flex';
        
        // Reset form values
        document.getElementById('model-select').selectedIndex = 0;
        generalThreshold.value = 0.35;
        generalThresholdValue.textContent = '0.35';
        document.getElementById('general-mcut').checked = false;
        characterThreshold.value = 0.85;
        characterThresholdValue.textContent = '0.85';
        document.getElementById('character-mcut').checked = false;
        document.getElementById('weighted-captions').checked = false;
        
        // Clear results
        clearResults();
    });
    
    function clearResults() {
        document.getElementById('tags-output').innerHTML = '<div class="placeholder-text">Tags will appear here after analysis</div>';
        document.getElementById('rating-output').innerHTML = '<div class="placeholder-text">Rating will appear here after analysis</div>';
        document.getElementById('character-output').innerHTML = '<div class="placeholder-text">Character tags will appear here after analysis</div>';
        document.getElementById('general-output').innerHTML = '<div class="placeholder-text">General tags will appear here after analysis</div>';
    }
    
    // Batch clear button
    const batchClearBtn = document.getElementById('batch-clear-btn');
    batchClearBtn.addEventListener('click', () => {
        document.getElementById('directory-input').value = '';
        document.getElementById('batch-model-select').selectedIndex = 0;
        batchGeneralThreshold.value = 0.35;
        batchGeneralThresholdValue.textContent = '0.35';
        document.getElementById('batch-general-mcut').checked = false;
        batchCharacterThreshold.value = 0.85;
        batchCharacterThresholdValue.textContent = '0.85';
        document.getElementById('batch-character-mcut').checked = false;
        document.getElementById('batch-append').checked = false;
        document.getElementById('batch-exclude-character').checked = false;
        document.getElementById('batch-weighted-captions').checked = false;
        
        document.getElementById('batch-output').innerHTML = '<div class="placeholder-text">Results will appear here after processing</div>';
    });
    
    // Copy tags button
    const copyTagsBtn = document.getElementById('copy-tags-btn');
    copyTagsBtn.addEventListener('click', () => {
        const tagsOutput = document.getElementById('tags-output');
        if (tagsOutput.textContent && !tagsOutput.querySelector('.placeholder-text')) {
            navigator.clipboard.writeText(tagsOutput.textContent)
                .then(() => {
                    // Show copy success feedback
                    const originalIcon = copyTagsBtn.innerHTML;
                    copyTagsBtn.innerHTML = '<i class="fas fa-check"></i>';
                    setTimeout(() => {
                        copyTagsBtn.innerHTML = originalIcon;
                    }, 2000);
                })
                .catch(err => {
                    console.error('Failed to copy text: ', err);
                });
        }
    });
    
    // Submit button functionality
    const submitBtn = document.getElementById('submit-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    submitBtn.addEventListener('click', () => {
        if (!previewImage.src || previewImage.style.display === 'none') {
            alert('Please select an image first');
            return;
        }
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        // Create form data
        const formData = new FormData();
        
        // Get the file from the input
        const fileInput = document.getElementById('image-upload');
        if (fileInput.files.length > 0) {
            formData.append('image', fileInput.files[0]);
            
            // Add other parameters
            formData.append('model', document.getElementById('model-select').value);
            formData.append('general_thresh', document.getElementById('general-threshold').value);
            formData.append('general_mcut_enabled', document.getElementById('general-mcut').checked);
            formData.append('character_thresh', document.getElementById('character-threshold').value);
            formData.append('character_mcut_enabled', document.getElementById('character-mcut').checked);
            formData.append('weighted_captions', document.getElementById('weighted-captions').checked);
            
            // Send request to server
            fetch('/api/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                displayResults(data);
                loadingOverlay.style.display = 'none';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during processing');
                loadingOverlay.style.display = 'none';
            });
        } else {
            alert('Please select an image first');
            loadingOverlay.style.display = 'none';
        }
    });
    
    function displayResults(data) {
        // Display tags
        const tagsOutput = document.getElementById('tags-output');
        tagsOutput.innerHTML = '';
        tagsOutput.textContent = data.tags;
        
        // Display rating
        const ratingOutput = document.getElementById('rating-output');
        ratingOutput.innerHTML = '';
        
        if (data.rating) {
            const ratingList = document.createElement('ul');
            ratingList.className = 'rating-list';
            
            Object.entries(data.rating)
                .sort((a, b) => b[1] - a[1])
                .forEach(([key, value]) => {
                    const listItem = document.createElement('li');
                    listItem.textContent = `${key}: ${value.toFixed(4)}`;
                    ratingList.appendChild(listItem);
                });
            
            ratingOutput.appendChild(ratingList);
        } else {
            ratingOutput.innerHTML = '<div class="placeholder-text">No rating data available</div>';
        }
        
        // Display character tags
        const characterOutput = document.getElementById('character-output');
        characterOutput.innerHTML = '';
        
        if (data.character_tags && Object.keys(data.character_tags).length > 0) {
            Object.entries(data.character_tags)
                .sort((a, b) => b[1] - a[1])
                .forEach(([key, value]) => {
                    const tag = document.createElement('span');
                    tag.className = 'tag character-tag';
                    tag.textContent = `${key} (${value.toFixed(2)})`;
                    characterOutput.appendChild(tag);
                });
        } else {
            characterOutput.innerHTML = '<div class="placeholder-text">No character tags detected</div>';
        }
        
        // Display general tags
        const generalOutput = document.getElementById('general-output');
        generalOutput.innerHTML = '';
        
        if (data.general_tags && Object.keys(data.general_tags).length > 0) {
            Object.entries(data.general_tags)
                .sort((a, b) => b[1] - a[1])
                .forEach(([key, value]) => {
                    const tag = document.createElement('span');
                    tag.className = 'tag';
                    tag.textContent = `${key} (${value.toFixed(2)})`;
                    generalOutput.appendChild(tag);
                });
        } else {
            generalOutput.innerHTML = '<div class="placeholder-text">No general tags detected</div>';
        }
    }
    
    // Batch processing
    const batchSubmitBtn = document.getElementById('batch-submit-btn');
    
    batchSubmitBtn.addEventListener('click', () => {
        const directoryInput = document.getElementById('directory-input');
        
        if (!directoryInput.value) {
            alert('Please enter a directory path');
            return;
        }
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        // Create form data
        const formData = new FormData();
        formData.append('directory', directoryInput.value);
        formData.append('model', document.getElementById('batch-model-select').value);
        formData.append('general_thresh', document.getElementById('batch-general-threshold').value);
        formData.append('general_mcut_enabled', document.getElementById('batch-general-mcut').checked);
        formData.append('character_thresh', document.getElementById('batch-character-threshold').value);
        formData.append('character_mcut_enabled', document.getElementById('batch-character-mcut').checked);
        formData.append('append_to_existing', document.getElementById('batch-append').checked);
        formData.append('exclude_character_tags', document.getElementById('batch-exclude-character').checked);
        formData.append('weighted_captions', document.getElementById('batch-weighted-captions').checked);
        
        // Send request to server
        fetch('/api/batch', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayBatchResults(data);
            loadingOverlay.style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during batch processing');
            loadingOverlay.style.display = 'none';
        });
    });
    
    function displayBatchResults(data) {
        const batchOutput = document.getElementById('batch-output');
        batchOutput.innerHTML = '';
        
        if (data.error) {
            batchOutput.innerHTML = `<div class="error-message">${data.error}</div>`;
            return;
        }
        
        if (Object.keys(data).length === 0) {
            batchOutput.innerHTML = '<div class="placeholder-text">No images found in the specified directory</div>';
            return;
        }
        
        const resultsList = document.createElement('ul');
        resultsList.className = 'batch-results-list';
        
        Object.entries(data).forEach(([filename, tags]) => {
            const listItem = document.createElement('li');
            listItem.innerHTML = `<strong>${filename}</strong>: ${tags}`;
            resultsList.appendChild(listItem);
        });
        
        batchOutput.appendChild(resultsList);
    }
    
    // Directory browse button (simulated, as browser security prevents direct folder selection)
    const browseBtn = document.getElementById('browse-btn');
    browseBtn.addEventListener('click', () => {
        alert('Due to browser security restrictions, direct folder selection is not available. Please manually enter the full directory path.');
    });
});