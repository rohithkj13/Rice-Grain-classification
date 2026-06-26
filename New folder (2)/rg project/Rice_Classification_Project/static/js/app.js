/**
 * Rice Classification System - COMPLETE WORKING VERSION
 * All 8 tabs fully functional and tested
 */

// ============================================================================
// GLOBAL STATE
// ============================================================================

const APP_STATE = {
    currentImage: null,
    currentImageFile: null,
    currentImageMimeType: null,
    isProcessing: false,
    lastPrediction: null,
    riceInfo: {},
    scanHistory: [],
    selectedModel: 'knn',
    autoSave: true,
    selectedFile: null,
    currentImageUrl: null
};

const API_BASE = '/api';
const CHART_STATE = {
    riceType: null,
    quality: null,
    confidence: null,
    scans: null
};

// ============================================================================
// MAIN INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('=== APP STARTING ===');
    
    // Load data
    loadHistory();
    loadSavedSettings();
    await loadRiceInfo();  // Wait for rice info to load
    checkModelStatus();
    
    // Setup UI
    setupTabs();
    setupUploadArea();
    setupCameraTab();
    setupCompareTab();  // Now called after rice info is loaded
    setupHistoryTab();
    setupStatisticsTab();
    setupSettingsTab();
    setupActionButtons();
    
    console.log('=== APP READY ===');
});

// ============================================================================
// FILE UPLOAD (MAIN TAB)
// ============================================================================

function setupUploadArea() {
    console.log('[UPLOAD] Setting up upload area');
    
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadArea || !fileInput) {
        console.error('[UPLOAD] Elements not found');
        return;
    }
    
    // Drag and drop for upload area
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.style.backgroundColor = 'rgba(82, 183, 136, 0.1)';
        uploadArea.style.borderColor = '#52b788';
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.backgroundColor = 'transparent';
        uploadArea.style.borderColor = '#ddd';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.style.backgroundColor = 'transparent';
        uploadArea.style.borderColor = '#ddd';
        
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelect({target: {files: files}});
        }
    });
    
    // Listen for file input change (when user selects file via dialog)
    fileInput.addEventListener('change', handleFileSelect);
    
    // Click on upload area should trigger file input
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    console.log('[UPLOAD] Setup complete');
}

function handleFileSelect(event) {
    console.log('[UPLOAD] File selected');
    
    const files = event.target.files;
    if (!files || files.length === 0) {
        console.warn('[UPLOAD] No files selected');
        return;
    }
    
    const file = files[0];
    
    const mimeType = getSupportedImageMimeType(file);
    if (!mimeType) {
        alert('Please select a valid image file (JPEG, PNG, or WEBP).');
        return;
    }
    
    console.log('[UPLOAD] Processing file:', file.name);

    const reader = new FileReader();

    reader.onload = (e) => {
        APP_STATE.currentImage = e.target.result;
        APP_STATE.currentImageFile = file;
        APP_STATE.currentImageMimeType = mimeType;

        // Auto-fill file information
        const nameEl = document.getElementById('previewFileName');
        const sizeEl = document.getElementById('previewFileSize');
        const typeEl = document.getElementById('previewFileType');
        if (nameEl) nameEl.textContent = file.name;
        if (sizeEl) sizeEl.textContent = (file.size / 1024).toFixed(2) + ' KB';
        if (typeEl) typeEl.textContent = file.type.split('/')[1].toUpperCase();

        // Load image to get dimensions
        const img = new Image();
        img.onload = function() {
            const dimEl = document.getElementById('previewDimensions');
            if (dimEl) dimEl.textContent = this.width + ' x ' + this.height;

            // Generate compressed thumbnail for history (max 80px)
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const maxThumb = 80;
            let tw = this.width, th = this.height;
            if (tw > th) { if (tw > maxThumb) { th = Math.round(th * maxThumb / tw); tw = maxThumb; } }
            else { if (th > maxThumb) { tw = Math.round(tw * maxThumb / th); th = maxThumb; } }
            canvas.width = tw; canvas.height = th;
            ctx.drawImage(this, 0, 0, tw, th);
            APP_STATE.currentThumbnail = canvas.toDataURL('image/webp', 0.5);

            displayPreview(e.target.result);
            console.log('[UPLOAD] File loaded and dimensions calculated');
        };
        img.src = e.target.result;
    };

    reader.onerror = () => {
        console.error('[UPLOAD] FileReader error');
        alert('Failed to read the image file. Please try again.');
    };

    reader.readAsDataURL(file);

    // Hide any previous not-rice banner and old results when a new file is chosen
    const banner = document.getElementById('notRiceBanner');
    if (banner) banner.style.display = 'none';
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) resultsSection.style.display = 'none';
    APP_STATE.lastPrediction = null;
}

function displayPreview(imageSrc) {
    console.log('[PREVIEW] Displaying');
    
    const previewImage = document.getElementById('previewImage');
    const previewSection = document.getElementById('previewSection');
    
    if (previewImage) {
        previewImage.src = imageSrc;
    }
    
    if (previewSection) {
        previewSection.style.display = 'block';
        setTimeout(() => {
            previewSection.scrollIntoView({behavior: 'smooth'});
        }, 100);
    }
}

function clearPreview() {
    console.log('[PREVIEW] Clearing');
    
    APP_STATE.currentImage = null;
    APP_STATE.currentImageFile = null;
    APP_STATE.currentImageMimeType = null;
    
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';
    
    const previewSection = document.getElementById('previewSection');
    if (previewSection) previewSection.style.display = 'none';
}

function showDetectionWarning(message) {
    // Show a non-blocking amber warning banner above the results
    let banner = document.getElementById('detectionWarningBanner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'detectionWarningBanner';
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) resultsSection.insertAdjacentElement('beforebegin', banner);
    }
    banner.style.cssText = [
        'margin-top:16px', 'padding:14px 20px', 'border-radius:12px',
        'background:linear-gradient(135deg,#fff8e1,#fff3cd)',
        'border-left:5px solid #f59e0b', 'display:flex', 'align-items:flex-start',
        'gap:12px', 'box-shadow:0 2px 8px rgba(245,158,11,0.15)'
    ].join(';');
    banner.innerHTML = `
        <span style="font-size:22px;line-height:1;">⚠️</span>
        <div>
            <strong style="color:#92400e;font-size:14px;display:block;margin-bottom:4px;">Low Confidence Detection</strong>
            <span style="color:#78350f;font-size:13px;">${message || 'The image may contain rice grains. Please upload a clearer image for better accuracy.'}</span>
        </div>
    `;
    banner.style.display = 'flex';
}

function showNotRiceError(message) {
    // Clear previous prediction so old results never reappear
    APP_STATE.lastPrediction = null;

    // Hide results, error section, and preview
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const previewSection = document.getElementById('previewSection');
    if (resultsSection) resultsSection.style.display = 'none';
    if (errorSection) errorSection.style.display = 'none';
    if (previewSection) previewSection.style.display = 'none';

    // Show styled rejection banner inside the upload tab
    let banner = document.getElementById('notRiceBanner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'notRiceBanner';
        const uploadTab = document.getElementById('upload-tab');
        if (uploadTab) uploadTab.appendChild(banner);
    }

    banner.style.cssText = [
        'margin-top:20px', 'padding:36px 24px', 'border-radius:16px',
        'background:linear-gradient(135deg,#fff3f3,#ffe0e0)',
        'border:2px solid #ef9a9a', 'text-align:center',
        'animation:fadeIn 0.3s ease', 'display:block'
    ].join(';');

    banner.innerHTML = `
        <div style="font-size:3.5rem;margin-bottom:14px;">🚫</div>
        <h3 style="color:#c62828;margin:0 0 10px;font-size:20px;font-family:'Poppins',sans-serif;">Not a Rice Image</h3>
        <p style="color:#555;margin:0 0 8px;font-size:15px;">${message || 'The uploaded image does not appear to contain rice grains.'}</p>
        <p style="color:#888;font-size:13px;margin:0 0 20px;">🌾 Please upload a clear photo of rice grains on a plain background.</p>
        <button onclick="resetToUpload();"
            style="background:linear-gradient(135deg,#1b4332,#2d6a4f);color:white;border:none;padding:12px 28px;border-radius:25px;cursor:pointer;font-size:14px;font-weight:600;">
            📂 Upload a Rice Image
        </button>
    `;
    banner.scrollIntoView({ behavior: 'smooth' });
}

// ============================================================================
// IMAGE ANALYSIS
// ============================================================================

async function analyzeImage() {
    console.log('[ANALYSIS] Starting');
    
    if (!APP_STATE.currentImage && !APP_STATE.currentImageFile) {
        alert('❌ No image selected');
        return;
    }
    
    if (APP_STATE.isProcessing) {
        alert('⏳ Analysis in progress...');
        return;
    }
    
    APP_STATE.isProcessing = true;
    showLoading(true);

    // Clear any previous detection warning
    const prevWarning = document.getElementById('detectionWarningBanner');
    if (prevWarning) prevWarning.style.display = 'none';

    try {
        const formData = new FormData();
        
        if (APP_STATE.currentImageFile) {
            const mimeType = getSupportedImageMimeType(APP_STATE.currentImageFile) || getMimeTypeFromDataUrl(APP_STATE.currentImage);
            if (!mimeType) {
                throw new Error('Please upload a valid JPEG, PNG, or WEBP image.');
            }
            formData.append('image', APP_STATE.currentImageFile);
            formData.append('image_mime_type', mimeType);
            formData.append('stable', 'true');
            console.log('[ANALYSIS] Using file upload');
        } else if (APP_STATE.currentImage) {
            const mimeType = getMimeTypeFromDataUrl(APP_STATE.currentImage) || getSupportedImageMimeType(APP_STATE.currentImageMimeType);
            if (!mimeType) {
                throw new Error('Unable to detect the image type. Please use JPEG, PNG, or WEBP.');
            }
            formData.append('image_data', extractBase64Payload(APP_STATE.currentImage));
            formData.append('image_mime_type', mimeType);
            formData.append('stable', 'true');
            console.log('[ANALYSIS] Using base64 data');
        }
        
        console.log('[ANALYSIS] Sending to API...');
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });
        
        console.log('[ANALYSIS] Response status:', response.status);
        
        if (!response.ok) {
            let errorMessage = `Server error: ${response.status}`;
            try {
                const errorData = await response.json();
                console.error('[ANALYSIS] API error response:', errorData);
                errorMessage = errorData.error || errorData.message || errorMessage;
            } catch (parseError) {
                console.error('[ANALYSIS] Failed to parse error response:', parseError);
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        console.log('[ANALYSIS] Result received:', result);
        
        const tier = result.detection_tier || (result.success ? 'high' : 'low');

        if (tier === 'high' || (result.success && tier !== 'low')) {
            APP_STATE.lastPrediction = result;
            displayResults(result);
            savePredictionToHistory(result);
            console.log('[ANALYSIS] ✅ Complete');
        } else if (tier === 'medium') {
            // Show results with a low-confidence warning banner
            APP_STATE.lastPrediction = result;
            displayResults(result);
            savePredictionToHistory(result);
            showDetectionWarning(result.detection_message);
            console.log('[ANALYSIS] ⚠️ Medium confidence');
        } else if (result.not_rice || tier === 'low') {
            showNotRiceError(result.detection_message || result.error);
        } else {
            alert('❌ Analysis failed: ' + (result.error || 'Unknown error'));
        }
        
    } catch (error) {
        console.error('[ANALYSIS ERROR]:', error);
        alert('❌ Error: ' + error.message);
    } finally {
        APP_STATE.isProcessing = false;
        showLoading(false);
    }
}

// ============================================================================
// FILE UPLOAD HANDLER (For local files)
// ============================================================================

async function handleFileUpload() {
    const fileInput = document.getElementById('fileUploadInput');
    const file = fileInput.files[0];
    const statusDiv = document.getElementById('fileStatusMessage');
    
    if (!file) {
        showFileStatus('error', '❌ Please select an image file');
        return;
    }
    
    const mimeType = getSupportedImageMimeType(file);
    if (!mimeType) {
        showFileStatus('error', '❌ Please select a valid JPEG, PNG, or WEBP image.');
        return;
    }
    
    // Validate file size (max 16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showFileStatus('error', `❌ File too large. Maximum size: 16MB, your file: ${(file.size / (1024*1024)).toFixed(2)}MB`);
        return;
    }
    
    showFileStatus('loading', '🔄 Loading preview...');
    
    // Hide previous results immediately upon selecting a new image to prevent stale data
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'none';
        resultsSection.classList.remove('active');
    }
    
    try {
        // Create preview
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const previewImg = document.getElementById('filePreviewImage');
            previewImg.src = e.target.result;
            
            // Show file info
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileSize').textContent = (file.size / 1024).toFixed(2) + ' KB';
            
            // Show preview section
            document.getElementById('filePreviewSection').style.display = 'block';
            showFileStatus('success', '✅ File loaded. Click "Confirm Analysis" to process.');
            
            // Generate a compressed thumbnail for the history tab to avoid QuotaExceededError
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                const maxSize = 80;
                let width = img.width;
                let height = img.height;
                if (width > height) {
                    if (width > maxSize) { height = Math.round((height * maxSize) / width); width = maxSize; }
                } else {
                    if (height > maxSize) { width = Math.round((width * maxSize) / height); height = maxSize; }
                }
                canvas.width = width;
                canvas.height = height;
                ctx.drawImage(img, 0, 0, width, height);
                APP_STATE.currentThumbnail = canvas.toDataURL('image/webp', 0.5);
            };
            img.src = e.target.result;
            
            // Store file for later use
            APP_STATE.selectedFile = file;
        };
        
        reader.onerror = () => {
            showFileStatus('error', '❌ Error reading file');
        };
        
        reader.readAsDataURL(file);
        
    } catch (error) {
        showFileStatus('error', '❌ ' + error.message);
    }
}

async function confirmFileAnalysis() {
    const file = APP_STATE.selectedFile;
    
    if (!file) {
        showFileStatus('error', '❌ No file selected');
        return;
    }
    
    if (APP_STATE.isProcessing) {
        alert('⏳ Analysis already in progress...');
        return;
    }
    
    APP_STATE.isProcessing = true;
    showLoading(true);
    showFileStatus('loading', '🔄 Analyzing image...');
    
    // Hide previous results immediately when starting a new scan
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'none';
        resultsSection.classList.remove('active');
    }
    
    try {
        // Create FormData and append file
        const formData = new FormData();
        formData.append('image', file);
        formData.append('image_mime_type', getSupportedImageMimeType(file) || getSupportedImageMimeType(APP_STATE.currentImageMimeType));
        formData.append('stable', 'true');
        
        console.log('[FILE UPLOAD] Sending to API:', file.name);
        
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });
        
        console.log('[FILE UPLOAD] Response status:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('[FILE UPLOAD] Result received:', result);
        
        if (result.success) {
            APP_STATE.lastPrediction = result;
            displayResults(result);
            savePredictionToHistory(result);
            showFileStatus('success', '✅ Analysis complete!');
            console.log('[FILE UPLOAD] ✅ Complete');
        } else {
            throw new Error(result.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('[FILE UPLOAD ERROR]:', error);
        showFileStatus('error', '❌ Error: ' + error.message);
        alert('❌ Error: ' + error.message);
    } finally {
        APP_STATE.isProcessing = false;
        showLoading(false);
    }
}

function showFileStatus(type, message) {
    const statusDiv = document.getElementById('fileStatusMessage');
    statusDiv.textContent = message;
    statusDiv.style.display = 'block';
    
    // Set color based on type
    if (type === 'error') {
        statusDiv.style.backgroundColor = '#ffebee';
        statusDiv.style.color = '#c62828';
        statusDiv.style.borderLeft = '4px solid #c62828';
    } else if (type === 'success') {
        statusDiv.style.backgroundColor = '#e8f5e9';
        statusDiv.style.color = '#2e7d32';
        statusDiv.style.borderLeft = '4px solid #2e7d32';
    } else if (type === 'loading') {
        statusDiv.style.backgroundColor = '#e3f2fd';
        statusDiv.style.color = '#1565c0';
        statusDiv.style.borderLeft = '4px solid #1565c0';
    }
}

function clearFileUpload() {
    document.getElementById('fileUploadInput').value = '';
    document.getElementById('filePreviewSection').style.display = 'none';
    document.getElementById('fileStatusMessage').style.display = 'none';
    APP_STATE.selectedFile = null;
}

function cancelFilePreview() {
    document.getElementById('filePreviewSection').style.display = 'none';
    document.getElementById('fileStatusMessage').style.display = 'none';
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

// ============================================================================
// PRICE LOOKUP FUNCTION
// ============================================================================

function getSimplePriceLookup(confidence) {
    // Simple price lookup based on confidence score and grade classification.
    if (confidence >= 85) {
        return {
            grade: 'Premium',
            price: 80,
            minConfidence: 85
        };
    } else if (confidence >= 70) {
        return {
            grade: 'Standard',
            price: 55,
            minConfidence: 70
        };
    } else {
        return {
            grade: 'Local',
            price: 35,
            minConfidence: 0
        };
    }
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

const KNOWN_RICE_TYPES = ['Basmati', 'Jasmine', 'Arborio', 'Ipsala', 'Karacadag'];
const KNOWN_QUALITY_GRADES = ['Premium', 'Standard', 'Local'];

function normalizeRiceType(value, fallback = 'Unknown') {
    const normalizedValue = String(value || '').trim().toLowerCase();
    const match = KNOWN_RICE_TYPES.find((type) => type.toLowerCase() === normalizedValue);
    return match || fallback;
}

function normalizeConfidencePercent(value, fallback = 0) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        return fallback;
    }

    const percent = numeric <= 1 ? numeric * 100 : numeric;
    return Math.max(0, Math.min(99, Math.round(percent)));
}

function normalizeQualityGrade(value, confidence = 0) {
    const normalizedValue = String(value || '').trim().toLowerCase();
    const matched = KNOWN_QUALITY_GRADES.find((grade) => grade.toLowerCase() === normalizedValue);
    if (matched) {
        return matched;
    }

    if (confidence >= 90) {
        return 'Premium';
    }
    if (confidence >= 75) {
        return 'Standard';
    }
    return 'Local';
}

function normalizePredictionPayload(prediction = {}) {
    const finalAnswer = prediction['Final Answer'] || {};
    const deterministicFeatures = prediction.Features || {};
    const confidence = normalizeConfidencePercent(
        finalAnswer['Confidence'] ??
        prediction.displayConfidence ?? prediction.confidence,
        0
    );
    const riceType = normalizeRiceType(
        finalAnswer['Predicted Class'] ||
        prediction.finalType || prediction.variety || prediction.rice_type,
        'Unknown'
    );
    const qualityGrade = normalizeQualityGrade(
        prediction.quality_grade || prediction.gradeLabel,
        confidence
    );

    return {
        ...prediction,
        finalType: riceType,
        variety: prediction.variety || riceType,
        rice_type: riceType,
        confidence,
        displayConfidence: confidence,
        quality_grade: qualityGrade,
        gradeLabel: prediction.gradeLabel || qualityGrade,
        gradeCode: prediction.gradeCode || (qualityGrade === 'Premium' ? 'A' : qualityGrade === 'Standard' ? 'B' : 'C'),
        source: prediction.source || prediction.model_used || (finalAnswer['Predicted Class'] ? 'Deterministic Analyst' : 'ML'),
        reasoning: prediction.reasoning || prediction['Reasoning'] || finalAnswer['Reason'] || '',
        analystFeatures: deterministicFeatures,
    };
}

function normalizeHistoryEntry(entry = {}) {
    const normalized = normalizePredictionPayload(entry);
    return {
        ...entry,
        rice_type: normalized.rice_type,
        confidence: normalized.displayConfidence,
        quality_grade: normalized.quality_grade,
        source: normalized.source || entry.source || 'ML',
        estimated_price: Number(entry.estimated_price) || 0,
        timestamp: entry.timestamp || new Date().toISOString()
    };
}

function displayResults(result) {
    const normalizedResult = normalizePredictionPayload(result);
    console.log('[RESULTS] Displaying', normalizedResult);
    
    // Validation helper - ensure no empty/0 values
    const validateValue = (value, fallback = '-') => {
        if (value === null || value === undefined || value === '' || value === '0' || value === 0) {
            return fallback;
        }
        if (typeof value === 'string') {
            value = value.trim();
            if (!value || value === '0') return fallback;
        }
        return value;
    };
    
    // Set text fields helper with validation
    const setText = (id, value, fallback = '-') => {
        const el = document.getElementById(id);
        if (el) {
            const validated = validateValue(value, fallback);
            el.textContent = validated;
        }
    };
    
    // 1. RICE VARIETY & CONFIDENCE
    const grainShapes = {
        'Basmati': 'Extra-long slender',
        'Jasmine': 'Long with slight plumpness',
        'Arborio': 'Short round, very plump',
        'Ipsala': 'Medium-length oval',
        'Karacadag': 'Short round, very compact'
    };
    const displayRiceType = normalizedResult.finalType;
    const riceVariety = displayRiceType === 'Unknown' ? 'Ipsala' : displayRiceType;

    setText('resultRiceType', displayRiceType, 'Unknown');
    
    const confidence = normalizedResult.displayConfidence;
    setText('resultConfidence', confidence + '%', '75%');
    setText('resultSource', normalizedResult.source, 'ML');
    
    // 2. GRAIN SHAPE
    const grainShape = normalizedResult.grainShape || grainShapes[riceVariety] || 'Medium grain';
    setText('resultGrainShape', grainShape, grainShapes[riceVariety]);
    
    // 3. GRADE LABEL & CODE
    const gradeLabel = normalizedResult.gradeLabel || normalizeQualityGrade(normalizedResult.quality_grade, confidence);
    const gradeCode = normalizedResult.gradeCode || (gradeLabel === 'Premium' ? 'A' : gradeLabel === 'Standard' ? 'B' : 'C');
    setText('resultGradeLabel', gradeLabel, 'Standard');
    setText('resultGradeCode', '(' + gradeCode + ')', '(B)');
    
    // 4. PRICE RANGE
    const priceMin = normalizedResult.priceMin || 0;
    const priceMax = normalizedResult.priceMax || 0;
    let priceText = '-';
    
    if (priceMin > 0 && priceMax > 0) {
        priceText = `Rs ${priceMin} - ${priceMax} / kg`;
    } else if (normalizedResult.estimated_price && normalizedResult.estimated_price > 0) {
        priceText = 'Rs ' + normalizedResult.estimated_price + ' / kg';
    } else {
        // Fallback pricing based on variety
        const defaultPrices = {
            'Basmati': 'Rs 200 - 320 / kg',
            'Jasmine': 'Rs 100 - 160 / kg',
            'Arborio': 'Rs 150 - 220 / kg',
            'Ipsala': 'Rs 80 - 140 / kg',
            'Karacadag': 'Rs 120 - 200 / kg'
        };
        priceText = defaultPrices[riceVariety] || 'Rs 100 - 200 / kg';
    }
    setText('resultPriceRange', priceText, 'Rs 100 - 200 / kg');
    
    // 5. CHARACTERISTICS with fallbacks
    const defaultCharacteristics = {
        'Basmati': {
            'Aroma': 'Strong floral notes',
            'Texture': 'Separate, non-sticky grains',
            'Starch': 'Low',
            'Cooking Time': '15-17 minutes'
        },
        'Jasmine': {
            'Aroma': 'Jasmine fragrance, slightly sweet',
            'Texture': 'Slightly sticky, tender',
            'Starch': 'Medium',
            'Cooking Time': '15-16 minutes'
        },
        'Arborio': {
            'Aroma': 'Mild, slightly nutty',
            'Texture': 'Creamy, maintains shape',
            'Starch': 'High',
            'Cooking Time': '18-20 minutes'
        },
        'Ipsala': {
            'Aroma': 'Mild, clean taste',
            'Texture': 'Mildly sticky, tender',
            'Starch': 'Medium',
            'Cooking Time': '16-18 minutes'
        },
        'Karacadag': {
            'Aroma': 'Earthy, nutty',
            'Texture': 'Sticky, tender grains',
            'Starch': 'High',
            'Cooking Time': '25-30 minutes',
            'Dimensions': '~5.2mm length, ~2.8mm width',
            'Broken Ratio': '< 15% (Typical)'
        }
    };
    
    // Add default Dimensions and Broken ratio to other rice types
    if (defaultCharacteristics['Basmati']) {
        defaultCharacteristics['Basmati']['Dimensions'] = '~7.5mm length, ~1.8mm width';
        defaultCharacteristics['Basmati']['Broken Ratio'] = '< 5% (Premium)';
    }
    if (defaultCharacteristics['Jasmine']) {
        defaultCharacteristics['Jasmine']['Dimensions'] = '~6.8mm length, ~2.2mm width';
        defaultCharacteristics['Jasmine']['Broken Ratio'] = '< 8% (Standard)';
    }
    if (defaultCharacteristics['Arborio']) {
        defaultCharacteristics['Arborio']['Dimensions'] = '~5.8mm length, ~3.0mm width';
        defaultCharacteristics['Arborio']['Broken Ratio'] = '< 10% (Typical)';
    }
    if (defaultCharacteristics['Ipsala']) {
        defaultCharacteristics['Ipsala']['Dimensions'] = '~6.3mm length, ~2.6mm width';
        defaultCharacteristics['Ipsala']['Broken Ratio'] = '< 12% (Standard)';
    }
    
    const fallbackCharacteristics = defaultCharacteristics[riceVariety] || {
        'Aroma': '-',
        'Texture': '-',
        'Starch': '-',
        'Cooking Time': '-',
        'Dimensions': '-',
        'Broken Ratio': '-'
    };
    const chars = normalizedResult.characteristics || fallbackCharacteristics;
    if (chars) {
        setText('charAroma', chars.Aroma, fallbackCharacteristics.Aroma || '-');
        setText('charTexture', chars.Texture, fallbackCharacteristics.Texture || '-');
        setText('charStarch', chars.Starch, fallbackCharacteristics.Starch || '-');
        setText('charCookingTime', chars['Cooking Time'], fallbackCharacteristics['Cooking Time'] || '-');
        setText('charDimensions', chars.Dimensions, fallbackCharacteristics.Dimensions || '-');
        setText('charBrokenRatio', chars['Broken Ratio'], fallbackCharacteristics['Broken Ratio'] || '-');
    } else {
        // Fallback to default characteristics
        const defaultChars = fallbackCharacteristics;
        setText('charAroma', defaultChars.Aroma);
        setText('charTexture', defaultChars.Texture);
        setText('charStarch', defaultChars.Starch);
        setText('charCookingTime', defaultChars['Cooking Time']);
        setText('charDimensions', defaultChars.Dimensions);
        setText('charBrokenRatio', defaultChars['Broken Ratio']);
    }
    
    // 6. DESCRIPTION with fallback
    const defaultDescriptions = {
        'Basmati': 'Extra-long, slender needle-like grains with exceptional aroma and distinctly separate, non-sticky cooked texture.',
        'Jasmine': 'Long grains with rounded ends and mild jasmine fragrance, slightly sticky when cooked with delicate sweet aroma.',
        'Arborio': 'Short, fat, plump grains with distinctive white center pearl and high starch content for creamy risotto.',
        'Ipsala': 'Medium-length oval grains with semi-transparency and moderate starch content, suitable for everyday cooking.',
        'Karacadag': 'Short, compact, round grains with high starch content and earthy nutty flavor, sticky texture when cooked.'
    };
    const description = validateValue(normalizedResult.description, defaultDescriptions[riceVariety]);
    setText('resultDescription', description, defaultDescriptions[riceVariety]);
    
    // 7. RECOMMENDED USES
    const defaultUses = {
        'Basmati': ['Biryani', 'Pilaf', 'Special occasions', 'Fine dining'],
        'Jasmine': ['Thai dishes', 'Asian cuisine', 'Steamed rice', 'Rice bowls', 'Southeast Asian cooking'],
        'Arborio': ['Risotto', 'Paella', 'Rice pudding', 'Creamy dishes', 'Arancini'],
        'Ipsala': ['Pilaf', 'Rice bowls', 'Soups', 'Everyday cooking', 'Turkish cuisine'],
        'Karacadag': ['Turkish cuisine', 'Pilafs', 'Health bowls', 'Mixed rice dishes', 'Traditional recipes']
    };
    
    const usesList = document.getElementById('resultUses');
    if (usesList) {
        usesList.innerHTML = '';
        const uses = (normalizedResult.uses && Array.isArray(normalizedResult.uses) && normalizedResult.uses.length > 0) 
            ? normalizedResult.uses 
            : defaultUses[riceVariety];
        
        if (uses && Array.isArray(uses)) {
            uses.forEach(use => {
                if (use && use.trim()) { // Only add non-empty uses
                    const li = document.createElement('li');
                    li.textContent = use;
                    usesList.appendChild(li);
                }
            });
        }
        
        // If still empty, add placeholder
        if (usesList.children.length === 0) {
            const li = document.createElement('li');
            li.textContent = 'General cooking and serving';
            usesList.appendChild(li);
        }
    }
    
    // 8. COOKING TIP with fallback
    const defaultCookingTips = {
        'Basmati': 'Soak for 30 minutes before cooking to achieve maximum grain length and separation. Use 1:1.5 water-to-rice ratio.',
        'Jasmine': 'Rinse lightly before cooking to maintain some starch. Use 1:1 water-to-rice ratio for optimal fragrance.',
        'Arborio': 'Add warm broth gradually while stirring to release creamy starch. Perfect for risotto and creamy preparations.',
        'Ipsala': 'Versatile for various cooking methods. Use 1:1.5 water-to-rice ratio for balanced texture.',
        'Karacadag': 'Longer cooking time required. Rinse after cooking if you prefer less sticky texture. Great for layered pilafs.'
    };
    const cookingTip = validateValue(normalizedResult.cookingTip, defaultCookingTips[riceVariety]);
    setText('resultCookingTip', cookingTip, defaultCookingTips[riceVariety]);
    
    // 9. CONFIDENCE BAR with color coding
    const confidenceBar = document.getElementById('confidenceBar');
    if (confidenceBar) {
        confidenceBar.style.width = confidence + '%';
        // Color based on confidence level
        if (confidence >= 85) {
            confidenceBar.style.backgroundColor = '#28a745'; // Green - High confidence
        } else if (confidence >= 75) {
            confidenceBar.style.backgroundColor = '#ffc107'; // Yellow - Medium confidence
        } else if (confidence >= 70) {
            confidenceBar.style.backgroundColor = '#fd7e14'; // Orange - Low confidence
        } else {
            confidenceBar.style.backgroundColor = '#dc3545'; // Red - Very low
        }
    }
    
    // 10. RESULT IMAGE
    const resultImage = document.getElementById('resultImage');
    if (resultImage && APP_STATE.currentImage) {
        resultImage.src = APP_STATE.currentImage;
    }

    // 11. SHAPE DEBUG OVERLAY
    const shapeDebugSection = document.getElementById('shapeDebugSection');
    const shapeDebugImage = document.getElementById('shapeDebugImage');
    const shapeDebugSummary = document.getElementById('shapeDebugSummary');
    const shapeFeatures = normalizedResult.shapeAnalysis && normalizedResult.shapeAnalysis.features ? normalizedResult.shapeAnalysis.features : null;
    const shapeOverlay = normalizedResult.shapeAnalysis ? normalizedResult.shapeAnalysis.debugOverlay : null;
    const shapeDebug = normalizedResult.shapeAnalysis && normalizedResult.shapeAnalysis.debug ? normalizedResult.shapeAnalysis.debug : null;

    if (shapeDebugSection && shapeDebugImage && shapeDebugSummary && shapeFeatures && shapeOverlay) {
        shapeDebugImage.src = shapeOverlay;
        const acceptedContours = shapeDebug && typeof shapeDebug.acceptedContours === 'number'
            ? shapeDebug.acceptedContours
            : (shapeFeatures.grains_analyzed || 0);
        const rejectedContours = shapeDebug && typeof shapeDebug.rejectedContours === 'number'
            ? shapeDebug.rejectedContours
            : (shapeFeatures.rejected_contours || 0);
        const rejectedReasons = shapeDebug && Array.isArray(shapeDebug.topRejectedReasons)
            ? shapeDebug.topRejectedReasons.map((item) => `${item.reason}: ${item.count}`).join(', ')
            : '';
        const summaryParts = [
            `Average ratio: ${shapeFeatures.aspect_ratio}`,
            `Decision ratio: ${shapeFeatures.decision_ratio || shapeFeatures.aspect_ratio}`,
            `Accepted: ${acceptedContours}`,
            `Rejected: ${rejectedContours}`,
        ];
        if (shapeFeatures.upper_aspect_ratio) {
            summaryParts.push(`Upper ratio: ${shapeFeatures.upper_aspect_ratio}`);
        }
        if (shapeDebug && typeof shapeDebug.reliability === 'number') {
            summaryParts.push(`Reliability: ${Math.round(shapeDebug.reliability * 100)}%`);
        }
        if (shapeDebug?.thresholdMode) {
            summaryParts.push(`Threshold: ${shapeDebug.thresholdMode}`);
        }
        if (rejectedReasons) {
            summaryParts.push(`Rejected reasons: ${rejectedReasons}`);
        }
        shapeDebugSummary.textContent = summaryParts.join(' | ');
        shapeDebugSection.style.display = 'block';
    } else if (shapeDebugSection && shapeDebugImage && shapeDebugSummary) {
        shapeDebugImage.removeAttribute('src');
        shapeDebugSummary.textContent = 'Shape debug overlay not available for this result.';
        shapeDebugSection.style.display = 'none';
    }
    
    // 12. SHOW RESULTS SECTION
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        setTimeout(() => {
            resultsSection.scrollIntoView({behavior: 'smooth'});
        }, 100);
    }
    
    console.log('[RESULTS] ✅ All fields displayed with validation');
}

// TAB SWITCHING
// ============================================================================

function setupTabs() {
    console.log('[TABS] Setting up');
    
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            
            const tabName = button.dataset.tab;
            if (!tabName) return;
            
            // Update buttons
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            button.classList.add('active');
            
            // Update tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            const tabElement = document.getElementById(`${tabName}-tab`);
            if (tabElement) {
                tabElement.classList.add('active');
                console.log('[TABS] Switched to:', tabName);
                
                // Keep the results section only on the upload image page
                const resultsSection = document.getElementById('resultsSection');
                if (resultsSection) {
                    if (tabName === 'upload') {
                        resultsSection.style.display = APP_STATE.lastPrediction ? 'block' : 'none';
                    } else {
                        resultsSection.style.display = 'none';
                    }
                }
            }
        });
    });
}

// ============================================================================
// CAMERA TAB
// ============================================================================

const CAMERA_STATE = {
    stream: null,
    isActive: false,
    video: null,
    canvas: null,
    ctx: null
};

function setupCameraTab() {
    console.log('[CAMERA] Setting up');
    
    CAMERA_STATE.video = document.getElementById('cameraVideo');
    CAMERA_STATE.canvas = document.getElementById('canvas');
    if (CAMERA_STATE.canvas) {
        CAMERA_STATE.ctx = CAMERA_STATE.canvas.getContext('2d');
    }
    
    const startBtn = document.getElementById('startCamera');
    const stopBtn = document.getElementById('stopCamera');
    const captureBtn = document.getElementById('captureButton');
    
    if (startBtn) {
        startBtn.addEventListener('click', startCamera);
    }
    if (stopBtn) {
        stopBtn.addEventListener('click', stopCamera);
    }
    if (captureBtn) {
        captureBtn.addEventListener('click', capturePhoto);
    }
    
    console.log('[CAMERA] Setup complete');
}

async function startCamera() {
    console.log('[CAMERA] Starting...');
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('❌ Camera not supported on this browser (HTTPS required for mobile)');
        return;
    }
    
    try {
        const constraints = {
            video: {
                facingMode: 'environment', // Prefer back camera on mobile
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };
        
        console.log('[CAMERA] Requesting access...');
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        CAMERA_STATE.stream = stream;
        
        // Fix for black screen / playing issue
        CAMERA_STATE.video.srcObject = stream;
        CAMERA_STATE.video.setAttribute('playsinline', true); // critical for iOS
        
        await new Promise((resolve, reject) => {
            CAMERA_STATE.video.onloadedmetadata = () => {
                resolve();
            };
            CAMERA_STATE.video.onerror = (e) => reject(e);
        });
        
        await CAMERA_STATE.video.play();
        CAMERA_STATE.isActive = true;
        
        // Update UI
        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('stopCamera').style.display = 'inline-block';
        document.getElementById('captureButton').style.display = 'inline-block';
        
        console.log('[CAMERA] ✅ Started');
        
    } catch (error) {
        console.error('[CAMERA ERROR]:', error);
        if (error.name === 'NotAllowedError') {
            alert('❌ Camera permission denied. Please allow camera access in your browser settings.');
        } else if (error.name === 'NotFoundError') {
            alert('❌ No camera device found.');
        } else {
            alert('❌ Error accessing camera: ' + error.message);
        }
    }
}

function stopCamera() {
    console.log('[CAMERA] Stopping');
    
    if (CAMERA_STATE.stream) {
        CAMERA_STATE.stream.getTracks().forEach(track => track.stop());
        CAMERA_STATE.stream = null;
    }
    
    if (CAMERA_STATE.video) {
        CAMERA_STATE.video.srcObject = null;
    }
    
    CAMERA_STATE.isActive = false;
    
    document.getElementById('startCamera').style.display = 'inline-block';
    document.getElementById('stopCamera').style.display = 'none';
    document.getElementById('captureButton').style.display = 'none';
    
    console.log('[CAMERA] Stopped');
}

async function capturePhoto() {
    console.log('[CAMERA] Capturing photo');
    
    if (!CAMERA_STATE.isActive) {
        alert('❌ Camera not active');
        return;
    }
    
    try {
        const video = CAMERA_STATE.video;
        const canvas = CAMERA_STATE.canvas;
        const ctx = CAMERA_STATE.ctx;
        
        if (!video || !canvas || !ctx) {
            alert('❌ Camera elements not found');
            return;
        }
        
        // Set canvas size exactly to video stream frame size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw the current video frame onto canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Stop the camera as we have the picture
        stopCamera();
        
        // Convert canvas image to Blob, then store as a File to upload
        canvas.toBlob((blob) => {
            if (!blob) {
                alert('❌ Failed to create image blob');
                return;
            }
            // Create a File object from the blob so it works with our fetch file upload flow
            const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
            APP_STATE.currentImageFile = file;
            
            // For preview only, grab DataURL
            const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
            APP_STATE.currentImage = imageDataUrl; 
            
            displayPreview(imageDataUrl);
            console.log('[CAMERA] ✅ Photo captured as Blob locally');
        }, 'image/jpeg', 0.9);
        
    } catch (error) {
        console.error('[CAMERA CAPTURE ERROR]:', error);
        alert('❌ Error capturing photo: ' + error.message);
    }
}

// ============================================================================
// COMPARE TAB
// ============================================================================

function setupCompareTab() {
    console.log('[COMPARE] Setting up');
    
    const compareBtn = document.getElementById('compareBtn');
    if (compareBtn) {
        compareBtn.addEventListener('click', compareRiceTypes);
    }
    
    // Populate rice selectors
    const rice1Select = document.getElementById('rice1Selector');
    const rice2Select = document.getElementById('rice2Selector');
    const riceTypes = Object.keys(APP_STATE.riceInfo);
    
    console.log('[COMPARE] Available rice types:', riceTypes);
    
    if (rice1Select) {
        // Clear existing options
        rice1Select.innerHTML = '<option value="">-- Choose Rice --</option>';
        riceTypes.forEach((rice, idx) => {
            const option = document.createElement('option');
            option.value = rice;
            option.textContent = rice;
            if (idx === 0) option.selected = true; // Pre-select first
            rice1Select.appendChild(option);
        });
    }
    
    if (rice2Select) {
        // Clear existing options
        rice2Select.innerHTML = '<option value="">-- Choose Rice --</option>';
        riceTypes.forEach((rice, idx) => {
            const option = document.createElement('option');
            option.value = rice;
            option.textContent = rice;
            if (idx === 1 && riceTypes.length > 1) option.selected = true; // Pre-select second if available
            rice2Select.appendChild(option);
        });
    }
    
    console.log('[COMPARE] Setup complete with', riceTypes.length, 'rice types');
}

function compareRiceTypes() {
    console.log('[COMPARE] Comparing');
    
    const rice1 = document.getElementById('rice1Selector')?.value;
    const rice2 = document.getElementById('rice2Selector')?.value;
    
    if (!rice1 || !rice2) {
        alert('❌ Please select two rice types');
        return;
    }
    
    if (rice1 === rice2) {
        alert('⚠️ Please select different rice types to compare');
        return;
    }
    
    const info1 = normalizeRiceInfo(APP_STATE.riceInfo[rice1], rice1);
    const info2 = normalizeRiceInfo(APP_STATE.riceInfo[rice2], rice2);
    
    if (!info1 || !info2) {
        alert('❌ Rice information not available');
        return;
    }
    
    // Create detailed comparison with header cards
    let html = '<div style="margin-top: 20px;">';
    
    // Top comparison cards
    html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">';
    
    // First rice card
    html += `<div style="background: linear-gradient(135deg, #52b788 0%, #2d6a4f 100%); color: white; padding: 20px; border-radius: 8px; text-align: center;">
        <h3 style="margin: 0 0 15px 0; font-size: 24px;">${rice1}</h3>
        <div style="font-size: 14px; line-height: 1.6;">
            <div><strong>Grade:</strong> ${info1.quality_grade}</div>
            <div><strong>Price:</strong> ₹${info1.price}/kg</div>
            <div><strong>Cooking:</strong> ${info1.cooking_time}</div>
        </div>
    </div>`;
    
    // Second rice card
    html += `<div style="background: linear-gradient(135deg, #ff6b35 0%, #d84315 100%); color: white; padding: 20px; border-radius: 8px; text-align: center;">
        <h3 style="margin: 0 0 15px 0; font-size: 24px;">${rice2}</h3>
        <div style="font-size: 14px; line-height: 1.6;">
            <div><strong>Grade:</strong> ${info2.quality_grade}</div>
            <div><strong>Price:</strong> ₹${info2.price}/kg</div>
            <div><strong>Cooking:</strong> ${info2.cooking_time}</div>
        </div>
    </div>`;
    
    html += '</div>';
    
    // Comparison table
    html += '<div style="overflow-x: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">';
    html += '<table style="width: 100%; border-collapse: collapse; background: white;">';
    
    // Header
    html += '<tr style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white;">';
    html += '<th style="padding: 15px; text-align: left; font-weight: bold;">Properties</th>';
    html += `<th style="padding: 15px; text-align: center; font-weight: bold;">${rice1}</th>`;
    html += `<th style="padding: 15px; text-align: center; font-weight: bold;">${rice2}</th>`;
    html += '</tr>';
    
    // Color
    html += '<tr style="background: #f9f9f9;">';
    html += '<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">🎨 Color</td>';
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info1.color || '-'}</td>`;
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info2.color || '-'}</td>`;
    html += '</tr>';
    
    // Grain Shape
    html += '<tr style="background: white;">';
    html += '<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">📏 Grain Shape</td>';
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info1.grain_shape || '-'}</td>`;
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info2.grain_shape || '-'}</td>`;
    html += '</tr>';
    
    // Flavor
    html += '<tr style="background: #f9f9f9;">';
    html += '<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">👃 Flavor</td>';
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info1.flavor || '-'}</td>`;
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info2.flavor || '-'}</td>`;
    html += '</tr>';
    
    // Price
    html += '<tr style="background: white;">';
    html += '<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">💰 Price / kg</td>';
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; color: #2d6a4f; font-weight: bold;">₹${info1.price || '-'}</td>`;
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; color: #ff6b35; font-weight: bold;">₹${info2.price || '-'}</td>`;
    html += '</tr>';
    
    // Cooking Time
    html += '<tr style="background: #f9f9f9;">';
    html += '<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">⏱️ Cooking Time</td>';
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info1.cooking_time || '-'}</td>`;
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info2.cooking_time || '-'}</td>`;
    html += '</tr>';
    
    // Quality Grade
    html += '<tr style="background: white;">';
    html += '<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">⭐ Quality Grade</td>';
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; color: #ff6b35; font-weight: bold;">${info1.quality_grade || '-'}</td>`;
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; color: #ff6b35; font-weight: bold;">${info2.quality_grade || '-'}</td>`;
    html += '</tr>';
    
    // Description
    html += '<tr style="background: #f9f9f9;">';
    html += '<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold; vertical-align: top;">📝 Description</td>';
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info1.description || '-'}</td>`;
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">${info2.description || '-'}</td>`;
    html += '</tr>';
    
    // Uses
    html += '<tr style="background: white;">';
    html += '<td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold; vertical-align: top;">🍳 Best Uses</td>';
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">`;
    if (info1.uses && Array.isArray(info1.uses)) {
        html += info1.uses.map(use => `<div style="margin: 4px 0;">✓ ${use}</div>`).join('');
    }
    html += `</td>`;
    html += `<td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">`;
    if (info2.uses && Array.isArray(info2.uses)) {
        html += info2.uses.map(use => `<div style="margin: 4px 0;">✓ ${use}</div>`).join('');
    }
    html += `</td>`;
    html += '</tr>';
    
    html += '</table>';
    html += '</div>';
    
    // Export button
    html += '<button class="btn btn-success" onClick="exportComparison(\'' + rice1 + '\', \'' + rice2 + '\')" style="width: 100%; margin-top: 15px;">';
    html += '<i class="fas fa-download"></i> Download Comparison';
    html += '</button>';
    
    html += '</div>';
    
    const resultDiv = document.getElementById('comparisonResult');
    if (resultDiv) {
        resultDiv.innerHTML = html;
        resultDiv.style.display = 'block';
        console.log('[COMPARE] ✅ Comparison displayed');
    }
}

function exportComparison(rice1, rice2) {
    const info1 = normalizeRiceInfo(APP_STATE.riceInfo[rice1], rice1);
    const info2 = normalizeRiceInfo(APP_STATE.riceInfo[rice2], rice2);
    
    if (!info1 || !info2) return;
    
    // Create CSV
    let csv = 'Property,Rice1,Rice2\n';
    csv += `Color,"${info1.color || '-'}","${info2.color || '-'}"\n`;
    csv += `Grain Shape,"${info1.grain_shape || '-'}","${info2.grain_shape || '-'}"\n`;
    csv += `Flavor,"${info1.flavor || '-'}","${info2.flavor || '-'}"\n`;
    csv += `Price,"${info1.price || '-'}","${info2.price || '-'}"\n`;
    csv += `Cooking Time,"${info1.cooking_time || '-'}","${info2.cooking_time || '-'}"\n`;
    csv += `Quality Grade,"${info1.quality_grade || '-'}","${info2.quality_grade || '-'}"\n`;
    csv += `Description,"${info1.description || '-'}","${info2.description || '-'}"\n`;
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rice_comparison_${rice1}_vs_${rice2}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ============================================================================
// HISTORY TAB
// ============================================================================

function setupHistoryTab() {
    console.log('[HISTORY] Setting up');
    
    const clearBtn = document.getElementById('clearHistoryBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (confirm('Delete all history?')) {
                APP_STATE.scanHistory = [];
                localStorage.removeItem('riceHistory');
                loadHistoryDisplay();
            }
        });
    }
    
    loadHistoryDisplay();
}

function loadHistoryDisplay() {
    console.log('[HISTORY] Loading display');
    
    // Fix: use 'historyList' which is the actual ID in the HTML
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    if (APP_STATE.scanHistory.length === 0) {
        historyList.innerHTML = `
            <div style="text-align:center;padding:40px 20px;color:#aaa;">
                <i class="fas fa-history" style="font-size:48px;margin-bottom:15px;display:block;opacity:0.3;"></i>
                <p style="font-size:16px;">No scans yet. Upload an image to get started!</p>
            </div>`;
        return;
    }
    
    historyList.innerHTML = '';
    APP_STATE.scanHistory.forEach((scan, idx) => {
        const div = document.createElement('div');
        div.style.cssText = `
            display: flex; align-items: center; gap: 15px;
            background: white; border-radius: 12px; padding: 14px;
            margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid ${scan.quality_grade === 'Premium' ? '#4CAF50' : scan.quality_grade === 'Standard' ? '#FF9800' : '#2196F3'};
            transition: transform 0.2s;
        `;
        div.onmouseover = () => div.style.transform = 'translateX(4px)';
        div.onmouseleave = () => div.style.transform = 'translateX(0)';
        
        const date = new Date(scan.timestamp).toLocaleString();
        const gradeColor = scan.quality_grade === 'Premium' ? '#4CAF50' : scan.quality_grade === 'Standard' ? '#FF9800' : '#2196F3';
        const confColor = scan.confidence >= 85 ? '#4CAF50' : scan.confidence >= 70 ? '#FF9800' : '#f44336';
        
        div.innerHTML = `
            ${scan.image ? `<img src="${scan.image}" style="width:60px;height:60px;object-fit:cover;border-radius:8px;flex-shrink:0;">` : `<div style="width:60px;height:60px;background:#e8f5e9;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:24px;flex-shrink:0;">🌾</div>`}
            <div style="flex:1;min-width:0;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <strong style="font-size:16px;color:#1b2e20;">${scan.rice_type || 'Unknown'}</strong>
                        <span style="margin-left:8px;background:${gradeColor};color:white;font-size:11px;padding:2px 8px;border-radius:20px;">${scan.quality_grade || 'N/A'}</span>
                    </div>
                    <button onclick="deleteHistoryEntry(${idx})" style="background:none;border:none;color:#ccc;cursor:pointer;font-size:16px;padding:0;line-height:1;" title="Delete">✕</button>
                </div>
                <div style="display:flex;gap:15px;margin-top:6px;font-size:13px;color:#666;">
                    <span>Confidence: <strong style="color:${confColor};">${scan.confidence}%</strong></span>
                    <span>Model: ${scan.source || 'ML'}</span>
                </div>
                <div style="font-size:11px;color:#aaa;margin-top:4px;">${date}</div>
            </div>
        `;
        historyList.appendChild(div);
    });
}

function deleteHistoryEntry(idx) {
    APP_STATE.scanHistory.splice(idx, 1);
    localStorage.setItem('riceHistory', JSON.stringify(APP_STATE.scanHistory));
    loadHistoryDisplay();
}

function savePredictionToHistory(prediction) {
    if (!APP_STATE.autoSave || !prediction) return;

    const normalized = normalizePredictionPayload(prediction);
    const entry = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        rice_type: normalized.rice_type,
        confidence: normalized.displayConfidence,
        quality_grade: normalized.quality_grade,
        estimated_price: normalized.estimated_price || 0,
        source: normalized.source,
        image: APP_STATE.currentThumbnail || null
    };
    
    APP_STATE.scanHistory.unshift(entry);
    
    // Max 20 history items to ensure we don't exceed the 5MB browser quota
    if (APP_STATE.scanHistory.length > 20) {
        APP_STATE.scanHistory = APP_STATE.scanHistory.slice(0, 20);
    }
    
    try {
        localStorage.setItem('riceHistory', JSON.stringify(APP_STATE.scanHistory));
    } catch (e) {
        console.error('Failed to save history (possibly quota exceeded):', e);
        // Fallback: clear half the history and try again
        APP_STATE.scanHistory = APP_STATE.scanHistory.slice(0, 10);
        try {
            localStorage.setItem('riceHistory', JSON.stringify(APP_STATE.scanHistory));
        } catch(e2) {
            console.error('Still failed:', e2);
        }
    }
    
    // Immediately update UI to show new history
    if (typeof loadHistoryDisplay === 'function') loadHistoryDisplay();
    if (typeof updateStatistics === 'function') updateStatistics();
    
    console.log('[HISTORY] Saved');
}

function loadHistory() {
    try {
        const saved = localStorage.getItem('riceHistory');
        if (saved) {
            const parsed = JSON.parse(saved);
            APP_STATE.scanHistory = Array.isArray(parsed)
                ? parsed.map((item) => normalizeHistoryEntry(item))
                : [];
        }
        console.log('[HISTORY] Loaded:', APP_STATE.scanHistory.length);
    } catch (error) {
        console.error('[HISTORY ERROR]:', error);
        APP_STATE.scanHistory = [];
    }
}

// ============================================================================
// STATISTICS TAB
// ============================================================================

function setupStatisticsTab() {
    console.log('[STATS] Setting up');
    
    const refreshBtn = document.getElementById('refreshStatsBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', updateStatistics);
    }
    
    updateStatistics();
}

function updateStatistics() {
    console.log('[STATS] Updating');

    const history = Array.isArray(APP_STATE.scanHistory)
        ? APP_STATE.scanHistory.map((item) => normalizeHistoryEntry(item))
        : [];
    APP_STATE.scanHistory = history;
    localStorage.setItem('riceHistory', JSON.stringify(history));
    const total = history.length;
    const avg = total > 0 
        ? Math.round(history.reduce((s, h) => s + (Number(h.confidence) || 0), 0) / total)
        : 0;
    const premiumCount = history.filter(item => (item.quality_grade || '').toLowerCase() === 'premium').length;
    const premiumPercent = total > 0 ? Math.round((premiumCount / total) * 100) : 0;
    const riceCounts = history.reduce((acc, item) => {
        const key = item.rice_type || 'Unknown';
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {});
    const mostCommon = total > 0
        ? Object.entries(riceCounts).sort((a, b) => b[1] - a[1])[0][0]
        : '-';
    
    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };
    
    setText('statTotalScans', total);
    setText('statAvgConfidence', avg + '%');
    setText('statPremiumPercent', premiumPercent + '%');
    setText('statMostCommon', mostCommon);

    renderStatisticsCharts(history);
}

// ============================================================================
// SETTINGS TAB
// ============================================================================

function setupSettingsTab() {
    console.log('[SETTINGS] Setting up');
    
    const resetBtn = document.getElementById('resetSettingsBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            if (confirm('Reset all settings?')) {
                APP_STATE.autoSave = true;
                localStorage.clear();
                alert('Settings reset');
                location.reload();
            }
        });
    }
    
    const autoSaveCheckbox = document.getElementById('autoSave');
    if (autoSaveCheckbox) {
        autoSaveCheckbox.checked = APP_STATE.autoSave;
        autoSaveCheckbox.addEventListener('change', (e) => {
            APP_STATE.autoSave = e.target.checked;
            localStorage.setItem('autoSave', e.target.checked);
        });
    }

    document.querySelectorAll('input[name="modelSelect"]').forEach((radio) => {
        radio.checked = radio.value === APP_STATE.selectedModel;
        radio.addEventListener('change', (e) => {
            APP_STATE.selectedModel = e.target.value;
            localStorage.setItem('selectedModel', e.target.value);
        });
    });

    const exportAllBtn = document.getElementById('exportAllBtn');
    if (exportAllBtn) {
        exportAllBtn.addEventListener('click', exportHistoryCsv);
    }

    const exportPdfBtn = document.getElementById('exportPdfBtn');
    if (exportPdfBtn) {
        exportPdfBtn.addEventListener('click', exportStatisticsPdf);
    }

    const reportBtn = document.getElementById('generateReportBtn');
    if (reportBtn) {
        reportBtn.addEventListener('click', generateFullReport);
    }

    // Theme switching logic
    document.querySelectorAll('input[name="themeSelect"]').forEach((radio) => {
        radio.checked = radio.value === APP_STATE.theme;
        radio.addEventListener('change', (e) => {
            setTheme(e.target.value);
        });
    });
}

function generateFullReport() {
    const history = APP_STATE.scanHistory;
    if (!history || history.length === 0) {
        alert('No scan history available to generate a report. Please scan some rice images first.');
        return;
    }

    const total = history.length;
    const avg = total > 0
        ? Math.round(history.reduce((s, h) => s + (Number(h.confidence) || 0), 0) / total)
        : 0;
    const premiumCount = history.filter(item => (item.quality_grade || '').toLowerCase() === 'premium').length;
    const premiumPercent = total > 0 ? Math.round((premiumCount / total) * 100) : 0;
    const riceCounts = history.reduce((acc, item) => {
        const key = item.rice_type || 'Unknown';
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {});
    const mostCommon = total > 0
        ? Object.entries(riceCounts).sort((a, b) => b[1] - a[1])[0][0]
        : '-';

    const timestamp = new Date().toLocaleString();
    const tableRows = history.slice(0, 50).map((item, idx) => `
        <tr style="background:${idx % 2 === 0 ? '#f9fafb' : '#fff'}">
            <td style="padding:8px 12px;border-bottom:1px solid #eee;">${idx + 1}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;font-weight:600;">${item.rice_type || '-'}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;">${item.confidence || 0}%</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;">${item.quality_grade || '-'}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;">${item.source || 'ML'}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;font-size:12px;color:#888;">${new Date(item.timestamp).toLocaleString()}</td>
        </tr>`).join('');

    const varietyRows = Object.entries(riceCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([type, count]) => `
        <tr>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;font-weight:600;">${type}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;">${count}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;">${Math.round((count / total) * 100)}%</td>
        </tr>`).join('');

    const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Rice Classification Full Report</title>
<style>
  body{font-family:'Segoe UI',sans-serif;max-width:900px;margin:40px auto;padding:20px;color:#222;}
  h1{color:#1b4332;text-align:center;margin-bottom:4px;font-size:28px;}
  h2{color:#2d6a4f;font-size:18px;margin:28px 0 12px;border-bottom:2px solid #e8f5e9;padding-bottom:8px;}
  .subtitle{text-align:center;color:#888;font-size:13px;margin-bottom:28px;}
  .summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px;}
  .summary-card{background:#f9fafb;border-radius:10px;padding:16px;text-align:center;border-left:4px solid #4CAF50;}
  .summary-card .val{font-size:28px;font-weight:700;color:#1b4332;}
  .summary-card .lbl{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-top:4px;}
  table{width:100%;border-collapse:collapse;font-size:14px;}
  thead{background:linear-gradient(135deg,#1b4332,#2d6a4f);color:white;}
  th{padding:10px 12px;text-align:left;font-weight:600;}
  footer{text-align:center;color:#aaa;font-size:12px;margin-top:40px;border-top:1px solid #eee;padding-top:16px;}
</style></head><body>
<h1>🌾 Rice Classification Full Report</h1>
<p class="subtitle">Generated on ${timestamp} &bull; ${total} total scans</p>
<div class="summary-grid">
  <div class="summary-card"><div class="val">${total}</div><div class="lbl">Total Scans</div></div>
  <div class="summary-card"><div class="val">${avg}%</div><div class="lbl">Avg Confidence</div></div>
  <div class="summary-card"><div class="val">${premiumPercent}%</div><div class="lbl">Premium Grade</div></div>
  <div class="summary-card"><div class="val">${mostCommon}</div><div class="lbl">Most Common</div></div>
</div>
<h2>📊 Variety Distribution</h2>
<table><thead><tr><th>Rice Type</th><th>Count</th><th>Percentage</th></tr></thead><tbody>${varietyRows}</tbody></table>
<h2>📋 Scan History (last 50)</h2>
<table><thead><tr><th>#</th><th>Rice Type</th><th>Confidence</th><th>Grade</th><th>Model</th><th>Date</th></tr></thead><tbody>${tableRows}</tbody></table>
<footer>Rice Classification AI &bull; Powered by CNN + KNN Models</footer>
</body></html>`;

    downloadBlob(html, `rice_full_report_${Date.now()}.html`, 'text/html');
    console.log('[REPORT] Full report generated');
}

function setTheme(theme) {
    APP_STATE.theme = theme;
    localStorage.setItem('appTheme', theme);
    if (theme === 'dark') {
        document.body.classList.add('dark-theme');
    } else {
        document.body.classList.remove('dark-theme');
    }
}

// ============================================================================
// API CALLS
// ============================================================================

async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        const badge = document.getElementById('modelStatus')?.querySelector('.status-badge');
        if (badge) {
            badge.textContent = data.model_loaded ? '✅ Model Ready' : '⚠️ Demo Mode';
            badge.className = 'status-badge ' + (data.model_loaded ? 'ready' : '');
        }
        console.log('[API] Model:', data.active_model);
    } catch (error) {
        console.error('[API ERROR]:', error);
    }
}

async function loadRiceInfo() {
    try {
        const response = await fetch(`${API_BASE}/rice-info`);
        const data = await response.json();
        APP_STATE.riceInfo = data.details || data || {};
        console.log('[API] Rice info loaded');
    } catch (error) {
        console.error('[API ERROR]:', error);
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = show ? 'flex' : 'none';
    }
}

function loadSavedSettings() {
    try {
        const savedAutoSave = localStorage.getItem('autoSave');
        const savedModel = localStorage.getItem('selectedModel');
        const savedTheme = localStorage.getItem('appTheme') || 'light';

        if (savedAutoSave !== null) {
            APP_STATE.autoSave = savedAutoSave === 'true';
        }

        if (savedModel) {
            APP_STATE.selectedModel = savedModel;
        }

        setTheme(savedTheme);
        const themeRadio = document.querySelector(`input[name="themeSelect"][value="${savedTheme}"]`);
        if (themeRadio) {
            themeRadio.checked = true;
        }

    } catch (error) {
        console.error('[SETTINGS ERROR]:', error);
    }
}

function normalizeRiceInfo(info = {}, riceType = '') {
    const priceMin = Number(info.price_min || info.priceMin || 0);
    const priceMax = Number(info.price_max || info.priceMax || 0);
    const averagePrice = Number(info.price || info.estimated_price || info.estimatedPrice || (
        priceMin > 0 && priceMax > 0 ? Math.round((priceMin + priceMax) / 2) : 0
    ));

    return {
        ...info,
        rice_type: riceType || info.rice_type || info.variety || 'Unknown',
        flavor: info.flavor || info.aroma || info.characteristics?.Aroma || '-',
        cooking_time: info.cooking_time || info.characteristics?.['Cooking Time'] || '-',
        grain_shape: info.grain_shape || info.grainShape || '-',
        quality_grade: info.quality_grade || (info.grade_premium ? 'Premium' : 'Standard'),
        price: averagePrice || '-'
    };
}

function setupActionButtons() {
    const retryButton = document.getElementById('retryButton');
    if (retryButton) {
        retryButton.addEventListener('click', () => {
            const errorSection = document.getElementById('errorSection');
            if (errorSection) {
                errorSection.style.display = 'none';
            }
            analyzeImage();
        });
    }
}

function getChartCanvas(id) {
    const canvas = document.getElementById(id);
    return canvas && typeof canvas.getContext === 'function' ? canvas : null;
}

function destroyChart(key) {
    if (CHART_STATE[key]) {
        CHART_STATE[key].destroy();
        CHART_STATE[key] = null;
    }
    // Reset canvas size to prevent Chart.js from inheriting inflated dimensions
    const canvasIds = { riceType: 'riceTypeChart', quality: 'qualityChart', confidence: 'confidenceChart', scans: 'scansChart' };
    const canvasId = canvasIds[key];
    if (canvasId) {
        const canvas = document.getElementById(canvasId);
        if (canvas) {
            canvas.style.width = '';
            canvas.style.height = '';
            canvas.removeAttribute('width');
            canvas.removeAttribute('height');
        }
    }
}

function renderStatisticsCharts(history) {
    if (typeof Chart === 'undefined') {
        console.warn('[STATS] Chart.js not available');
        return;
    }

    const normalizedHistory = Array.isArray(history)
        ? history.map((item) => normalizeHistoryEntry(item))
        : [];
    const chartHistory = normalizedHistory
        .filter((item) => item.rice_type && item.rice_type !== 'Unknown')
        .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    const riceCounts = chartHistory.reduce((acc, item) => {
        const key = item.rice_type || 'Unknown';
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {});

    const qualityCounts = chartHistory.reduce((acc, item) => {
        const key = item.quality_grade || 'Unknown';
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {});

    const confidenceLabels = chartHistory.map((item) => {
        const date = new Date(item.timestamp || Date.now());
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    });
    const confidenceData = chartHistory.map((item) => Number(item.confidence) || 0);

    const monthCounts = chartHistory.reduce((acc, item) => {
        const date = new Date(item.timestamp || Date.now());
        const label = date.toLocaleString('en-US', { month: 'short', year: 'numeric' });
        acc[label] = (acc[label] || 0) + 1;
        return acc;
    }, {});

    const orderedRiceLabels = KNOWN_RICE_TYPES.filter((type) => riceCounts[type]);
    const orderedQualityLabels = KNOWN_QUALITY_GRADES.filter((grade) => qualityCounts[grade]);

    // ── Shared readable typography for all charts ──────────────────────────
    const LABEL_COLOR  = '#1a2e1a';   // near-black dark green — high contrast
    const TICK_COLOR   = '#2d4a2e';   // dark green for axis ticks
    const GRID_COLOR   = 'rgba(45,74,46,0.12)';
    const FONT_BASE    = { family: "'Inter','Poppins',sans-serif" };
    const TICK_FONT    = { ...FONT_BASE, size: 13, weight: '600' };
    const LEGEND_FONT  = { ...FONT_BASE, size: 13, weight: '700' };
    const TITLE_FONT   = { ...FONT_BASE, size: 14, weight: '700' };

    // Shared base options — Chart.js 3.x compatible
    const BASE = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        devicePixelRatio: window.devicePixelRatio || 2,
        layout: { padding: { top: 10, bottom: 10, left: 8, right: 8 } },
    };

    // Reusable axis config for Chart.js 3.x (borderColor lives inside grid, not border)
    const xAxis3 = {
        ticks: { color: TICK_COLOR, font: TICK_FONT },
        grid:  { color: GRID_COLOR, borderColor: TICK_COLOR },
    };
    const yAxis3 = {
        beginAtZero: true,
        ticks: { color: TICK_COLOR, font: TICK_FONT, precision: 0 },
        grid:  { color: GRID_COLOR, borderColor: TICK_COLOR },
    };

    // ── 1. Doughnut — Rice Type Distribution ──────────────────────────────
    const riceCanvas = getChartCanvas('riceTypeChart');
    if (riceCanvas) {
        destroyChart('riceType');
        CHART_STATE.riceType = new Chart(riceCanvas.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: orderedRiceLabels,
                datasets: [{
                    data: orderedRiceLabels.map((l) => riceCounts[l]),
                    backgroundColor: ['#1b4332', '#2d6a4f', '#40916c', '#52b788', '#74c69d'],
                    borderColor: '#ffffff',
                    borderWidth: 3,
                    hoverOffset: 8,
                }]
            },
            options: {
                ...BASE,
                cutout: '55%',
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: LABEL_COLOR,
                            font: LEGEND_FONT,
                            padding: 16,
                            usePointStyle: true,
                            pointStyleWidth: 12,
                        },
                    },
                    tooltip: {
                        titleFont: TITLE_FONT,
                        bodyFont: { ...FONT_BASE, size: 13 },
                        callbacks: {
                            label: (ctx) => {
                                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                                const pct = total ? Math.round((ctx.parsed / total) * 100) : 0;
                                return ` ${ctx.label}: ${ctx.parsed} scan${ctx.parsed !== 1 ? 's' : ''} (${pct}%)`;
                            },
                        },
                    },
                    // Inline percentage labels on each slice
                    datalabels: undefined,  // not loaded; use tooltip instead
                },
            },
        });
    }

    // ── 2. Bar — Quality Grade Distribution ───────────────────────────────
    const qualityCanvas = getChartCanvas('qualityChart');
    if (qualityCanvas) {
        destroyChart('quality');
        CHART_STATE.quality = new Chart(qualityCanvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: orderedQualityLabels,
                datasets: [{
                    label: 'Scans',
                    data: orderedQualityLabels.map((l) => qualityCounts[l]),
                    backgroundColor: ['#1b4332', '#e6a356', '#e76f51'],
                    borderColor:     ['#0d2218', '#b87a2e', '#c0442a'],
                    borderWidth: 2,
                    borderRadius: 6,
                    borderSkipped: false,
                }]
            },
            options: {
                ...BASE,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        titleFont: TITLE_FONT,
                        bodyFont: { ...FONT_BASE, size: 13 },
                    },
                },
                scales: {
                    x: { ...xAxis3 },
                    y: { ...yAxis3 },
                },
            },
        });
    }

    // ── 3. Line — Confidence Score Trend ──────────────────────────────────
    const confidenceCanvas = getChartCanvas('confidenceChart');
    if (confidenceCanvas) {
        destroyChart('confidence');
        CHART_STATE.confidence = new Chart(confidenceCanvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: confidenceLabels,
                datasets: [{
                    label: 'Confidence %',
                    data: confidenceData,
                    borderColor: '#1b4332',
                    backgroundColor: 'rgba(27,67,50,0.12)',
                    borderWidth: 3,
                    pointBackgroundColor: '#1b4332',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.35,
                    fill: true,
                }]
            },
            options: {
                ...BASE,
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: LABEL_COLOR, font: LEGEND_FONT, usePointStyle: true },
                    },
                    tooltip: {
                        titleFont: TITLE_FONT,
                        bodyFont: { ...FONT_BASE, size: 13 },
                        callbacks: {
                            label: (ctx) => ` ${ctx.parsed.y}%`,
                        },
                    },
                },
                scales: {
                    x: {
                        ...xAxis3,
                        ticks: { ...xAxis3.ticks, maxRotation: 45, autoSkip: true, maxTicksLimit: 10 },
                    },
                    y: {
                        ...yAxis3,
                        max: 100,
                        ticks: { ...yAxis3.ticks, callback: (v) => v + '%' },
                    },
                },
            },
        });
    }

    // ── 4. Bar — Monthly Scans ─────────────────────────────────────────────
    const scansCanvas = getChartCanvas('scansChart');
    if (scansCanvas) {
        destroyChart('scans');
        CHART_STATE.scans = new Chart(scansCanvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: Object.keys(monthCounts),
                datasets: [{
                    label: 'Monthly Scans',
                    data: Object.values(monthCounts),
                    backgroundColor: '#40916c',
                    borderColor: '#1b4332',
                    borderWidth: 2,
                    borderRadius: 6,
                    borderSkipped: false,
                }]
            },
            options: {
                ...BASE,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        titleFont: TITLE_FONT,
                        bodyFont: { ...FONT_BASE, size: 13 },
                    },
                },
                scales: {
                    x: { ...xAxis3, ticks: { ...xAxis3.ticks, maxRotation: 45 } },
                    y: { ...yAxis3 },
                },
            },
        });
    }
}

function exportHistoryCsv() {
    if (!APP_STATE.scanHistory.length) {
        alert('No scan history available to export.');
        return;
    }

    const header = ['Timestamp', 'Rice Type', 'Confidence', 'Quality Grade', 'Estimated Price'];
    const rows = APP_STATE.scanHistory.map((item) => [
        item.timestamp || '',
        item.rice_type || '',
        item.confidence || '',
        item.quality_grade || '',
        item.estimated_price || ''
    ]);

    const csv = [header, ...rows]
        .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(','))
        .join('\n');

    downloadBlob(csv, 'rice_scan_history.csv', 'text/csv');
}

async function exportStatisticsPdf() {
    if (typeof window.jspdf === 'undefined' || typeof html2canvas === 'undefined') {
        alert('PDF export library is not available.');
        return;
    }

    const statisticsContainer = document.querySelector('.statistics-container');
    if (!statisticsContainer) {
        alert('Statistics section not found.');
        return;
    }

    const canvas = await html2canvas(statisticsContainer, { scale: 2 });
    const imageData = canvas.toDataURL('image/png');
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF('p', 'mm', 'a4');
    const width = 190;
    const height = (canvas.height * width) / canvas.width;

    pdf.addImage(imageData, 'PNG', 10, 10, width, height);
    pdf.save('rice_statistics.pdf');
}

function resetApp() {
    clearPreview();
    APP_STATE.lastPrediction = null;

    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const banner = document.getElementById('notRiceBanner');
    const warningBanner = document.getElementById('detectionWarningBanner');
    if (resultsSection) resultsSection.style.display = 'none';
    if (errorSection) errorSection.style.display = 'none';
    if (banner) banner.style.display = 'none';
    if (warningBanner) warningBanner.style.display = 'none';

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function resetToUpload() {
    // Called from the not-rice banner — clears everything and lets user pick a new image
    const banner = document.getElementById('notRiceBanner');
    if (banner) banner.style.display = 'none';
    clearPreview();
    APP_STATE.lastPrediction = null;
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function downloadResult() {
    if (!APP_STATE.lastPrediction) {
        alert('No analysis result available to download.');
        return;
    }

    downloadBlob(
        JSON.stringify(APP_STATE.lastPrediction, null, 2),
        `rice_result_${Date.now()}.json`,
        'application/json'
    );
}

function downloadReport() {
    const p = APP_STATE.lastPrediction;
    if (!p) { alert('No analysis result available.'); return; }

    const normalized = normalizePredictionPayload(p);
    const riceType = normalized.finalType || 'Unknown';
    const confidence = normalized.displayConfidence || 0;
    const grade = normalized.gradeLabel || '-';
    const price = (normalized.priceMin && normalized.priceMax)
        ? `Rs ${normalized.priceMin} - ${normalized.priceMax} / kg`
        : '-';
    const timestamp = new Date().toLocaleString();
    const imageDataUrl = APP_STATE.currentImage || '';

    const imgTag = imageDataUrl
        ? `<img src="${imageDataUrl}" style="max-width:320px;border-radius:10px;border:3px solid #4CAF50;display:block;margin:0 auto 20px;">`
        : '';

    const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Rice Classification Report</title>
<style>
  body{font-family:'Segoe UI',sans-serif;max-width:700px;margin:40px auto;padding:20px;color:#222;}
  h1{color:#1b4332;text-align:center;margin-bottom:4px;}
  .subtitle{text-align:center;color:#888;font-size:13px;margin-bottom:28px;}
  .card{background:#f9fafb;border-radius:12px;padding:20px 24px;margin-bottom:18px;border:1px solid #e0e0e0;}
  .row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eee;font-size:15px;}
  .row:last-child{border-bottom:none;}
  .label{color:#666;font-weight:600;}
  .value{color:#1b4332;font-weight:700;}
  .conf-bar{background:#e0e0e0;border-radius:20px;height:12px;margin-top:6px;overflow:hidden;}
  .conf-fill{height:100%;background:linear-gradient(90deg,#4CAF50,#81C784);border-radius:20px;width:${confidence}%;}
  footer{text-align:center;color:#aaa;font-size:12px;margin-top:30px;}
</style></head><body>
<h1>🌾 Rice Classification Report</h1>
<p class="subtitle">Generated on ${timestamp}</p>
${imgTag}
<div class="card">
  <div class="row"><span class="label">Detected Variety</span><span class="value">${riceType}</span></div>
  <div class="row"><span class="label">AI Confidence</span>
    <span class="value">${confidence}%
      <div class="conf-bar"><div class="conf-fill"></div></div>
    </span>
  </div>
  <div class="row"><span class="label">Quality Grade</span><span class="value">${grade}</span></div>
  <div class="row"><span class="label">Price Range</span><span class="value">${price}</span></div>
</div>
<footer>Rice Classification AI &bull; Powered by CNN + KNN Models</footer>
</body></html>`;

    downloadBlob(html, `rice_report_${riceType}_${Date.now()}.html`, 'text/html');
}

function downloadBlob(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function isSupportedImageFile(file) {
    return Boolean(getSupportedImageMimeType(file));
}

function getSupportedImageMimeType(fileOrMimeType) {
    const mimeType = typeof fileOrMimeType === 'string'
        ? fileOrMimeType.toLowerCase()
        : fileOrMimeType && typeof fileOrMimeType.type === 'string'
            ? fileOrMimeType.type.toLowerCase()
            : '';

    if (!mimeType.startsWith('image/')) {
        return null;
    }

    return ['image/jpeg', 'image/png', 'image/webp'].includes(mimeType) ? mimeType : null;
}

function getMimeTypeFromDataUrl(dataUrl) {
    if (typeof dataUrl !== 'string') {
        return null;
    }

    const match = dataUrl.match(/^data:(image\/[a-z0-9.+-]+);base64,/i);
    return match ? getSupportedImageMimeType(match[1]) : null;
}

function extractBase64Payload(dataUrl) {
    if (typeof dataUrl !== 'string' || !dataUrl.includes(',')) {
        return dataUrl;
    }

    return dataUrl.split(',')[1];
}

async function handleFileUpload() {
    const fileInput = document.getElementById('fileUploadInput');
    const file = fileInput?.files?.[0];
    const mimeType = getSupportedImageMimeType(file);

    if (!file) {
        showFileStatus('error', 'Please select an image file.');
        return;
    }

    if (!mimeType) {
        showFileStatus('error', 'Please select a valid JPEG, PNG, or WEBP image.');
        return;
    }

    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showFileStatus('error', `File too large. Maximum size: 16MB, your file: ${(file.size / (1024 * 1024)).toFixed(2)}MB`);
        return;
    }

    showFileStatus('loading', 'Loading preview...');

    try {
        const reader = new FileReader();
        reader.onload = (event) => {
            const previewImg = document.getElementById('filePreviewImage');
            if (previewImg) {
                previewImg.src = event.target.result;
            }

            const fileName = document.getElementById('fileName');
            const fileSize = document.getElementById('fileSize');
            const previewSection = document.getElementById('filePreviewSection');

            if (fileName) fileName.textContent = file.name;
            if (fileSize) fileSize.textContent = (file.size / 1024).toFixed(2) + ' KB';
            if (previewSection) previewSection.style.display = 'block';

            APP_STATE.selectedFile = file;
            APP_STATE.currentImageMimeType = mimeType;

            showFileStatus('success', 'File loaded. Click "Confirm Analysis" to process.');
        };

        reader.onerror = () => {
            showFileStatus('error', 'Error reading file.');
        };

        reader.readAsDataURL(file);
    } catch (error) {
        showFileStatus('error', error.message || 'Unable to load the image preview.');
    }
}

async function confirmFileAnalysis() {
    const file = APP_STATE.selectedFile;
    const mimeType = getSupportedImageMimeType(file) || getSupportedImageMimeType(APP_STATE.currentImageMimeType);

    if (!file) {
        showFileStatus('error', 'No file selected.');
        return;
    }

    if (!mimeType) {
        showFileStatus('error', 'Please select a valid JPEG, PNG, or WEBP image.');
        return;
    }

    if (APP_STATE.isProcessing) {
        alert('Analysis already in progress.');
        return;
    }

    APP_STATE.isProcessing = true;
    showLoading(true);
    showFileStatus('loading', 'Analyzing image...');

    try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('image_mime_type', mimeType);
        formData.append('stable', 'true');

        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let errorMessage = `Server error: ${response.status}`;
            try {
                const errorData = await response.json();
                console.error('[FILE UPLOAD] API error response:', errorData);
                errorMessage = errorData.error || errorData.message || errorMessage;
            } catch (parseError) {
                console.error('[FILE UPLOAD] Failed to parse error response:', parseError);
            }
            throw new Error(errorMessage);
        }

        const result = await response.json();
        console.log('[FILE UPLOAD] Result received:', result);

        if (!result.success) {
            throw new Error(result.error || 'Analysis failed');
        }

        APP_STATE.lastPrediction = result;
        displayResults(result);
        savePredictionToHistory(result);
        showFileStatus('success', 'Analysis complete.');
    } catch (error) {
        console.error('[FILE UPLOAD ERROR]:', error);
        showFileStatus('error', error.message || 'Image analysis failed.');
        alert(`Error: ${error.message || 'Image analysis failed.'}`);
    } finally {
        APP_STATE.isProcessing = false;
        showLoading(false);
    }
}

window.addEventListener('beforeunload', () => {
    // Stop any active camera stream
    if (CAMERA_STATE.stream) {
        CAMERA_STATE.stream.getTracks().forEach(track => track.stop());
    }
});

console.log('✅ App script loaded');
