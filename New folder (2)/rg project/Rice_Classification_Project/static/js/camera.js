/**
 * Smart Mobile Rice Quality Scanner - Camera Functionality
 * Handles mobile and desktop camera capture with canvas
 */

// ============================================================================
// CAMERA STATE
// ============================================================================

const CAMERA_STATE = {
    stream: null,
    isActive: false,
    canvas: null,
    ctx: null,
    video: null
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('📹 Camera module loading...');
    setupCameraElements();
    setupCameraEventListeners();
    checkCamera();
    console.log('✓ Camera module ready');
});

/**
 * Check camera support on startup
 */
async function checkCamera() {
    try {
        const hasCamera = await doesDeviceHaveCamera();
        if (hasCamera) {
            console.log('✓ Camera device detected');
        } else {
            console.warn('⚠️ No camera device found');
            const startBtn = document.getElementById('startCamera');
            if (startBtn) {
                startBtn.title = 'No camera device found on this device';
                startBtn.disabled = true;
                startBtn.style.opacity = '0.5';
            }
        }
        
        // Log browser info
        console.log(`Browser: ${navigator.userAgent.substring(0, 100)}`);
        console.log(`Protocol: ${location.protocol}`);
    } catch (error) {
        console.warn('Could not check camera:', error);
    }
}

/**
 * Setup camera DOM elements
 */
function setupCameraElements() {
    CAMERA_STATE.video = document.getElementById('cameraVideo');
    CAMERA_STATE.canvas = document.getElementById('canvas');
    CAMERA_STATE.ctx = CAMERA_STATE.canvas.getContext('2d');
}

/**
 * Setup camera event listeners
 */
function setupCameraEventListeners() {
    const startBtn = document.getElementById('startCamera');
    const stopBtn = document.getElementById('stopCamera');
    const captureBtn = document.getElementById('captureButton');
    
    if (startBtn) startBtn.addEventListener('click', startCamera);
    if (stopBtn) stopBtn.addEventListener('click', stopCamera);
    if (captureBtn) captureBtn.addEventListener('click', capturePhoto);
}

// ============================================================================
// CAMERA FUNCTIONS
// ============================================================================

/**
 * Start camera stream - SIMPLIFIED
 */
async function startCamera() {
    try {
        console.log('🎥 Starting camera...');
        
        const video = document.getElementById('cameraVideo');
        if (!video) {
            showError('Video element not found');
            return;
        }
        
        // Get camera stream
        let stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        }).catch(async () => {
            // Fallback: try any camera
            return await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        });
        
        CAMERA_STATE.stream = stream;
        video.srcObject = stream;
        CAMERA_STATE.isActive = true;
        
        // Update UI
        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('stopCamera').style.display = 'inline-block';
        document.getElementById('captureButton').style.display = 'inline-block';
        
        console.log('✓ Camera started');
        
    } catch (error) {
        console.error('❌ Camera failed:', error.name);
        let msg = '❌ Camera Failed\n\n';
        
        if (error.name === 'NotAllowedError') {
            msg += '1. Click camera icon in address bar\n2. Click "Allow"\n3. Refresh page';
        } else if (error.name === 'NotFoundError') {
            msg += 'Device may not have camera';
        } else {
            msg += error.message;
        }
        
        showError(msg);
    }
}

/**
 * Stop camera stream
 */
function stopCamera() {
    try {
        if (CAMERA_STATE.stream) {
            const tracks = CAMERA_STATE.stream.getTracks();
            tracks.forEach(track => {
                track.stop();
                console.log('✓ Camera track stopped:', track.kind);
            });
            CAMERA_STATE.stream = null;
        }
        
        CAMERA_STATE.isActive = false;
        if (CAMERA_STATE.video) {
            CAMERA_STATE.video.srcObject = null;
        }
        
        // Update UI
        const startBtn = document.getElementById('startCamera');
        const stopBtn = document.getElementById('stopCamera');
        const captureBtn = document.getElementById('captureButton');
        
        if (startBtn) startBtn.style.display = 'inline-block';
        if (stopBtn) stopBtn.style.display = 'none';
        if (captureBtn) captureBtn.style.display = 'none';
        
        console.log('✓ Camera stopped');
    } catch (error) {
        console.error('Error stopping camera:', error);
    }
}

/**
 * Capture photo from camera - SIMPLIFIED
 */
async function capturePhoto() {
    if (!CAMERA_STATE.isActive) {
        showError('Camera not active');
        return;
    }
    
    try {
        const video = CAMERA_STATE.video;
        const canvas = CAMERA_STATE.canvas;
        const ctx = CAMERA_STATE.ctx;
        
        // Wait if needed
        if (video.videoWidth === 0) {
            await new Promise(r => setTimeout(r, 1000));
        }
        
        if (video.videoWidth === 0) {
            showError('Camera still loading, wait and try again');
            return;
        }
        
        // Capture
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        
        const imageData = canvas.toDataURL('image/jpeg', 0.9);
        
        if (imageData.length < 500) {
            showError('Image too small, move camera closer');
            return;
        }
        
        APP_STATE.currentImage = imageData;
        APP_STATE.currentImageFile = null;
        
        stopCamera();
        displayPreview(imageData);
        
        console.log('✓ Photo captured');
        
    } catch (error) {
        showError('Capture failed: ' + error.message);
    }
}

/**
 * Wait for video to have valid dimensions
 */
function waitForVideoReady(msTimeout = 3000) {
    return new Promise((resolve) => {
        const startTime = Date.now();
        
        const checkReady = () => {
            if (CAMERA_STATE.video.videoWidth > 0 && CAMERA_STATE.video.videoHeight > 0) {
                console.log('✓ Video ready with dimensions:', CAMERA_STATE.video.videoWidth, 'x', CAMERA_STATE.video.videoHeight);
                resolve(true);
                return;
            }
            
            if (Date.now() - startTime > msTimeout) {
                console.warn('⏱️ Video ready timeout after', msTimeout, 'ms');
                resolve(false);
                return;
            }
            
            // Check every 100ms
            setTimeout(checkReady, 100);
        };
        
        checkReady();
    });
}

/**
 * Wait for video to have data
 */
function waitForVideoData(msTimeout = 2000) {
    return new Promise((resolve) => {
        const startTime = Date.now();
        
        const checkData = () => {
            if (CAMERA_STATE.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
                console.log('✓ Video has data, readyState:', CAMERA_STATE.video.readyState);
                resolve(true);
                return;
            }
            
            if (Date.now() - startTime > msTimeout) {
                console.warn('⏱️ Video data timeout after', msTimeout, 'ms');
                resolve(false);
                return;
            }
            
            setTimeout(checkData, 100);
        };
        
        checkData();
    });
}

// ============================================================================
// CAMERA PERMISSIONS
// ============================================================================

/**
 * Check camera permissions
 */
async function checkCameraPermission() {
    try {
        if (navigator.permissions && navigator.permissions.query) {
            const permission = await navigator.permissions.query({ name: 'camera' });
            return permission.state;
        }
    } catch (error) {
        console.warn('Could not check camera permission:', error);
    }
    return 'unknown';
}

/**
 * Request camera permission
 */
async function requestCameraPermission() {
    const permission = await checkCameraPermission();
    
    if (permission === 'denied') {
        showError('Camera permission is denied. Please enable it in your browser settings.');
        return false;
    }
    
    return true;
}

// ============================================================================
// CAMERA HELPERS
// ============================================================================

/**
 * Check if device has camera
 */
async function doesDeviceHaveCamera() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        return videoDevices.length > 0;
    } catch (error) {
        console.warn('Could not enumerate devices:', error);
        return true; // Assume it might have a camera
    }
}

/**
 * Get available cameras
 */
async function getAvailableCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices.filter(device => device.kind === 'videoinput');
    } catch (error) {
        console.error('Error getting cameras:', error);
        return [];
    }
}

// ============================================================================
// BROWSER COMPATIBILITY
// ============================================================================

/**
 * Check browser camera support
 */
function isCameraSupported() {
    return !!(
        navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia ||
        (navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
    );
}

/**
 * Get supported image formats
 */
function getSupportedImageFormats() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    const formats = {
        jpeg: canvas.toDataURL('image/jpeg').indexOf('image/jpeg') === 5,
        png: canvas.toDataURL('image/png').indexOf('image/png') === 5,
        webp: canvas.toDataURL('image/webp').indexOf('image/webp') === 5
    };
    
    return formats;
}

// ============================================================================
// MOBILE DETECTION
// ============================================================================

/**
 * Check if device is mobile
 */
function isMobileDevice() {
    const userAgent = navigator.userAgent.toLowerCase();
    return /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/.test(userAgent);
}

/**
 * Check if device orientation is portrait
 */
function isPortraitOrientation() {
    return window.innerHeight > window.innerWidth;
}

// ============================================================================
// FULLSCREEN CAMERA CAPTURE
// ============================================================================

/**
 * Enter fullscreen camera mode
 */
async function enterFullscreenCameraMode() {
    const videoElement = document.getElementById('cameraVideo');
    
    try {
        if (videoElement.requestFullscreen) {
            await videoElement.requestFullscreen();
        } else if (videoElement.webkitRequestFullscreen) {
            await videoElement.webkitRequestFullscreen();
        } else if (videoElement.mozRequestFullScreen) {
            await videoElement.mozRequestFullScreen();
        } else if (videoElement.msRequestFullscreen) {
            await videoElement.msRequestFullscreen();
        }
    } catch (error) {
        console.warn('Fullscreen not available:', error);
    }
}

// ============================================================================
// INITIALIZATION CHECK
// ============================================================================

// Check camera support on load
window.addEventListener('load', async () => {
    if (!isCameraSupported()) {
        console.warn('⚠ Camera is not supported on this browser');
        const cameraTab = document.querySelector('[data-tab="camera"]');
        if (cameraTab) {
            cameraTab.disabled = true;
            cameraTab.title = 'Camera not supported on this browser';
        }
    } else {
        console.log('✓ Camera is supported');
        
        // Check for available cameras
        const cameras = await getAvailableCameras();
        console.log(`Found ${cameras.length} camera(s)`);
    }
    
    // Log device info
    console.log(`Device: ${isMobileDevice() ? 'Mobile' : 'Desktop'}`);
    console.log(`Orientation: ${isPortraitOrientation() ? 'Portrait' : 'Landscape'}`);
    
    // Log supported formats
    const formats = getSupportedImageFormats();
    console.log('Supported formats:', formats);
});

// ============================================================================
// ERROR LOGGING
// ============================================================================

console.log('%c📷 Camera Module Loaded', 'color: #3498db; font-weight: bold;');
