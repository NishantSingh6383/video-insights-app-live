// Video Insights App - Main JavaScript

const API_BASE = '/api';

// Theme Management
function initTheme() {
    const saved = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(saved || (prefersDark ? 'dark' : 'light'));
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    const sun = document.getElementById('theme-icon-sun');
    const moon = document.getElementById('theme-icon-moon');
    if (sun && moon) {
        // Light mode: show sun, hide moon. Dark mode: hide sun, show moon.
        if (theme === 'dark') {
            sun.classList.add('hidden');
            moon.classList.remove('hidden');
        } else {
            sun.classList.remove('hidden');
            moon.classList.add('hidden');
        }
    }
    // Re-render the chart with theme-appropriate colors if visible
    if (state.lastAnalytics && !document.getElementById('analytics-section').classList.contains('hidden')) {
        renderScoresChart(state.lastAnalytics.technique_scores);
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(current === 'dark' ? 'light' : 'dark');
}

function themeColor(varName, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    return value || fallback;
}

// State
let state = {
    currentStep: 1,
    selectedFile: null,
    videoInfo: null,
    selectedTechnique: 'combined',
    selectedOutputType: 'dynamic',
    selectedSummaryLength: 'medium',
    summaryPercent: 20,
    techniques: [],
    lastResult: null,
    lastAnalytics: null,
    aiInsightsAvailable: false,
    config: {
        technique: 'combined',
        output_type: 'dynamic',
        frame_sample_rate: 5,
        max_frames: 2000,
        output_fps: 24,
        n_clusters: 15,
        motion_threshold: 5.0,
        summary_percent: 20,
    }
};

const SUMMARY_LENGTHS = {
    short: 10,
    medium: 20,
    long: 35,
};

// Technique icons (SVG paths)
const TECHNIQUE_ICONS = {
    motion: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>',
    color: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="13.5" cy="6.5" r="2.5"></circle><circle cx="19" cy="17" r="2.5"></circle><circle cx="6" cy="12" r="2.5"></circle><path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z"></path></svg>',
    event: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    object_detection: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>',
    combined: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>'
};

const PROCESSING_STEPS = {
    motion: [
        'Extracting frames from video...',
        'Computing optical flow between frames...',
        'Analyzing motion patterns...',
        'Clustering frames by motion intensity...',
        'Selecting representative key frames...',
        'Generating summary video...',
    ],
    color: [
        'Extracting frames from video...',
        'Computing color histograms...',
        'Analyzing color distribution...',
        'Clustering frames by visual similarity...',
        'Selecting diverse key frames...',
        'Generating summary video...',
    ],
    event: [
        'Extracting frames from video...',
        'Computing motion features...',
        'Detecting significant events...',
        'Identifying activity spikes...',
        'Extracting event segments...',
        'Generating summary video...',
    ],
    object_detection: [
        'Extracting frames from video...',
        'Analyzing frame contents...',
        'Detecting objects and entities...',
        'Scoring frame importance...',
        'Selecting frames with objects...',
        'Generating summary video...',
    ],
    combined: [
        'Extracting frames from video...',
        'Computing motion features...',
        'Analyzing color distribution...',
        'Detecting events...',
        'Computing combined scores...',
        'Clustering and selecting frames...',
        'Generating summary video...',
    ],
};

// Utility Functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function $(selector) {
    return document.querySelector(selector);
}

function $$(selector) {
    return document.querySelectorAll(selector);
}

// Step Management
function setStep(step) {
    state.currentStep = step;

    // Update step indicators
    $$('.step').forEach((el, index) => {
        const stepNum = index + 1;
        el.classList.remove('active', 'completed');
        if (stepNum === step) {
            el.classList.add('active');
        } else if (stepNum < step) {
            el.classList.add('completed');
            el.querySelector('.step-number').textContent = '✓';
        } else {
            el.querySelector('.step-number').textContent = stepNum;
        }
    });

    // Update step lines
    $$('.step-line').forEach((el, index) => {
        el.classList.toggle('active', index < step - 1);
    });

    // Show/hide sections
    $('#upload-section').classList.toggle('hidden', step !== 1);
    $('#configure-section').classList.toggle('hidden', step !== 2);
    $('#processing-section').classList.toggle('hidden', step !== 3);
    $('#results-section').classList.toggle('hidden', step !== 4);
}

// API Functions
async function fetchTechniques() {
    try {
        const response = await fetch(`${API_BASE}/techniques/`);
        state.techniques = await response.json();
        renderTechniques();
    } catch (error) {
        console.error('Failed to fetch techniques:', error);
    }
}

async function uploadVideo(file) {
    const formData = new FormData();
    formData.append('video', file);

    const progressContainer = $('#upload-progress');
    const progressFill = $('#progress-fill');
    const progressPercent = $('#progress-percent');
    const uploadBtn = $('#upload-btn');
    const errorDiv = $('#upload-error');

    progressContainer.classList.remove('hidden');
    errorDiv.classList.add('hidden');
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;margin-right:8px;"></span> Uploading...';

    try {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                progressFill.style.width = `${percent}%`;
                progressPercent.textContent = `${percent}%`;
            }
        });

        const response = await new Promise((resolve, reject) => {
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error(JSON.parse(xhr.responseText).error || 'Upload failed'));
                }
            };
            xhr.onerror = () => reject(new Error('Network error'));
            xhr.open('POST', `${API_BASE}/videos/upload/`);
            xhr.send(formData);
        });

        state.videoInfo = response;
        showConfigureStep();
    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.classList.remove('hidden');
    } finally {
        progressContainer.classList.add('hidden');
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            Upload Video
        `;
    }
}

async function summarizeVideo() {
    const errorDiv = $('#config-error');
    errorDiv.classList.add('hidden');

    // Calculate summary percent
    let summaryPercent = state.summaryPercent;
    if (state.selectedSummaryLength === 'custom') {
        summaryPercent = parseInt($('#custom-percent').value) || 20;
    }

    // Build request config
    // frame_sample_rate=2 means extract every 2nd frame (50% of original)
    const config = {
        technique: state.selectedTechnique,
        output_type: state.selectedOutputType,
        frame_sample_rate: parseInt($('#frame-sample-rate').value) || 2,
        max_frames: parseInt($('#max-frames').value) || 3000,
        output_fps: parseInt($('#output-fps').value) || 24,
        summary_percent: summaryPercent,
    };

    // Add technique-specific params
    const technique = state.techniques.find(t => t.id === state.selectedTechnique);
    if (technique) {
        technique.parameters.forEach(param => {
            const input = $(`#param-${param.name}`);
            if (input) {
                config[param.name] = param.type === 'int'
                    ? parseInt(input.value) || param.default
                    : parseFloat(input.value) || param.default;
            }
        });
    }

    // Show processing
    setStep(3);
    renderProcessingSteps();

    try {
        const response = await fetch(`${API_BASE}/videos/${state.videoInfo.file_id}/summarize/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Summarization failed');
        }

        const result = await response.json();
        state.lastResult = result;
        showResults(result);
    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.classList.remove('hidden');
        setStep(2);
    }
}

// Render Functions
function renderTechniques() {
    const grid = $('#technique-grid');
    grid.innerHTML = state.techniques.map(technique => `
        <button class="technique-card ${technique.id === state.selectedTechnique ? 'selected' : ''}"
                data-technique="${technique.id}">
            <div class="technique-header">
                <div class="technique-icon">
                    ${TECHNIQUE_ICONS[technique.id] || TECHNIQUE_ICONS.combined}
                </div>
                <span class="technique-name">${technique.name}</span>
            </div>
            <p class="technique-desc">${technique.description}</p>
        </button>
    `).join('');

    // Add click handlers
    $$('.technique-card').forEach(card => {
        card.addEventListener('click', () => selectTechnique(card.dataset.technique));
    });
}

function selectTechnique(techniqueId) {
    state.selectedTechnique = techniqueId;
    state.config.technique = techniqueId;

    // Update UI
    $$('.technique-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.technique === techniqueId);
    });

    // Show technique-specific parameters
    renderTechniqueParams();
}

function renderTechniqueParams() {
    const technique = state.techniques.find(t => t.id === state.selectedTechnique);
    const paramsContainer = $('#technique-params');
    const paramsGrid = $('#technique-params-grid');
    const paramsTitle = $('#technique-params-title');

    if (!technique || technique.parameters.length === 0) {
        paramsContainer.classList.add('hidden');
        return;
    }

    paramsContainer.classList.remove('hidden');
    paramsTitle.textContent = `${technique.name} Parameters`;

    paramsGrid.innerHTML = technique.parameters.map(param => `
        <div class="form-group">
            <label for="param-${param.name}">
                ${param.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </label>
            <input type="number"
                   id="param-${param.name}"
                   value="${param.default}"
                   min="${param.min}"
                   max="${param.max}"
                   step="${param.type === 'float' ? '0.1' : '1'}"
                   class="input">
            <p class="form-hint">${param.description}</p>
        </div>
    `).join('');
}

function renderProcessingSteps() {
    const steps = PROCESSING_STEPS[state.selectedTechnique] || PROCESSING_STEPS.combined;
    const container = $('#processing-steps');

    container.innerHTML = steps.map(step => `
        <div class="processing-step">
            <div class="processing-step-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                </svg>
            </div>
            <span>${step}</span>
        </div>
    `).join('');

    $('#processing-technique').textContent = `Using ${state.selectedTechnique.replace('_', ' ')} technique`;
}

function showConfigureStep() {
    const info = state.videoInfo;

    // Update video info
    $('#info-size').textContent = formatBytes(info.size_bytes);
    $('#info-duration').textContent = formatDuration(info.duration_seconds);
    $('#info-resolution').textContent = `${info.width} × ${info.height}`;
    $('#info-frames').textContent = info.total_frames.toLocaleString();
    $('#info-filename').textContent = info.filename;
    $('#info-fps').textContent = info.fps.toFixed(1);

    setStep(2);
}

function showResults(result) {
    const outputType = result.output_type || 'dynamic';
    const videoContainer = $('#result-video-container');
    const storyboardContainer = $('#result-storyboard-container');
    const downloadBtn = $('#download-btn');
    const downloadBtnText = $('#download-btn-text');

    // Update stats
    $('#stat-processed').textContent = result.total_frames_processed.toLocaleString();
    $('#stat-selected').textContent = result.key_frames_selected;
    $('#stat-compression').textContent = `${((1 - result.compression_ratio) * 100).toFixed(1)}%`;

    // Update other info
    $('#result-time').textContent = `Processed in ${result.duration_seconds.toFixed(1)} seconds`;
    $('#result-technique').textContent = result.technique.replace('_', ' ');

    // Handle different output types
    if (outputType === 'dynamic' || outputType === 'both') {
        videoContainer.classList.remove('hidden');
        $('#result-video').src = result.output_url;
        downloadBtn.href = result.output_url;
        downloadBtn.classList.remove('hidden');
        downloadBtnText.textContent = 'Download Video';
    } else {
        videoContainer.classList.add('hidden');
    }

    if (outputType === 'static' || outputType === 'both') {
        storyboardContainer.classList.remove('hidden');
        renderStoryboard(result.frames || []);

        if (result.storyboard_url) {
            $('#download-all-frames').onclick = () => {
                window.open(result.storyboard_url, '_blank');
            };
        }

        if (outputType === 'static') {
            downloadBtn.href = result.storyboard_url || '#';
            downloadBtnText.textContent = 'Download Storyboard';
        }
    } else {
        storyboardContainer.classList.add('hidden');
    }

    setStep(4);
}

function renderStoryboard(frames) {
    const grid = $('#storyboard-grid');

    if (!frames || frames.length === 0) {
        grid.innerHTML = '<p class="loading-placeholder">No frames available</p>';
        return;
    }

    grid.innerHTML = frames.map((frame, i) => `
        <div class="storyboard-frame" data-index="${i}">
            <img src="data:image/jpeg;base64,${frame.thumbnail}" alt="Frame ${i + 1}">
            <div class="storyboard-frame-info">
                <span class="storyboard-frame-number">#${i + 1}</span>
                <span class="storyboard-frame-time">${formatTimestamp(frame.timestamp)}</span>
            </div>
            <button class="storyboard-frame-download" data-index="${i}" title="Download frame">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
            </button>
        </div>
    `).join('');

    // Add click handlers for frame preview
    $$('.storyboard-frame').forEach(frame => {
        frame.addEventListener('click', (e) => {
            if (!e.target.closest('.storyboard-frame-download')) {
                const index = parseInt(frame.dataset.index);
                openFrameModal(frames[index], index);
            }
        });
    });

    // Add click handlers for individual downloads
    $$('.storyboard-frame-download').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const index = parseInt(btn.dataset.index);
            downloadFrame(frames[index], index);
        });
    });
}

function formatTimestamp(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function openFrameModal(frame, index) {
    const modal = $('#frame-modal');
    const img = $('#modal-image');
    const frameInfo = $('#modal-frame-info');
    const timestamp = $('#modal-timestamp');
    const downloadLink = $('#modal-download');

    img.src = `data:image/jpeg;base64,${frame.thumbnail}`;
    frameInfo.textContent = `Frame ${index + 1}`;
    timestamp.textContent = formatTimestamp(frame.timestamp);

    downloadLink.onclick = (e) => {
        e.preventDefault();
        downloadFrame(frame, index);
    };

    modal.classList.remove('hidden');
}

function closeFrameModal() {
    $('#frame-modal').classList.add('hidden');
}

function downloadFrame(frame, index) {
    const link = document.createElement('a');
    link.href = `data:image/jpeg;base64,${frame.thumbnail}`;
    link.download = `frame_${index + 1}_${frame.timestamp.toFixed(2)}s.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function resetApp() {
    state.selectedFile = null;
    state.videoInfo = null;
    state.selectedTechnique = 'combined';
    state.selectedOutputType = 'dynamic';
    state.selectedSummaryLength = 'medium';
    state.summaryPercent = 20;
    state.lastResult = null;
    state.lastAnalytics = null;

    // Reset AI insights card
    const insightsContent = $('#ai-insights-content');
    if (insightsContent) {
        insightsContent.innerHTML = '<button id="generate-insights-btn" class="btn btn-secondary">Generate Insights</button>';
    }

    // Reset UI
    $('#dropzone').classList.remove('hidden');
    $('#preview-section').classList.add('hidden');
    $('#video-preview').src = '';
    $('#file-name').textContent = '';
    $('#file-size').textContent = '';
    $('#upload-error').classList.add('hidden');
    $('#config-error').classList.add('hidden');

    // Reset file input so same file can be selected again
    $('#file-input').value = '';

    // Reset results section
    $('#result-video-container').classList.remove('hidden');
    $('#result-storyboard-container').classList.add('hidden');
    $('#result-video').src = '';

    // Reset technique and output type selection
    renderTechniques();
    renderTechniqueParams();
    selectOutputType('dynamic');
    selectSummaryLength('medium');

    // Reset custom percent input
    const customPercent = $('#custom-percent');
    if (customPercent) customPercent.value = 25;

    setStep(1);
}

function selectOutputType(outputType) {
    state.selectedOutputType = outputType;
    state.config.output_type = outputType;

    // Update UI
    $$('.output-type-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.output === outputType);
    });
}

function selectSummaryLength(length) {
    state.selectedSummaryLength = length;

    if (length === 'custom') {
        state.summaryPercent = parseInt($('#custom-percent').value) || 20;
    } else {
        state.summaryPercent = SUMMARY_LENGTHS[length] || 20;
    }

    state.config.summary_percent = state.summaryPercent;

    // Update UI
    $$('.summary-length-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.length === length);
    });
}

// Event Handlers
function initEventListeners() {
    const dropzone = $('#dropzone');
    const fileInput = $('#file-input');
    const previewSection = $('#preview-section');
    const videoPreview = $('#video-preview');

    // Dropzone events
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-over');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) handleFileSelect(file);
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFileSelect(file);
    });

    // Clear preview
    $('#clear-preview').addEventListener('click', () => {
        state.selectedFile = null;
        dropzone.classList.remove('hidden');
        previewSection.classList.add('hidden');
        videoPreview.src = '';
        fileInput.value = '';
    });

    // Upload button
    $('#upload-btn').addEventListener('click', () => {
        if (state.selectedFile) {
            uploadVideo(state.selectedFile);
        }
    });

    // Advanced options toggle
    $('#advanced-toggle').addEventListener('click', function() {
        const options = $('#advanced-options');
        options.classList.toggle('hidden');
        this.classList.toggle('open');
    });

    // Back button
    $('#back-btn').addEventListener('click', resetApp);

    // Summarize button
    $('#summarize-btn').addEventListener('click', summarizeVideo);

    // New video button
    $('#new-video-btn').addEventListener('click', resetApp);
}

function handleFileSelect(file) {
    // Validate file type
    const allowedTypes = ['video/mp4', 'video/mpeg', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    const allowedExtensions = ['.mp4', '.mpg', '.mpeg', '.avi', '.mov', '.mkv', '.webm'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(ext)) {
        alert('Invalid file type. Please upload a video file.');
        return;
    }

    // Validate file size (500MB)
    if (file.size > 500 * 1024 * 1024) {
        alert('File too large. Maximum size is 500MB.');
        return;
    }

    state.selectedFile = file;

    // Show preview
    const dropzone = $('#dropzone');
    const previewSection = $('#preview-section');
    const videoPreview = $('#video-preview');

    dropzone.classList.add('hidden');
    previewSection.classList.remove('hidden');

    videoPreview.src = URL.createObjectURL(file);
    $('#file-name').textContent = file.name;
    $('#file-size').textContent = formatBytes(file.size);
}

// Analytics Functions
async function loadAnalytics() {
    if (!state.videoInfo) return;

    const heatmapContainer = $('#heatmap-container');
    const statsContainer = $('#analytics-stats');
    const comparisonContainer = $('#comparison-container');
    const chartContainer = $('#chart-container');

    heatmapContainer.innerHTML = '<div class="loading-placeholder">Analyzing video frames...</div>';
    statsContainer.innerHTML = '<div class="loading-placeholder">Computing statistics...</div>';
    comparisonContainer.innerHTML = '<div class="loading-placeholder">Comparing techniques...</div>';

    try {
        // Fetch analytics data
        const response = await fetch(`${API_BASE}/videos/${state.videoInfo.file_id}/analytics/?sample_rate=10&max_frames=300`);
        const data = await response.json();
        state.lastAnalytics = data;

        // Render motion heatmap
        if (data.motion_heatmap) {
            heatmapContainer.innerHTML = `<img src="data:image/jpeg;base64,${data.motion_heatmap}" alt="Motion Heatmap">`;
        } else {
            heatmapContainer.innerHTML = '<div class="loading-placeholder">Heatmap not available</div>';
        }

        // Render statistics
        const stats = data.summary_stats;
        statsContainer.innerHTML = `
            <div class="analytics-stat">
                <p class="analytics-stat-value">${stats.total_frames}</p>
                <p class="analytics-stat-label">Total Frames</p>
            </div>
            <div class="analytics-stat">
                <p class="analytics-stat-value highlight">${stats.avg_motion.toFixed(2)}</p>
                <p class="analytics-stat-label">Avg Motion</p>
            </div>
            <div class="analytics-stat">
                <p class="analytics-stat-value">${stats.max_motion.toFixed(2)}</p>
                <p class="analytics-stat-label">Peak Motion</p>
            </div>
            <div class="analytics-stat">
                <p class="analytics-stat-value">${stats.high_activity_frames}</p>
                <p class="analytics-stat-label">High Activity</p>
            </div>
            <div class="analytics-stat">
                <p class="analytics-stat-value">${stats.low_activity_frames}</p>
                <p class="analytics-stat-label">Low Activity</p>
            </div>
            <div class="analytics-stat">
                <p class="analytics-stat-value">${stats.color_diversity_avg.toFixed(3)}</p>
                <p class="analytics-stat-label">Color Diversity</p>
            </div>
            <div class="analytics-stat">
                <p class="analytics-stat-value">${stats.motion_variance.toFixed(2)}</p>
                <p class="analytics-stat-label">Motion Variance</p>
            </div>
            <div class="analytics-stat">
                <p class="analytics-stat-value highlight">${stats.keyframes_count}</p>
                <p class="analytics-stat-label">Key Frames</p>
            </div>
        `;

        // Render chart
        renderScoresChart(data.technique_scores);

        // Fetch technique comparison
        const compResponse = await fetch(`${API_BASE}/videos/${state.videoInfo.file_id}/compare/?sample_rate=10&max_frames=200`);
        const compData = await compResponse.json();

        renderComparison(compData);

    } catch (error) {
        console.error('Failed to load analytics:', error);
        heatmapContainer.innerHTML = '<div class="loading-placeholder">Failed to load analytics</div>';
    }
}

function renderScoresChart(scores) {
    const canvas = $('#scores-chart');
    const ctx = canvas.getContext('2d');
    const container = $('#chart-container');

    // Set canvas size
    canvas.width = container.offsetWidth - 32;
    canvas.height = 250;

    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 20, right: 20, bottom: 30, left: 40 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background grid (theme-aware)
    ctx.strokeStyle = themeColor('--chart-grid', '#e5e7eb');
    ctx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartHeight / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();

        // Y-axis labels
        ctx.fillStyle = themeColor('--chart-label', '#9ca3af');
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText((1 - i * 0.25).toFixed(1), padding.left - 8, y + 4);
    }

    // Draw lines for each technique
    const colors = {
        motion: '#3b82f6',
        color: '#8b5cf6',
        event: '#f59e0b',
        combined: '#10b981'
    };

    const lineWidth = { motion: 1.5, color: 1.5, event: 1.5, combined: 2.5 };

    Object.entries(scores).forEach(([technique, values]) => {
        if (!values || values.length === 0) return;

        ctx.strokeStyle = colors[technique] || '#666';
        ctx.lineWidth = lineWidth[technique] || 1.5;
        ctx.beginPath();

        const xStep = chartWidth / (values.length - 1);

        values.forEach((value, i) => {
            const x = padding.left + i * xStep;
            const y = padding.top + chartHeight * (1 - value);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();
    });

    // X-axis label
    ctx.fillStyle = themeColor('--chart-label', '#9ca3af');
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Frame Index', width / 2, height - 5);
}

function renderComparison(data) {
    const container = $('#comparison-container');

    if (!data.techniques) {
        container.innerHTML = '<div class="loading-placeholder">No comparison data available</div>';
        return;
    }

    // Find best technique (most keyframes with good coverage)
    let bestTechnique = null;
    let bestScore = 0;

    Object.entries(data.techniques).forEach(([name, info]) => {
        const score = info.keyframe_count * (1 + info.coverage);
        if (score > bestScore) {
            bestScore = score;
            bestTechnique = name;
        }
    });

    const techniqueNames = {
        motion: 'Motion-based',
        color: 'Color/Histogram',
        event: 'Event-based',
        object_detection: 'Object Detection',
        combined: 'Combined'
    };

    container.innerHTML = Object.entries(data.techniques).map(([name, info]) => {
        const isBest = name === bestTechnique;
        const coveragePercent = (info.coverage * 100).toFixed(1);

        return `
            <div class="comparison-card ${isBest ? 'best' : ''}">
                <div class="comparison-card-header">
                    <span class="comparison-card-title">${techniqueNames[name] || name}</span>
                    ${isBest ? '<span class="comparison-badge">Best Match</span>' : ''}
                </div>
                <div class="comparison-stats">
                    <div class="comparison-stat-row">
                        <span class="comparison-stat-label">Key Frames</span>
                        <span class="comparison-stat-value">${info.keyframe_count}</span>
                    </div>
                    <div class="comparison-stat-row">
                        <span class="comparison-stat-label">Coverage</span>
                        <span class="comparison-stat-value">${coveragePercent}%</span>
                    </div>
                    <div class="comparison-stat-row">
                        <span class="comparison-stat-label">Frame Spread</span>
                        <span class="comparison-stat-value">${info.keyframe_indices.length > 0 ?
                            (info.keyframe_indices[info.keyframe_indices.length - 1] - info.keyframe_indices[0]) : 0
                        }</span>
                    </div>
                </div>
                <div class="comparison-bar">
                    <div class="comparison-bar-fill" style="width: ${Math.min(100, info.keyframe_count * 5)}%"></div>
                </div>
            </div>
        `;
    }).join('');
}

// Smart Insights (local analysis - always available)
function checkAiAvailability() {
    // Local insights are always available - no external API needed
    state.aiInsightsAvailable = true;
}

async function generateAiInsights() {
    if (!state.videoInfo) return;

    const content = $('#ai-insights-content');
    content.innerHTML = '<div class="loading-placeholder">Analyzing your video...</div>';

    try {
        const response = await fetch(`${API_BASE}/videos/${state.videoInfo.file_id}/insights/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state.lastResult ? {
                technique: state.lastResult.technique,
                output_type: state.lastResult.output_type,
                total_frames_processed: state.lastResult.total_frames_processed,
                key_frames_selected: state.lastResult.key_frames_selected,
                compression_ratio: state.lastResult.compression_ratio,
                duration_seconds: state.lastResult.duration_seconds,
                summary_percent: state.summaryPercent,
            } : {})
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to generate insights');
        }

        const p = document.createElement('p');
        p.className = 'ai-insights-text';
        p.textContent = data.insights;
        content.innerHTML = '';
        content.appendChild(p);
    } catch (error) {
        content.innerHTML = `<div class="loading-placeholder">${error.message}</div>
            <button id="generate-insights-btn" class="btn btn-secondary" style="margin-top:0.5rem;">Try Again</button>`;
        const retryBtn = $('#generate-insights-btn');
        if (retryBtn) retryBtn.addEventListener('click', generateAiInsights);
    }
}

function showAnalytics() {
    $('#results-section').classList.add('hidden');
    $('#analytics-section').classList.remove('hidden');

    // Show Smart Insights card - always available (local analysis)
    const aiCard = $('#ai-insights-card');
    if (aiCard) {
        aiCard.classList.remove('hidden');
        const content = $('#ai-insights-content');
        content.innerHTML = '<button id="generate-insights-btn" class="btn btn-secondary">Generate Insights</button>';
        const btn = $('#generate-insights-btn');
        if (btn) btn.onclick = generateAiInsights;
    }

    loadAnalytics();
}

function hideAnalytics() {
    $('#analytics-section').classList.add('hidden');
    $('#results-section').classList.remove('hidden');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initEventListeners();
    fetchTechniques();
    checkAiAvailability();
    setStep(1);

    // Theme toggle
    const themeToggle = $('#theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // Summary length selection handlers
    $$('.summary-length-card').forEach(card => {
        card.addEventListener('click', () => selectSummaryLength(card.dataset.length));
    });

    // Custom percent input handler
    const customPercent = $('#custom-percent');
    if (customPercent) {
        customPercent.addEventListener('click', (e) => {
            e.stopPropagation();
            selectSummaryLength('custom');
        });
        customPercent.addEventListener('change', () => {
            if (state.selectedSummaryLength === 'custom') {
                state.summaryPercent = parseInt(customPercent.value) || 20;
            }
        });
    }

    // Output type selection handlers
    $$('.output-type-card').forEach(card => {
        card.addEventListener('click', () => selectOutputType(card.dataset.output));
    });

    // Modal close handlers
    const closeModalBtn = $('#close-modal');
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', closeFrameModal);
    }

    const modal = $('#frame-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeFrameModal();
        });
    }

    // Analytics button handlers
    const analyticsBtn = $('#analytics-btn');
    if (analyticsBtn) {
        analyticsBtn.addEventListener('click', showAnalytics);
    }

    const backToResults = $('#back-to-results');
    if (backToResults) {
        backToResults.addEventListener('click', hideAnalytics);
    }

    // Escape key to close modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeFrameModal();
        }
    });
});
