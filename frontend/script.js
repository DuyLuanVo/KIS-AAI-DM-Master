// Configuration
const API_BASE_URL = 'http://localhost:8000';
const GRID_SIZE = 20; // 5x4 grid
const ITEMS_PER_PAGE = GRID_SIZE;

// Global state
let currentResults = [];
let currentGroupedResults = [];
let currentPage = 0;
let currentQueries = [];
let currentObjects = [];
let searchMode = 'text'; // 'text' or 'image'
let uploadedImageBase64 = null;

// Carousel state
let currentFrames = [];
let currentFrameIndex = 0;
let centerFrameIndex = 0;

// DOM elements
const queryInput = document.getElementById('queryInput');
const objectInput = document.getElementById('objectInput');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const textModeBtn = document.getElementById('textModeBtn');
const imageModeBtn = document.getElementById('imageModeBtn');
const searchBtn = document.getElementById('searchBtn');
const searchMoreBtn = document.getElementById('searchMoreBtn');
const resultsSection = document.getElementById('resultsSection');
const gridContainer = document.getElementById('gridContainer');
const paginationInfo = document.getElementById('paginationInfo');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const queryStats = document.getElementById('queryStats');

// Carousel elements
const carouselModal = document.getElementById('carouselModal');
const closeCarousel = document.getElementById('closeCarousel');
const prevFrame = document.getElementById('prevFrame');
const nextFrame = document.getElementById('nextFrame');
const currentFrameImg = document.getElementById('currentFrameImg');
const framePosition = document.getElementById('framePosition');
const frameTime = document.getElementById('frameTime');
const frameId = document.getElementById('frameId');
const frameIdx = document.getElementById('frameIdx');
const thumbnailContainer = document.getElementById('thumbnailContainer');
const carouselTitle = document.getElementById('carouselTitle');

// Event listeners
searchBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    handleSearch();
});
searchMoreBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    handleSearchMore();
});
prevBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    changePage(-1);
});
nextBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    changePage(1);
});
textModeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    setSearchMode('text');
});
imageModeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    setSearchMode('image');
});
imageInput.addEventListener('change', handleImageUpload);

// Carousel event listeners
closeCarousel.addEventListener('click', closeCarouselModal);
prevFrame.addEventListener('click', showPreviousFrame);
nextFrame.addEventListener('click', showNextFrame);
carouselModal.addEventListener('click', (e) => {
    if (e.target === carouselModal) {
        closeCarouselModal();
    }
});

// Keyboard navigation for carousel
document.addEventListener('keydown', (e) => {
    if (carouselModal.style.display === 'flex') {
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            showPreviousFrame();
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            showNextFrame();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            closeCarouselModal();
        }
    }
});

// Clipboard paste functionality
document.addEventListener('paste', async (e) => {
    // Only handle paste when in image search mode and not in carousel
    if (searchMode !== 'image' || carouselModal.style.display === 'flex') {
        return;
    }

    e.preventDefault();

    // Visual feedback
    imagePreview.classList.add('paste-active');
    setTimeout(() => {
        imagePreview.classList.remove('paste-active');
    }, 300);

    const items = e.clipboardData.items;
    let imageItem = null;

    // Find image in clipboard
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
            imageItem = items[i];
            break;
        }
    }

    if (imageItem) {
        console.log('📋 Image pasted from clipboard');
        const file = imageItem.getAsFile();
        await handleImageFile(file, 'clipboard');
    } else {
        console.log('📋 No image found in clipboard');
        // Show temporary message
        const originalContent = imagePreview.innerHTML;
        imagePreview.innerHTML = `
            <div class="upload-placeholder" style="color: #e74c3c;">
                ❌ Không tìm thấy ảnh trong clipboard<br>
                <small>Hãy copy ảnh trước khi dán</small>
            </div>
        `;
        setTimeout(() => {
            imagePreview.innerHTML = originalContent;
        }, 2000);
    }
});

// Set search mode
function setSearchMode(mode) {
    searchMode = mode;

    // Update button states
    textModeBtn.classList.toggle('active', mode === 'text');
    imageModeBtn.classList.toggle('active', mode === 'image');

    // Show/hide relevant sections
    document.getElementById('textInputSection').style.display = mode === 'text' ? 'block' : 'none';
    document.getElementById('imageInputSection').style.display = mode === 'image' ? 'block' : 'none';
}

// Handle image file (from upload or clipboard)
async function handleImageFile(file, source = 'upload') {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showError('Vui lòng chọn file ảnh hợp lệ');
        return;
    }

    console.log(`🖼️ Processing image from ${source}:`, file.name || 'clipboard', file.type);

    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedImageBase64 = e.target.result.split(',')[1]; // Remove data:image/...;base64, prefix

        const sourceLabel = source === 'clipboard' ? 'Ảnh từ clipboard' : 'Ảnh đã upload';
        imagePreview.innerHTML = `
            <img src="${e.target.result}"
                 alt="${sourceLabel}"
                 style="max-width: 100%; max-height: 200px; border-radius: 4px;" />
            <div style="margin-top: 8px; font-size: 12px; color: #666; text-align: center;">
                📋 ${sourceLabel} • ${file.type}
            </div>
        `;

        console.log('✅ Image processed successfully');
    };

    reader.onerror = function() {
        showError('Lỗi đọc file ảnh');
        console.error('❌ Error reading image file');
    };

    reader.readAsDataURL(file);
}

// Handle image upload
function handleImageUpload(event) {
    const file = event.target.files[0];
    handleImageFile(file, 'upload');
}

// Handle search button click
async function handleSearch() {
    console.log('🔍 Search started, mode:', searchMode);

    try {
        if (searchMode === 'text') {
            const queries = getQueriesFromInput();
            console.log('📝 Text queries:', queries);

            if (queries.length === 0) {
                showError('Vui lòng nhập ít nhất một câu query');
                return;
            }
            currentQueries = queries;
            currentObjects = getObjectsFromInput();
            console.log('🔍 Object filters:', currentObjects);

            await performTextSearch(queries, currentObjects, true);
        } else {
            console.log('🖼️ Image search mode');

            if (!uploadedImageBase64) {
                showError('Vui lòng upload ảnh để tìm kiếm');
                return;
            }
            currentObjects = getObjectsFromInput();
            console.log('🔍 Object filters:', currentObjects);

            await performImageSearch(uploadedImageBase64, currentObjects, true);
        }
    } catch (error) {
        console.error('❌ Error in handleSearch:', error);
        showError(`Lỗi tìm kiếm: ${error.message}`);
    }
}

// Handle search more button click
async function handleSearchMore() {
    if (searchMode === 'text') {
        if (currentQueries.length === 0) {
            showError('Không có query để tìm tiếp');
            return;
        }
        await performTextSearch(currentQueries, currentObjects, false);
    } else {
        if (!uploadedImageBase64) {
            showError('Không có ảnh để tìm tiếp');
            return;
        }
        await performImageSearch(uploadedImageBase64, currentObjects, false);
    }
}

// Get queries from textarea input
function getQueriesFromInput() {
    const input = queryInput.value.trim();
    if (!input) return [];

    return input.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
}

// Get objects from textarea input
function getObjectsFromInput() {
    const input = objectInput.value.trim();
    if (!input) return [];

    return input.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
}

// Perform text search API call
async function performTextSearch(queries, objects, isNewSearch) {
    console.log('📡 Starting text search API call...');

    try {
        showLoading(true);
        hideError();

        const requestBody = {
            query_texts: queries,
            object_filters: objects.length > 0 ? objects : null,
            limit: 1000,
            score_threshold: 0.0
        };

        console.log('📤 Request body:', requestBody);
        console.log('🌐 API URL:', `${API_BASE_URL}/api/v1/videos/search/text`);

        const response = await fetch(`${API_BASE_URL}/api/v1/videos/search/text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        console.log('📡 Response status:', response.status);
        console.log('📡 Response headers:', response.headers);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ API Error Response:', errorText);
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const data = await response.json();
        console.log('✅ API Response data:', data);

        handleSearchResponse(data, isNewSearch);

    } catch (error) {
        console.error('❌ Text search error:', error);
        showError(`Lỗi tìm kiếm văn bản: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Perform image search API call
async function performImageSearch(imageBase64, objects, isNewSearch) {
    console.log('📡 Starting image search API call...');

    try {
        showLoading(true);
        hideError();

        const requestBody = {
            image_base64: imageBase64,
            object_filters: objects.length > 0 ? objects : null,
            limit: 1000,
            score_threshold: 0.0
        };

        console.log('📤 Request body (image base64 length):', imageBase64.length);
        console.log('📤 Object filters:', objects);
        console.log('🌐 API URL:', `${API_BASE_URL}/api/v1/videos/search/image`);

        const response = await fetch(`${API_BASE_URL}/api/v1/videos/search/image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        console.log('📡 Response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ API Error Response:', errorText);
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const data = await response.json();
        console.log('✅ API Response data:', data);

        handleSearchResponse(data, isNewSearch);

    } catch (error) {
        console.error('❌ Image search error:', error);
        showError(`Lỗi tìm kiếm ảnh: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Handle search response
function handleSearchResponse(data, isNewSearch) {
    console.log('📊 Handling search response:', data);
    console.log('🆕 Is new search:', isNewSearch);

    try {
        if (isNewSearch) {
            currentResults = data.results || [];
            currentGroupedResults = data.grouped_by_video || [];
            currentPage = 0;
            console.log('🔄 Reset to new results:', currentResults.length, 'items');
        } else {
            // Append new results to existing ones
            const newResults = data.results || [];
            currentResults = [...currentResults, ...newResults];
            console.log('➕ Added', newResults.length, 'new results. Total:', currentResults.length);

            // Merge grouped results
            mergeGroupedResults(data.grouped_by_video || []);
        }

        console.log('📈 Current results count:', currentResults.length);
        console.log('📺 Grouped videos count:', currentGroupedResults.length);

        updateQueryStats(data);
        displayResults();
        updateButtonStates();

        console.log('✅ Search response handled successfully');

    } catch (error) {
        console.error('❌ Error handling search response:', error);
        showError(`Lỗi xử lý kết quả: ${error.message}`);
    }
}

// Merge grouped results for "search more"
function mergeGroupedResults(newGroupedResults) {
    const existingGroups = new Map();

    // Index existing groups
    currentGroupedResults.forEach(group => {
        existingGroups.set(group.video_id, group);
    });

    // Merge new groups
    newGroupedResults.forEach(newGroup => {
        if (existingGroups.has(newGroup.video_id)) {
            const existing = existingGroups.get(newGroup.video_id);
            existing.frames = [...existing.frames, ...newGroup.frames];
            existing.total_frames = existing.frames.length;
            existing.best_score = Math.max(existing.best_score, newGroup.best_score);
        } else {
            currentGroupedResults.push(newGroup);
        }
    });

    // Re-sort by best score
    currentGroupedResults.sort((a, b) => b.best_score - a.best_score);
}

// Display results in grid
function displayResults() {
    console.log('🎨 Displaying results...');
    console.log('📊 Total results:', currentResults.length);

    try {
        if (currentResults.length === 0) {
            console.log('📭 No results to display');
            resultsSection.style.display = 'none';
            return;
        }

        resultsSection.style.display = 'block';

        const startIndex = currentPage * ITEMS_PER_PAGE;
        const endIndex = Math.min(startIndex + ITEMS_PER_PAGE, currentResults.length);
        const pageResults = currentResults.slice(startIndex, endIndex);

        console.log(`📄 Page ${currentPage + 1}: showing ${startIndex + 1}-${endIndex} of ${currentResults.length}`);

        // Update pagination info
        paginationInfo.textContent = `Hiển thị ${startIndex + 1}-${endIndex} của ${currentResults.length} kết quả`;

        // Clear grid
        gridContainer.innerHTML = '';
        console.log('🧹 Grid cleared');

        // Add grid items
        pageResults.forEach((result, index) => {
            console.log(`🖼️ Creating grid item ${index + 1}:`, result.video_id);
            const gridItem = createGridItem(result);
            gridContainer.appendChild(gridItem);
        });

        // Fill remaining slots with empty items if needed
        const remainingSlots = ITEMS_PER_PAGE - pageResults.length;
        for (let i = 0; i < remainingSlots; i++) {
            const emptyItem = createEmptyGridItem();
            gridContainer.appendChild(emptyItem);
        }

        console.log('✅ Results displayed successfully');

    } catch (error) {
        console.error('❌ Error displaying results:', error);
        showError(`Lỗi hiển thị kết quả: ${error.message}`);
    }
}

// Create a grid item for a result
function createGridItem(result) {
    const gridItem = document.createElement('div');
    gridItem.className = 'grid-item';

    // Build image path - handle relative paths from data folder
    let imageUrl;
    if (result.jpg_path.startsWith('http')) {
        imageUrl = result.jpg_path;
    } else {
        // Remove leading slash if present and add data path
        const cleanPath = result.jpg_path.replace(/^\//, '');
        imageUrl = `/data/${cleanPath}`;
    }

    // Format timestamp
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

        gridItem.innerHTML = `
        <div class="grid-item-image" style="background-image: url('${imageUrl}')"
             onerror="this.classList.add('error'); this.innerHTML='Không thể tải ảnh';">
            <div class="rank-badge">#${result.rank}</div>
        </div>
        <div class="grid-item-info">
            <div class="grid-item-title">${result.video_id}</div>
            <div class="grid-item-score">Điểm: ${(result.similarity_score * 100).toFixed(1)}%</div>
            <div class="grid-item-time">⏱️ ${formatTime(result.pts_time)}</div>
            <div class="grid-item-frame">Frame: ${result.keyframe_idx}</div>
        </div>
    `;

    // Add click handler to open carousel
    gridItem.addEventListener('click', () => {
        openCarouselModal(result);
    });

    return gridItem;
}

// Create an empty grid item
function createEmptyGridItem() {
    const gridItem = document.createElement('div');
    gridItem.className = 'grid-item';
    gridItem.style.opacity = '0.3';

    gridItem.innerHTML = `
        <div class="grid-item-image" style="background-color: #f0f0f0;"></div>
        <div class="grid-item-info">
            <div class="grid-item-title">-</div>
            <div class="grid-item-score">-</div>
            <div class="grid-item-frame">-</div>
        </div>
    `;

    return gridItem;
}

// Change page
function changePage(direction) {
    const newPage = currentPage + direction;
    const maxPage = Math.ceil(currentResults.length / ITEMS_PER_PAGE) - 1;

    if (newPage >= 0 && newPage <= maxPage) {
        currentPage = newPage;
        displayResults();
        updateButtonStates();
    }
}

// Update query statistics
function updateQueryStats(data) {
    if (queryStats) {
        const statsHtml = `
            <div class="stats-item">
                <span class="stats-label">Tổng kết quả:</span>
                <span class="stats-value">${data.total_results}</span>
            </div>
            <div class="stats-item">
                <span class="stats-label">Thời gian:</span>
                <span class="stats-value">${Math.round(data.query_time_ms)}ms</span>
            </div>
            <div class="stats-item">
                <span class="stats-label">Video tìm thấy:</span>
                <span class="stats-value">${data.grouped_by_video.length}</span>
            </div>
        `;
        queryStats.innerHTML = statsHtml;
    }
}

// Update button states
function updateButtonStates() {
    const maxPage = Math.ceil(currentResults.length / ITEMS_PER_PAGE) - 1;

    prevBtn.disabled = currentPage <= 0;
    nextBtn.disabled = currentPage >= maxPage;

    // Update search more button based on search mode
    if (searchMode === 'text') {
        searchMoreBtn.disabled = currentQueries.length === 0;
    } else {
        searchMoreBtn.disabled = !uploadedImageBase64;
    }
}

// Show/hide loading spinner
function showLoading(show) {
    console.log('⏳ Loading:', show);

    loadingSpinner.style.display = show ? 'block' : 'none';
    searchBtn.disabled = show;

    // Only disable search more if loading or no valid queries/image
    if (show) {
        searchMoreBtn.disabled = true;
    } else {
        updateButtonStates(); // Re-enable based on current state
    }
}

// Show error message
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'block';
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Initialize the app
function init() {
    console.log('🚀 Initializing Video Search App...');
    console.log('🌐 API Base URL:', API_BASE_URL);

    // Set default search mode
    setSearchMode('text');

    // Set up some sample data for testing
    queryInput.value = '';
    objectInput.value = '';

    // Add keyboard shortcuts
    queryInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            handleSearch();
        }
    });

    objectInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            handleSearch();
        }
    });

    // Initialize button states
    updateButtonStates();

    console.log('✅ App initialized successfully');
}

// Carousel Functions
function openCarouselModal(selectedResult) {
    console.log('🎬 Opening carousel for:', selectedResult);

    try {
        // Parse current frame info from jpg_path
        // Example: "Keyframes_L26/keyframes/L26_V498/016.jpg"
        const pathParts = selectedResult.jpg_path.split('/');
        const fileName = pathParts[pathParts.length - 1]; // "016.jpg"
        const currentFrameNum = parseInt(fileName.replace('.jpg', '')); // 16
        const folderPath = pathParts.slice(0, -1).join('/'); // "Keyframes_L26/keyframes/L26_V498"

        // Generate nearby frames (±25 = 50 total)
        const rangeSize = 25;
        const startFrame = Math.max(1, currentFrameNum - rangeSize);
        const endFrame = currentFrameNum + rangeSize;

        console.log(`📊 Generating frames ${startFrame} to ${endFrame} (center: ${currentFrameNum})`);

        currentFrames = [];
        for (let i = startFrame; i <= endFrame; i++) {
            const frameFileName = i.toString().padStart(3, '0') + '.jpg';
            const framePath = `${folderPath}/${frameFileName}`;

            // Calculate estimated pts_time (rough calculation)
            const estimatedTime = selectedResult.pts_time + ((i - currentFrameNum) * (1 / (selectedResult.fps || 25)));
            const estimatedFrameIdx = selectedResult.frame_idx + ((i - currentFrameNum) * 1);

            currentFrames.push({
                original_id: `${selectedResult.video_id}_${i.toString().padStart(3, '0')}`,
                video_id: selectedResult.video_id,
                keyframe_idx: i,
                keyframe_name: frameFileName,
                jpg_path: framePath,
                pts_time: Math.max(0, estimatedTime),
                frame_idx: Math.max(0, estimatedFrameIdx),
                fps: selectedResult.fps || 25,
                objects: i === currentFrameNum ? selectedResult.objects : [],
                is_center: i === currentFrameNum
            });
        }

        centerFrameIndex = currentFrames.findIndex(f => f.is_center);
        currentFrameIndex = centerFrameIndex;

        // Update modal title
        carouselTitle.textContent = `Khung hình lân cận - ${selectedResult.video_id} (${startFrame}-${endFrame})`;

        // Render carousel
        renderCarousel();

        // Show modal
        carouselModal.style.display = 'flex';

        console.log('✅ Carousel opened successfully with', currentFrames.length, 'frames');

    } catch (error) {
        console.error('❌ Error opening carousel:', error);
        showError(`Lỗi tải khung hình: ${error.message}`);
    }
}

function closeCarouselModal() {
    console.log('🚪 Closing carousel');
    carouselModal.style.display = 'none';
    currentFrames = [];
    currentFrameIndex = 0;
    centerFrameIndex = 0;
}

function showPreviousFrame() {
    if (currentFrameIndex > 0) {
        currentFrameIndex--;
        updateCurrentFrame();
    }
}

function showNextFrame() {
    if (currentFrameIndex < currentFrames.length - 1) {
        currentFrameIndex++;
        updateCurrentFrame();
    }
}

function renderCarousel() {
    console.log('🎨 Rendering carousel with', currentFrames.length, 'frames');

    // Render thumbnails
    thumbnailContainer.innerHTML = '';
    currentFrames.forEach((frame, index) => {
        const thumbnail = createThumbnailItem(frame, index);
        thumbnailContainer.appendChild(thumbnail);
    });

    // Update current frame
    updateCurrentFrame();
}

function createThumbnailItem(frame, index) {
    const thumbnail = document.createElement('div');
    thumbnail.className = 'thumbnail-item';

    if (index === currentFrameIndex) {
        thumbnail.classList.add('active');
    }

    if (frame.is_center) {
        thumbnail.classList.add('center');
    }

    const imageUrl = `/data/${frame.jpg_path}`;

    thumbnail.innerHTML = `
        <img src="${imageUrl}"
             alt="Frame ${frame.keyframe_idx}"
             onerror="this.style.opacity='0.3'; this.alt='❌';"
        />
    `;

    thumbnail.addEventListener('click', () => {
        currentFrameIndex = index;
        updateCurrentFrame();
    });

    return thumbnail;
}

function updateCurrentFrame() {
    if (currentFrames.length === 0) return;

    const frame = currentFrames[currentFrameIndex];
    console.log('🖼️ Updating to frame:', frame.keyframe_idx);

    // Update main image
    const imageUrl = `/data/${frame.jpg_path}`;
    currentFrameImg.src = imageUrl;
    currentFrameImg.onerror = function() {
        this.style.opacity = '0.5';
        this.alt = 'Ảnh không tồn tại';
    };
    currentFrameImg.onload = function() {
        this.style.opacity = '1';
    };

    // Update frame info
    framePosition.textContent = `${currentFrameIndex + 1} / ${currentFrames.length}`;
    frameTime.textContent = formatTime(frame.pts_time);
    frameId.textContent = `${frame.video_id}_${frame.keyframe_idx}`;
    frameIdx.textContent = `Frame: ${frame.frame_idx}`;

    // Update navigation buttons
    prevFrame.disabled = currentFrameIndex === 0;
    nextFrame.disabled = currentFrameIndex === currentFrames.length - 1;

    // Update thumbnail selection
    document.querySelectorAll('.thumbnail-item').forEach((thumb, index) => {
        thumb.classList.toggle('active', index === currentFrameIndex);
    });

    // Scroll thumbnail into view
    const activeThumbnail = document.querySelector('.thumbnail-item.active');
    if (activeThumbnail) {
        activeThumbnail.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest',
            inline: 'center'
        });
    }
}

// Format time helper (already exists but ensuring it's available)
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

// Start the app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
