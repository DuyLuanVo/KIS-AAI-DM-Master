// Configuration
const API_BASE_URL = window.location.hostname 
    ? `${window.location.protocol}//${window.location.hostname}:8000` 
    : 'http://localhost:8000';
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

// Tab Elements
const searchTabBtn = document.getElementById('searchTabBtn');
const ingestTabBtn = document.getElementById('ingestTabBtn');
const searchTabContent = document.getElementById('searchTabContent');
const ingestTabContent = document.getElementById('ingestTabContent');

// Ingest Form Elements
const ingestBatchNameInput = document.getElementById('ingestBatchNameInput');
const ingestCategoriesInput = document.getElementById('ingestCategoriesInput');
const ingestDescInput = document.getElementById('ingestDescInput');
const ingestFilesInput = document.getElementById('ingestFilesInput');
const ingestFilesCountHint = document.getElementById('ingestFilesCountHint');
const startIngestBtn = document.getElementById('startIngestBtn');

// Ingest Status Containers
const ingestWelcomeBox = document.getElementById('ingestWelcomeBox');
const ingestProgressBox = document.getElementById('ingestProgressBox');
const ingestSuccessBox = document.getElementById('ingestSuccessBox');
const ingestSuccessMsg = document.getElementById('ingestSuccessMsg');
const ingestSuccessDetail = document.getElementById('ingestSuccessDetail');
const ingestDetectedChips = document.getElementById('ingestDetectedChips');
const goToSearchAfterIngestBtn = document.getElementById('goToSearchAfterIngestBtn');
const ingestResultSummaryBadge = document.getElementById('ingestResultSummaryBadge');

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
if (prevFrame) prevFrame.addEventListener('click', showPreviousFrame);
if (nextFrame) nextFrame.addEventListener('click', showNextFrame);
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

// Get objects from textarea input (supports newline, comma, semicolon)
function getObjectsFromInput() {
    const input = objectInput.value.trim();
    if (!input) return [];

    return input.split(/[\n,;]+/)
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
        console.log('🌐 API URL:', `${API_BASE_URL}/api/v1/images/search/text`);

        const response = await fetch(`${API_BASE_URL}/api/v1/images/search/text`, {
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
        console.log('🌐 API URL:', `${API_BASE_URL}/api/v1/images/search/image`);

        const response = await fetch(`${API_BASE_URL}/api/v1/images/search/image`, {
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

// Helper to extract a friendly architectural landmark title
function getCleanImageTitle(result) {
    if (result.jpg_path) {
        const filename = result.jpg_path.split('/').pop();
        const stem = filename.substring(0, filename.lastIndexOf('.')) || filename;
        if (stem && !/^\d+$/.test(stem)) {
            return decodeURIComponent(stem).replace(/_/g, ' ');
        }
    }
    if (result.original_id && !result.original_id.startsWith('video_') && !/^\d+$/.test(result.original_id)) {
        return decodeURIComponent(result.original_id).replace(/_/g, ' ');
    }
    if (result.video_id && !result.video_id.startsWith('chan_') && !result.video_id.startsWith('L21_') && !result.video_id.startsWith('Album_ARCH_')) {
        return result.video_id;
    }
    return `Ảnh tư liệu #${result.rank}`;
}

// Helper to extract clean original object labels from YOLOv8 results
function getFormattedObjectLabels(objects, maxCount = null) {
    if (!objects || !Array.isArray(objects) || objects.length === 0) {
        return [];
    }

    const uniqueLabels = [];
    const seen = new Set();

    for (const item of objects) {
        let rawLabel = '';
        if (typeof item === 'string') {
            rawLabel = item.trim();
        } else if (item && typeof item === 'object') {
            rawLabel = (item.label || item.name || item.class || '').trim();
        }

        if (rawLabel) {
            const lower = rawLabel.toLowerCase();
            if (!seen.has(lower)) {
                seen.add(lower);
                uniqueLabels.push(rawLabel);
            }
        }
    }

    if (maxCount && maxCount > 0) {
        return uniqueLabels.slice(0, maxCount);
    }
    return uniqueLabels;
}

// Create a grid item for a result
function createGridItem(result) {
    const gridItem = document.createElement('div');
    gridItem.className = 'grid-item';

    // Build image path - prioritize image_url, fall back to backend redirect endpoint
    let imageUrl = result.image_url || result.jpg_path;
    if (!imageUrl.startsWith('http')) {
        const cleanPath = imageUrl.replace(/^\//, '');
        imageUrl = `${API_BASE_URL}/api/v1/images/keyframes/${cleanPath}`;
    }

    const title = getCleanImageTitle(result);
    const detectedLabels = getFormattedObjectLabels(result.objects, 3);
    const objText = detectedLabels.length > 0 ? `✨ ${detectedLabels.join(', ')}` : `🏛️ Toàn cảnh`;

    gridItem.innerHTML = `
        <div class="grid-item-image" style="background-image: url('${imageUrl}')"
             onerror="this.classList.add('error'); this.innerHTML='Không thể tải ảnh';">
            <div class="rank-badge">#${result.rank}</div>
        </div>
        <div class="grid-item-info">
            <div class="grid-item-title" title="${title}">🏛️ ${title}</div>
            <div class="grid-item-score">🎯 AI Score: ${(result.similarity_score * 100).toFixed(1)}%</div>
            <div class="grid-item-time" style="color: #38bdf8; font-size: 11px; margin-top: 4px;">${objText}</div>
            <div class="grid-item-frame" style="color: #64748b; font-size: 10px;">Thứ hạng: #${result.rank}</div>
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
                <span class="stats-label">Tổng số ảnh:</span>
                <span class="stats-value">${data.total_results}</span>
            </div>
            <div class="stats-item">
                <span class="stats-label">Thời gian:</span>
                <span class="stats-value">${Math.round(data.query_time_ms)}ms</span>
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

    // Suggestion chips for architectural elements
    document.querySelectorAll('.sugg-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const filterVal = chip.getAttribute('data-filter');
            const currentObjs = getObjectsFromInput();
            const idx = currentObjs.findIndex(o => o.toLowerCase() === filterVal.toLowerCase());
            
            if (idx >= 0) {
                currentObjs.splice(idx, 1);
                chip.classList.remove('active');
            } else {
                currentObjs.push(filterVal);
                chip.classList.add('active');
            }
            objectInput.value = currentObjs.join('\n');
        });
    });


    // Tab switching event listeners
    if (searchTabBtn && ingestTabBtn) {
        searchTabBtn.addEventListener('click', () => {
            searchTabBtn.classList.add('active');
            ingestTabBtn.classList.remove('active');
            if (searchTabContent) searchTabContent.style.display = 'block';
            if (ingestTabContent) ingestTabContent.style.display = 'none';
        });

        ingestTabBtn.addEventListener('click', () => {
            ingestTabBtn.classList.add('active');
            searchTabBtn.classList.remove('active');
            if (ingestTabContent) ingestTabContent.style.display = 'block';
            if (searchTabContent) searchTabContent.style.display = 'none';
        });
    }

    if (goToSearchAfterIngestBtn && searchTabBtn) {
        goToSearchAfterIngestBtn.addEventListener('click', () => {
            searchTabBtn.click();
        });
    }

    // Ingest file select count hint updater
    if (ingestFilesInput) {
        ingestFilesInput.addEventListener('change', (e) => {
            const count = e.target.files.length;
            if (ingestFilesCountHint) {
                if (count > 0) {
                    ingestFilesCountHint.innerHTML = `✅ <strong>Đã chọn:</strong> ${count} tệp ảnh kiến trúc sẵn sàng xử lý.`;
                } else {
                    ingestFilesCountHint.innerHTML = `💡 <strong>Mẹo:</strong> Bạn có thể bôi đen hoặc dùng phím tắt <kbd>Ctrl + A</kbd> trong cửa sổ duyệt file để chọn cùng lúc nhiều tệp ảnh.`;
                }
            }
        });
    }

    // Start Ingest listener
    if (startIngestBtn) {
        startIngestBtn.addEventListener('click', handleStartIngestion);
    }

    // Initialize button states
    updateButtonStates();

    console.log('✅ App initialized successfully');
}

// Detail Modal Functions for Architectural Photography
function openCarouselModal(selectedResult) {
    console.log('Opening photo detail modal for:', selectedResult);
    try {
        const cleanTitle = selectedResult.monument_name || getCleanImageTitle(selectedResult);
        if (carouselTitle) {
            carouselTitle.textContent = `Chi tiết Công trình & Phân tích AI - ${cleanTitle}`;
        }
        const cleanPath = selectedResult.jpg_path.replace(/^\//, '');
        const imageUrl = `${API_BASE_URL}/api/v1/images/keyframes/${cleanPath}`;

        currentFrameImg.src = imageUrl;
        currentFrameImg.onerror = function() {
            this.style.opacity = '0.5';
            this.alt = 'Ảnh không tải được';
        };
        currentFrameImg.onload = function() {
            this.style.opacity = '1';
        };

        const fname = selectedResult.jpg_path ? selectedResult.jpg_path.split('/').pop() : selectedResult.original_id;
        if (framePosition) framePosition.textContent = cleanTitle;
        if (frameTime) frameTime.textContent = `Độ khớp: ${(selectedResult.similarity_score * 100).toFixed(1)}%`;
        if (frameId) frameId.textContent = decodeURIComponent(fname || 'N/A');

        // Card 2: Mô tả lịch sử kiến trúc
        const monumentDescEl = document.getElementById('monumentDesc');
        if (monumentDescEl) {
            monumentDescEl.textContent = selectedResult.description || `Tư liệu hình ảnh công trình kiến trúc ${cleanTitle}.`;
        }

        // Card 3: Thể loại & Phân loại
        const monumentCatsEl = document.getElementById('monumentCategories');
        if (monumentCatsEl) {
            const cats = selectedResult.categories || [];
            if (cats.length > 0) {
                monumentCatsEl.innerHTML = cats.map(c => `<span class="category-tag">${c}</span>`).join('');
            } else {
                monumentCatsEl.innerHTML = `<span class="category-tag">Kiến trúc Việt Nam</span>`;
            }
        }

        // Card 4: Cấu kiện kiến trúc nhận diện (YOLO-World)
        if (frameIdx) {
            if (selectedResult.objects && selectedResult.objects.length > 0) {
                frameIdx.innerHTML = selectedResult.objects.map(obj => {
                    const name = obj.label || obj.name || obj.display_name;
                    const conf = obj.confidence ? ` (${(obj.confidence * 100).toFixed(0)}%)` : '';
                    return `<span class="object-chip">${name}${conf}</span>`;
                }).join('');
            } else {
                frameIdx.innerHTML = `<span class="no-object-text">Toàn cảnh kiến trúc (Không phát hiện cấu kiện phụ)</span>`;
            }
        }

        // Card 5: Thông tin tư liệu (Niên đại & Tác giả)
        const monumentDateEl = document.getElementById('monumentDate');
        if (monumentDateEl) {
            monumentDateEl.textContent = selectedResult.date || 'Tư liệu lưu trữ';
        }
        const monumentAuthorEl = document.getElementById('monumentAuthor');
        if (monumentAuthorEl) {
            monumentAuthorEl.textContent = selectedResult.author || 'Kho Wikimedia Commons';
        }

        // Action: Open full original image
        const viewFullBtn = document.getElementById('viewFullImageBtn');
        if (viewFullBtn) {
            viewFullBtn.href = selectedResult.source_url || imageUrl;
        }

        // Action: Search similar landmark/architecture
        const searchSimilarBtn = document.getElementById('searchSimilarBtn');
        if (searchSimilarBtn) {
            searchSimilarBtn.onclick = () => {
                closeCarouselModal();
                if (queryInput) {
                    queryInput.value = cleanTitle;
                    if (textModeBtn) textModeBtn.click();
                    handleSearch();
                }
            };
        }

        if (prevFrame) prevFrame.style.display = 'none';
        if (nextFrame) nextFrame.style.display = 'none';

        carouselModal.style.display = 'flex';
    } catch (error) {
        console.error('Error in openCarouselModal:', error);
        showError(`Lỗi tải ảnh chi tiết: ${error.message}`);
    }
}

function closeCarouselModal() {
    console.log('🚪 Closing detail modal');
    carouselModal.style.display = 'none';
}

function showPreviousFrame() {
    // No-op for individual architectural photos
}

function showNextFrame() {
    // No-op for individual architectural photos
}

// ==========================================================================
// INGEST PIPELINE (ZERO-REDIS DIRECT FASTAPI -> MINIO & QDRANT)
// ==========================================================================

async function handleStartIngestion() {
    if (!ingestFilesInput || !ingestFilesInput.files || ingestFilesInput.files.length === 0) {
        alert("Vui lòng chọn ít nhất 1 tệp ảnh kiến trúc từ máy tính của bạn.");
        return;
    }

    const batchName = ingestBatchNameInput ? ingestBatchNameInput.value.trim() : "";
    const categories = ingestCategoriesInput ? ingestCategoriesInput.value.trim() : "";
    const description = ingestDescInput ? ingestDescInput.value.trim() : "";
    const files = ingestFilesInput.files;

    const formData = new FormData();
    formData.append("batch_name", batchName || "Bộ sưu tập Kiến trúc");
    if (categories) formData.append("categories", categories);
    if (description) formData.append("description", description);

    for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }

    // UI state: Loading
    startIngestBtn.disabled = true;
    startIngestBtn.textContent = "⌛ Đang xử lý AI & Lưu vào MinIO/Qdrant...";
    if (ingestWelcomeBox) ingestWelcomeBox.style.display = "none";
    if (ingestSuccessBox) ingestSuccessBox.style.display = "none";
    if (ingestProgressBox) ingestProgressBox.style.display = "block";
    if (ingestResultSummaryBadge) ingestResultSummaryBadge.style.display = "none";

    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/images/ingest/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.text();
            throw new Error(err || "Lỗi nạp ảnh lên hệ thống.");
        }

        const data = await response.json();
        console.log("Upload response:", data);

        // Show success
        if (ingestProgressBox) ingestProgressBox.style.display = "none";
        if (ingestSuccessBox) ingestSuccessBox.style.display = "block";
        if (ingestSuccessMsg) ingestSuccessMsg.textContent = `🎉 Nạp thành công ${data.uploaded_count} ảnh vào hệ thống!`;
        if (ingestSuccessDetail) ingestSuccessDetail.textContent = `Mã mẻ ảnh: ${data.batch_id} • Đã lưu trữ ảnh gốc lên MinIO S3 và đánh chỉ mục vector vào Qdrant.`;

        if (ingestResultSummaryBadge) {
            ingestResultSummaryBadge.style.display = "inline-block";
            ingestResultSummaryBadge.textContent = `${data.uploaded_count} ảnh đã nạp`;
        }

        // Render detected chips
        if (ingestDetectedChips) {
            const labels = data.all_detected_objects || [];
            if (labels.length > 0) {
                ingestDetectedChips.innerHTML = labels.map(l => `<span class="category-tag" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);">${l}</span>`).join(' ');
            } else {
                ingestDetectedChips.innerHTML = `<span class="no-object-text">Toàn cảnh kiến trúc (Không phát hiện cấu kiện phụ)</span>`;
            }
        }

        // Reset file input
        ingestFilesInput.value = "";
        if (ingestFilesCountHint) {
            ingestFilesCountHint.innerHTML = `💡 <strong>Mẹo:</strong> Bạn có thể bôi đen hoặc dùng phím tắt <kbd>Ctrl + A</kbd> trong cửa sổ duyệt file để chọn cùng lúc nhiều tệp ảnh.`;
        }

    } catch (error) {
        console.error("Ingest error:", error);
        alert(`Lỗi nạp ảnh: ${error.message}`);
        if (ingestProgressBox) ingestProgressBox.style.display = "none";
        if (ingestWelcomeBox) ingestWelcomeBox.style.display = "block";
    } finally {
        startIngestBtn.disabled = false;
        startIngestBtn.textContent = "⚡ Phân tích AI & Lưu Vào Kho";
    }
}

// Start the app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
