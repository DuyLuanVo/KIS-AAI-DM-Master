// Mock API Server for testing frontend
const http = require('http');
const url = require('url');

// Mock data
const mockResults = [
    {
        original_id: "video_001",
        jpg_path: "https://picsum.photos/300/200?random=1",
        frame_idx: 123,
        similarity_score: 0.95
    },
    {
        original_id: "video_002",
        jpg_path: "https://picsum.photos/300/200?random=2",
        frame_idx: 45,
        similarity_score: 0.92
    },
    {
        original_id: "video_001",
        jpg_path: "https://picsum.photos/300/200?random=3",
        frame_idx: 156,
        similarity_score: 0.90
    },
    {
        original_id: "video_003",
        jpg_path: "https://picsum.photos/300/200?random=4",
        frame_idx: 78,
        similarity_score: 0.88
    },
    {
        original_id: "video_002",
        jpg_path: "https://picsum.photos/300/200?random=5",
        frame_idx: 234,
        similarity_score: 0.85
    },
    {
        original_id: "video_004",
        jpg_path: "https://picsum.photos/300/200?random=6",
        frame_idx: 89,
        similarity_score: 0.83
    },
    {
        original_id: "video_001",
        jpg_path: "https://picsum.photos/300/200?random=7",
        frame_idx: 189,
        similarity_score: 0.81
    },
    {
        original_id: "video_003",
        jpg_path: "https://picsum.photos/300/200?random=8",
        frame_idx: 112,
        similarity_score: 0.79
    },
    {
        original_id: "video_005",
        jpg_path: "https://picsum.photos/300/200?random=9",
        frame_idx: 67,
        similarity_score: 0.77
    },
    {
        original_id: "video_002",
        jpg_path: "https://picsum.photos/300/200?random=10",
        frame_idx: 145,
        similarity_score: 0.75
    },
    {
        original_id: "video_004",
        jpg_path: "https://picsum.photos/300/200?random=11",
        frame_idx: 203,
        similarity_score: 0.73
    },
    {
        original_id: "video_001",
        jpg_path: "https://picsum.photos/300/200?random=12",
        frame_idx: 267,
        similarity_score: 0.71
    },
    {
        original_id: "video_003",
        jpg_path: "https://picsum.photos/300/200?random=13",
        frame_idx: 34,
        similarity_score: 0.69
    },
    {
        original_id: "video_005",
        jpg_path: "https://picsum.photos/300/200?random=14",
        frame_idx: 178,
        similarity_score: 0.67
    },
    {
        original_id: "video_002",
        jpg_path: "https://picsum.photos/300/200?random=15",
        frame_idx: 92,
        similarity_score: 0.65
    },
    {
        original_id: "video_004",
        jpg_path: "https://picsum.photos/300/200?random=16",
        frame_idx: 156,
        similarity_score: 0.63
    },
    {
        original_id: "video_001",
        jpg_path: "https://picsum.photos/300/200?random=17",
        frame_idx: 223,
        similarity_score: 0.61
    },
    {
        original_id: "video_003",
        jpg_path: "https://picsum.photos/300/200?random=18",
        frame_idx: 78,
        similarity_score: 0.59
    },
    {
        original_id: "video_005",
        jpg_path: "https://picsum.photos/300/200?random=19",
        frame_idx: 134,
        similarity_score: 0.57
    },
    {
        original_id: "video_002",
        jpg_path: "https://picsum.photos/300/200?random=20",
        frame_idx: 189,
        similarity_score: 0.55
    },
    {
        original_id: "video_006",
        jpg_path: "https://picsum.photos/300/200?random=21",
        frame_idx: 45,
        similarity_score: 0.53
    },
    {
        original_id: "video_004",
        jpg_path: "https://picsum.photos/300/200?random=22",
        frame_idx: 167,
        similarity_score: 0.51
    },
    {
        original_id: "video_001",
        jpg_path: "https://picsum.photos/300/200?random=23",
        frame_idx: 289,
        similarity_score: 0.49
    },
    {
        original_id: "video_003",
        jpg_path: "https://picsum.photos/300/200?random=24",
        frame_idx: 112,
        similarity_score: 0.47
    },
    {
        original_id: "video_005",
        jpg_path: "https://picsum.photos/300/200?random=25",
        frame_idx: 203,
        similarity_score: 0.45
    }
];

// Create HTTP server
const server = http.createServer((req, res) => {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    // Handle preflight requests
    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }
    
    const parsedUrl = url.parse(req.url, true);
    
    // Handle video search endpoint
    if (req.method === 'POST' && parsedUrl.pathname === '/api/v1/videos/search') {
        let body = '';
        
        req.on('data', chunk => {
            body += chunk.toString();
        });
        
        req.on('end', () => {
            try {
                const requestData = JSON.parse(body);
                console.log('Received search request:', requestData);
                console.log('Query text:', requestData.query_text);
                console.log('List object:', requestData.list_object);
                
                // Simulate some processing time
                setTimeout(() => {
                    // Return mock results
                    const response = {
                        results: mockResults
                    };
                    
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify(response));
                }, 500); // 500ms delay to simulate API processing
                
            } catch (error) {
                console.error('Error parsing request:', error);
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Invalid JSON' }));
            }
        });
        
        return;
    }
    
    // Handle health check
    if (req.method === 'GET' && parsedUrl.pathname === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'healthy', message: 'Mock server is running' }));
        return;
    }
    
    // Handle root path
    if (req.method === 'GET' && parsedUrl.pathname === '/') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(`
            <html>
                <head><title>Mock API Server</title></head>
                <body>
                    <h1>Mock API Server</h1>
                    <p>Server is running on port 8000</p>
                    <p>Available endpoints:</p>
                    <ul>
                        <li>POST /api/v1/videos/search - Video search</li>
                        <li>GET /health - Health check</li>
                    </ul>
                </body>
            </html>
        `);
        return;
    }
    
    // 404 for unknown routes
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
});

const PORT = 8000;

server.listen(PORT, () => {
    console.log(`Mock API server running on http://localhost:${PORT}`);
    console.log('Available endpoints:');
    console.log('  POST /api/v1/videos/search - Video search');
    console.log('  GET /health - Health check');
    console.log('  GET / - Server info');
});

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down mock server...');
    server.close(() => {
        console.log('Mock server stopped');
        process.exit(0);
    });
});
