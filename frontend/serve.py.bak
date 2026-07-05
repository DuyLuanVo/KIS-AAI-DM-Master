#!/usr/bin/env python3
"""
Simple HTTP Server để serve frontend
Chạy: python serve.py
"""
import http.server
import os
import socketserver
import webbrowser
from pathlib import Path

# Cấu hình
PORT = 3000
DIRECTORY = Path(__file__).parent


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request Handler với CORS headers"""

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def translate_path(self, path):
        # Maps requests for /data/... to the actual project-level data folder
        if path.startswith('/data/'):
            data_dir = Path(__file__).resolve().parent.parent / "data"
            relative_path = path[len('/data/'):]
            # Clean URL decoding for path
            import urllib.parse
            decoded_path = urllib.parse.unquote(relative_path)
            return str(data_dir / decoded_path.replace('/', os.sep))
        return super().translate_path(path)


def main():
    # Chuyển đến thư mục frontend
    os.chdir(DIRECTORY)

    # Tạo server
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"🌐 Serving frontend at http://localhost:{PORT}")
        print(f"📁 Directory: {DIRECTORY}")

        # Mở browser
        webbrowser.open(f'http://localhost:{PORT}')

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")


if __name__ == "__main__":
    main()
