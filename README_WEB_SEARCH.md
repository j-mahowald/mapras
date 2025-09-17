# ColPali Visual Search Web Interface

A minimal HTML-based search tool that provides the same functionalities as `embedding_search_tool.py` through a web interface.

## Features

- **Visual Search**: Search existing embeddings with text queries
- **Image Display**: Shows corresponding images from the `images/` folder
- **URL Addition**: Add new images from URLs
- **Merged Embeddings**: Create and search combined original + URL embeddings
- **Statistics**: View embedding counts and file status
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

1. **Start the web server:**
   ```bash
   python web_search_app.py
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:5001
   ```

3. **Start searching!** Try queries like:
   - "city skyline"
   - "mountain landscape" 
   - "ocean view"
   - "forest trees"

## Usage Guide

### Basic Search
1. Enter a search query in the text box
2. Choose number of results (1-20)
3. Click "Search" to find similar images
4. Results show images with similarity scores

### Advanced Features

**Add Images from URLs:**
1. Paste image URLs (one per line) in the URL section
2. Optionally add titles (one per line)
3. Click "Add URLs" to download and process images

**Create Merged Embeddings:**
1. After adding URL images, click "Create Merged Embeddings"
2. This combines original embeddings with new URL images
3. Check "Use Merged Embeddings" in search to include URL images

**View Statistics:**
- Click "Show Stats" to see embedding counts
- Shows original, merged, and file existence status

### Image Display
- Results automatically display images from the `images/` folder
- If an image file is not found, a placeholder message appears
- Images are automatically resized to fit the interface

## API Endpoints

The web interface uses these REST API endpoints:

- `POST /api/search` - Search embeddings
- `GET /api/stats` - Get statistics  
- `POST /api/add_urls` - Add images from URLs
- `POST /api/merge` - Create merged embeddings
- `GET /api/health` - Health check

## File Structure

```
├── web_search_app.py          # Flask backend application
├── templates/
│   └── index.html            # Frontend HTML interface
├── embedding_search_tool.py   # Core search functionality
├── images/                   # Image files directory
└── visual_corpus/            # Embeddings and metadata
```

## Requirements

- Python 3.7+
- Flask
- All dependencies from `embedding_search_tool.py`

## Notes

- The web interface automatically filters out corrupted embeddings (NaN values)
- Images are served from the `images/` directory relative to the script location
- The server runs in debug mode for development - not suitable for production
- Default port is 5001 (change in `web_search_app.py` if needed)

## Troubleshooting

**Port in use error:**
- Change the port in `web_search_app.py` (line with `app.run()`)
- On macOS, disable AirPlay Receiver if using port 5000

**Images not displaying:**
- Ensure the `images/` directory exists and contains your image files
- Check that image filenames match the titles in search results
- Image files should be in standard formats (JPG, PNG, etc.)

**Search returning same results:**
- This was fixed by filtering out corrupted embeddings
- The web interface uses the improved search functionality

Enjoy exploring your visual corpus with the web interface! 🔍