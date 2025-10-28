# MapRAS: Retrieval-Augmented Search with ColPali

## The current link can be found at: https://fb1f72e6cf8d.ngrok-free.app

A visual search system for document retrieval using ColPali embeddings. Designed for searching large collections of historical documents and maps from the Library of Congress digital archives. 

## Features
- **Visual semantic search** - Query documents using natural language
- **Multiple input formats** - Search URLs, local files, and PDFs
- **Distributed embeddings** - Handle large corpora split across multiple files
- **Citation integration** - Automatic Chicago-style citations from Library of Congress metadata
- **LLM analysis** - Optional result analysis using local Ollama models
- **Web interface** - Flask-based UI for search and corpus management

## Requirements

### Installation
```
pip install colpali-engine flask pandas pillow pdf2image requests
pip install git+https://github.com/illuin-tech/colpali
```

### LLM analysis
```
# Install Ollama from https://ollama.ai
ollama pull llama3.2:1b
```

## Usage

### Starting the web interface
```
python app.py
```
Access at `http://localhost:5001`.
The application automatically loads embeddings from `./embeddings/batch_*.pkl` files on startup. Download at [LINK ZENODO??]

### Creating embeddings
To process a dataset
```
python create_dataset.py
```
This expects `merged_files.csv` with columns: file_url, resource_url, segment_num, p1_item_id, file_count. See Zenodo.
