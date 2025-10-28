#!/usr/bin/env python3
"""flask-based web interface for the MapRAS."""

import os
import csv
import requests
from functools import lru_cache
import traceback

from flask import Flask, render_template, request, jsonify, send_from_directory

from src.tool import VisualSearchTool

app = Flask(__name__)

# global stuff
search_tool = None
metadata_cache = {}  # file_url -> {'resource_url': str, 'segment_number': int}
citation_cache = {}  # (resource_url, segment_number) -> citation string

# ollama configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"  # can change to llama3.2:3b for better but slower.
# may add others later

class OllamaClient:
	"""ollama api client"""
	
	def __init__(self, api_url=OLLAMA_API_URL, model=OLLAMA_MODEL):
		self.api_url = api_url
		self.model = model
		self.available = self._check_availability()
	
	def _check_availability(self):
		"""check if ollama running."""
		try:
			response = requests.get("http://localhost:11434/api/tags", timeout=2)
			return response.status_code == 200
		except:
			return False
	
	def generate(self, prompt, stream=False):
		"""respond to a prompt"""
		if not self.available:
			raise Exception("Ollama is not running. Start it with: ollama serve")
		
		try:
			payload = {
				"model": self.model,
				"prompt": prompt,
				"stream": stream,
				"options": {
					"temperature": 0.7,
					"num_predict": 500
				}
			}
			
			response = requests.post(self.api_url, json=payload, timeout=60)
			response.raise_for_status()
			
			if stream:
				return response.iter_lines()
			else:
				return response.json()['response']
		except Exception as e:
			raise Exception(f"Ollama API error: {str(e)}")

ollama_client = OllamaClient()

def load_metadata_csv():
	"""load the merged_files into memory for quick lookups. it is used often"""
	global metadata_cache
	csv_path = os.path.expanduser('merged_files.csv')
	
	if not os.path.exists(csv_path):
		print(f"Warning: {csv_path} not found. Citations will not be available.")
		return
	
	try:
		with open(csv_path, 'r', encoding='utf-8') as f:
			reader = csv.DictReader(f)
			for row in reader:
				file_url = row.get('file_url', '').strip()
				resource_url = row.get('resource_url', '').strip()
				segment_number = row.get('segment_num', '').strip()
				
				if file_url and resource_url and segment_number:
					metadata_cache[file_url] = {
						'resource_url': resource_url,
						'segment_number': int(segment_number)+1
					}
		
		print(f"Loaded {len(metadata_cache)} metadata entries from CSV")
	except Exception as e:
		print(f"Error loading metadata CSV: {e}")

def convert_display_url_to_csv_format(display_url):
	"""display URL is auto '1500,' but csv uses 'pct:100'"""
	if '1500,' in display_url:
		return display_url.replace('1500,', 'pct:100')
	return display_url

@lru_cache(maxsize=1000)
def fetch_chicago_citation(resource_url, segment_number):
	"""chicago citation from the resource api"""
	global citation_cache
	
	cache_key = (resource_url, segment_number)
	if cache_key in citation_cache:
		return citation_cache[cache_key]
	
	try:
		api_url = f'{resource_url}?fo=json&sp={segment_number}'
		
		headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
		response = requests.get(api_url, timeout=10, headers=headers)
		response.raise_for_status()
		
		data = response.json()
		
		cite_this = data.get('cite_this', {})
		chicago_citation = cite_this.get('chicago', None)
		
		if chicago_citation:
			citation_cache[cache_key] = chicago_citation
			return chicago_citation
		else:
			print(f"No Chicago citation found for {api_url}")
			return None
			
	except Exception as e:
		print(f"Error fetching citation from {resource_url}: {e}")
		return None

@lru_cache(maxsize=1000)
def fetch_full_metadata(resource_url, segment_number):
	"""get full metadata from the resource AapiPI for llm"""
	try:
		api_url = f'{resource_url}?fo=json&sp={segment_number}'
		headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
		response = requests.get(api_url, timeout=10, headers=headers)
		response.raise_for_status()
		
		data = response.json()
		item = data.get('item')
		subitem = item.get('item')
		
		# relevant fields (this is overkill)
		metadata = {
			'descriptions': item.get('description', []),
			'title': subitem.get('title', 'Unknown'),
			'date': subitem.get('date', 'Unknown'),
			'creator': subitem.get('contributors', []),
			'type': subitem.get('format', []),
			'location': subitem.get('location', []),
			'subject': subitem.get('subject', []),
		}
		
		return metadata
		
	except Exception as e:
		print(f"Error fetching metadata from {resource_url}: {e}")
		return None

def add_citation(result):
	"""add chicago citation and real title to result if it exists"""
	try:
		csv_url = convert_display_url_to_csv_format(result['source'])
		metadata = metadata_cache.get(csv_url)
		
		if metadata:
			result['resource_url'] = metadata['resource_url']
			result['segment_number'] = metadata['segment_number']
			
			# fetch citation
			citation = fetch_chicago_citation(
				metadata['resource_url'],
				metadata['segment_number']
			)
			
			# fetch full metadata for actual title
			full_metadata = fetch_full_metadata(
				metadata['resource_url'],
				metadata['segment_number']
			)
			
			if citation:
				result['chicago_citation'] = citation
				result['has_citation'] = True
			else:
				result['has_citation'] = False
			
			# use actual title from metadata if available
			if full_metadata and full_metadata.get('title'):
				result['title'] = full_metadata['title']
		else:
			result['has_citation'] = False
			
	except Exception as e:
		print(f"Error enriching result: {e}")
		result['has_citation'] = False

	return result

def analyze_search_results(query, results):
	"""analyze search results using llm + metadata"""
	if not ollama_client.available:
		return {
			'error': 'Ollama is not running. Start it with: ollama serve',
			'instructions': 'Install Ollama from https://ollama.ai and run: ollama pull llama3.2:3b'
		}
	
	if not results:
		return {'analysis': 'No results to analyze.'}
	
	# gather metadata for results with citations
	metadata_list = []
	for result in results:
		if 'resource_url' in result and 'segment_number' in result:
			metadata = fetch_full_metadata(
				result['resource_url'],
				result['segment_number']
			)
			if metadata:
				metadata['rank'] = result['rank']
				metadata['score'] = result['score']
				metadata_list.append(metadata)
		else:
			print(f"No resource URL/segment for result rank {result['rank']}, skipping metadata fetch.")
	
	if not metadata_list:
		return {'analysis': 'No metadata available for these results.'}
	
	prompt = f"""You are analyzing search results from the Library of Congress digital collections.

Search Query: "{query}"

Results ({len(metadata_list)} items with metadata):

"""
	
	for i, meta in enumerate(metadata_list, 1):
		prompt += f"\n--- Result {i} (Rank {meta['rank']}, Score: {meta['score']:.3f}) ---\n"
		prompt += f"Title: {meta['title']}\n"
		
		if meta.get('date'):
			prompt += f"Date: {meta['date']}\n"
		
		if meta.get('location'):
			locations = meta['location'] if isinstance(meta['location'], list) else [meta['location']]
			if locations:
				prompt += f"Location: {', '.join(str(l) for l in locations)}\n"
		
		if meta.get('creator'):
			creators = meta['creator'] if isinstance(meta['creator'], list) else [meta['creator']]
			if creators:
				prompt += f"Creator: {', '.join(str(c) for c in creators)}\n"
		
		if meta.get('subject'):
			subjects = meta['subject'] if isinstance(meta['subject'], list) else [meta['subject']]
			if subjects:
				prompt += f"Subjects: {', '.join(str(s) for s in subjects)}\n"
		
		# if meta.get('type'):
		# 	types = meta['type'] if isinstance(meta['type'], list) else [meta['type']]
		# 	if types:
		# 		prompt += f"Format: {', '.join(str(t) for t in types)}\n"
		
		if meta.get('descriptions'):
			descriptions = meta['descriptions'] if isinstance(meta['descriptions'], list) else [meta['descriptions']]
			if descriptions and descriptions[0]:
				# Truncate long descriptions
				desc = str(descriptions[0])
				prompt += f"Description: {desc}\n"
	
	prompt += """

Based on the metadata above, provide a very brief qualitative analysis that:
1. Identifies the main themes, time periods, and subject matter represented in these results
2. Notes any patterns in the types of materials (maps, photographs, documents, etc.)
3. Highlights key creators, locations, or topics that dominate the results

Write in a clear, informative style as if explaining these results to a researcher."""
	
	try:
		print(f"Sending prompt to Ollama ({len(prompt)} chars)...")
		print('prompt used: ', prompt)
		analysis_text = ollama_client.generate(prompt)
		return {
			'analysis': analysis_text.strip(),
			'results_analyzed': len(metadata_list),
			'total_results': len(results)
		}
		
	except Exception as e:
		return {'error': str(e)}


def get_search_tool():
	"""Get or create the search tool instance."""
	global search_tool
	if search_tool is None:
		search_tool = VisualSearchTool('./embeddings')
		
		load_metadata_csv()
		
		try: # first try distributed
			print("Loading distributed embeddings on startup...")
			search_tool.load_distributed_corpus()
			print(f"Loaded {len(search_tool.embeddings)} embeddings")
		except Exception as e:
			print(f"Could not load distributed corpus on startup: {e}")
			try: # fallback to regular if distributed loading fails
				search_tool.load_corpus()
			except Exception as e2:
				print(f"No corpus loaded on startup: {e2}")
	return search_tool

@app.route('/')
def index():
	"""Serve the main HTML interface."""
	return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
	"""Serve images from the images directory."""
	images_dir = os.path.join(os.getcwd(), 'images')
	try:
		return send_from_directory(images_dir, filename)
	except FileNotFoundError:
		return "Image not found", 404

@app.route('/api/image/<int:doc_index>')
def serve_document_image(doc_index):
	"""Serve image from document source by index."""
	try:
		tool = get_search_tool()
		if doc_index < 0 or doc_index >= len(tool.documents):
			return "Document index out of range", 404
		
		doc = tool.documents[doc_index]
		source = doc['source']
		
		# if url: redirect
		if source.startswith('http://') or source.startswith('https://'):
			from flask import redirect
			return redirect(source)
		
		# if local file: serve directly
		if os.path.exists(source):
			directory = os.path.dirname(source)
			filename = os.path.basename(source)
			return send_from_directory(directory, filename)
		
		return "Image not found", 404
		
	except Exception as e:
		traceback.print_exc()
		return f"Error serving image: {str(e)}", 500

@app.route('/api/search', methods=['POST'])
def api_search():
	"""handle search requests"""
	try:
		data = request.get_json()
		query = data.get('query', '').strip()
		k = data.get('k', 5)

		if not query:
			return jsonify({'error': 'Query is required'}), 400

		tool = get_search_tool()
		results = tool.search(query, k=k)
		
		results_with_citations = [add_citation(result) for result in results]

		return jsonify({
			'success': True,
			'results': results_with_citations,
			'query': query,
			'k': k
		})

	except Exception as e:
		traceback.print_exc()
		return jsonify({'error': str(e)}), 500

@app.route('/api/search_image', methods=['POST'])
def api_search_image():
	try: 
		data = request.get_json()
		image_url = data.get('image_url')
		k = data.get('k', 5)

		tool = get_search_tool()
		image = tool.download_image(image_url)  # Already exists
		results = tool.search_by_image(image, k=k)
		results_with_citations = [add_citation(result) for result in results]

		return jsonify({
			'success': True,
			'results': results_with_citations,
			'query': image_url,
			'k': k
		})
	except Exception as e:
		traceback.print_exc()
		return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
	"""stats about embeddings"""
	try:
		tool = get_search_tool()
		stats = tool.get_stats()
		return jsonify(stats)
	except Exception as e:
		return jsonify({'error': str(e)}), 500

@app.route('/api/add_urls', methods=['POST'])
def api_add_urls():
	"""add images from urls"""
	try:
		data = request.get_json()
		urls = data.get('urls', [])
		titles = data.get('titles', [])

		if not urls:
			return jsonify({'error': 'URLs are required'}), 400

		tool = get_search_tool()
		tool.add_images_from_urls(urls, titles=titles if titles else None)

		return jsonify({
			'success': True,
			'count': len(urls),
			'message': f'Successfully processed {len(urls)} images'
		})

	except Exception as e:
		import traceback
		traceback.print_exc()
		return jsonify({'error': str(e)}), 500

@app.route('/api/add_files', methods=['POST'])
def api_add_files():
	"""add images from local files"""
	try:
		data = request.get_json()
		filepaths = data.get('filepaths', [])
		titles = data.get('titles', [])

		if not filepaths:
			return jsonify({'error': 'File paths are required'}), 400

		tool = get_search_tool()
		tool.add_images_from_files(filepaths, titles=titles if titles else None)

		return jsonify({
			'success': True,
			'count': len(filepaths),
			'message': f'Successfully processed {len(filepaths)} files'
		})

	except Exception as e:
		import traceback
		traceback.print_exc()
		return jsonify({'error': str(e)}), 500

@app.route('/api/add_pdf', methods=['POST'])
def api_add_pdf():
	"""add a pdf file"""
	try:
		data = request.get_json()
		pdf_path = data.get('pdf_path', '').strip()
		title_prefix = data.get('title_prefix', None)

		if not pdf_path:
			return jsonify({'error': 'PDF path is required'}), 400

		tool = get_search_tool()
		tool.add_pdf(pdf_path, title_prefix=title_prefix)

		return jsonify({
			'success': True,
			'message': f'Successfully processed PDF: {pdf_path}'
		})

	except Exception as e:
		import traceback
		traceback.print_exc()
		return jsonify({'error': str(e)}), 500

@app.route('/api/save', methods=['POST'])
def api_save():
	"""save corpus to file"""
	try:
		tool = get_search_tool()
		tool.save_corpus()

		stats = tool.get_stats()
		return jsonify({
			'success': True,
			'message': f'Saved {stats["total_embeddings"]} embeddings to corpus'
		})

	except Exception as e:
		import traceback
		traceback.print_exc()
		return jsonify({'error': str(e)}), 500

@app.route('/api/load', methods=['POST'])
def api_load():
	"""load corpus from file"""
	try:
		tool = get_search_tool()
		tool.load_corpus()

		stats = tool.get_stats()
		return jsonify({
			'success': True,
			'total_embeddings': stats['total_embeddings'],
			'message': f'Loaded {stats["total_embeddings"]} embeddings'
		})

	except Exception as e:
		import traceback
		traceback.print_exc()
		return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
	"""health check endpt"""
	return jsonify({
		'status': 'healthy',
		'service': 'ColPali Visual Search API'
	})

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
	"""analyze search results w/ llm"""
	try:
		data = request.get_json()
		query = data.get('query', '')
		results = data.get('results', [])
		
		if not results:
			return jsonify({'error': 'No results provided'}), 400
		
		analysis_result = analyze_search_results(query, results)
		
		if 'error' in analysis_result:
			return jsonify(analysis_result), 500
		
		return jsonify({
			'success': True,
			**analysis_result
		})
		
	except Exception as e:
		import traceback
		traceback.print_exc()
		return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
	"""handle 404"""
	if request.path.startswith('/api/'):
		return jsonify({'error': 'API endpoint not found'}), 404
	return render_template('index.html')

@app.errorhandler(500)
def internal_error(error):
	"""handle 500"""
	if request.path.startswith('/api/'):
		return jsonify({'error': 'Internal server error'}), 500
	return "Internal server error", 500

@app.after_request
def add_ngrok_header(response):
    """ngrok header to bypass browser warning (doesn't work)"""
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

def main():
	"""run flask app"""
	print("Starting ColPali Visual Search Web Interface...")
	print("=" * 50)
	print("Features available:")
	print("• Visual search with query text")
	print("• Add images from URLs")
	print("• Add images from local files")
	print("• Add PDFs")
	print("• Load/save corpus")
	print("• View statistics")
	print("=" * 50)
	print("\nNote: Distributed embeddings (batch_*.pkl) are loaded automatically on startup")
	print("\nStarting server...")
	print("Access the web interface at: http://localhost:5001")
	print("API endpoints available at: http://localhost:5001/api/")
	print("\nPress Ctrl+C to stop the server")
	app.run(host='0.0.0.0', port=5001, debug=True)

if __name__ == '__main__':
	main()