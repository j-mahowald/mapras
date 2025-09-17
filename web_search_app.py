#!/usr/bin/env python3
"""
Web interface for the ColPali Embedding Search Tool.
Flask application that provides a REST API for the search functionality.
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from embedding_search_tool import EmbeddingSearchTool

app = Flask(__name__)

# Global search tool instance
search_tool = None

def get_search_tool():
	"""Get or create the search tool instance."""
	global search_tool
	if search_tool is None:
		search_tool = EmbeddingSearchTool()
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
		# Return a placeholder or 404
		return "Image not found", 404

@app.route('/api/search', methods=['POST'])
def api_search():
	"""Handle search requests."""
	try:
		data = request.get_json()
		query = data.get('query', '').strip()
		k = data.get('k', 5)
		use_merged = data.get('use_merged', False)

		if not query:
			return jsonify({'error': 'Query is required'}), 400

		tool = get_search_tool()

		if use_merged:
			results = tool.search_merged_embeddings(query, k=k)
		else:
			results = tool.search_original_embeddings(query, k=k)

		return jsonify({
			'success': True,
			'results': results,
			'query': query,
			'k': k,
			'use_merged': use_merged
		})

	except Exception as e:
		return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
	"""Get statistics about embeddings."""
	try:
		tool = get_search_tool()
		stats = tool.get_stats()
		return jsonify(stats)
	except Exception as e:
		return jsonify({'error': str(e)}), 500

@app.route('/api/add_urls', methods=['POST'])
def api_add_urls():
	"""Add images from URLs."""
	try:
		data = request.get_json()
		urls = data.get('urls', [])
		titles = data.get('titles', [])

		if not urls:
			return jsonify({'error': 'URLs are required'}), 400

		tool = get_search_tool()
		new_data = tool.add_images_from_urls(urls, titles=titles if titles else None)

		return jsonify({
			'success': True,
			'count': len(new_data),
			'message': f'Successfully processed {len(new_data)} images'
		})

	except Exception as e:
		return jsonify({'error': str(e)}), 500

@app.route('/api/merge', methods=['POST'])
def api_merge():
	"""Create merged embeddings."""
	try:
		tool = get_search_tool()
		success = tool.create_merged_embeddings()

		if success:
			stats = tool.get_stats()
			return jsonify({
				'success': True,
				'total_embeddings': stats['merged_embeddings'],
				'message': f'Created merged embeddings with {stats["merged_embeddings"]} total embeddings'
			})
		else:
			return jsonify({'error': 'Failed to create merged embeddings'}), 500

	except Exception as e:
		return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
	"""Health check endpoint."""
	return jsonify({
		'status': 'healthy',
		'service': 'ColPali Visual Search API'
	})

@app.errorhandler(404)
def not_found(error):
	"""Handle 404 errors."""
	if request.path.startswith('/api/'):
		return jsonify({'error': 'API endpoint not found'}), 404
	return render_template('index.html')

@app.errorhandler(500)
def internal_error(error):
	"""Handle 500 errors."""
	if request.path.startswith('/api/'):
		return jsonify({'error': 'Internal server error'}), 500
	return "Internal server error", 500

def main():
	"""Run the Flask application."""
	print("Starting ColPali Visual Search Web Interface...")
	print("=" * 50)
	print("Features available:")
	print("• Visual search with query text")
	print("• Add images from URLs")
	print("• Create and search merged embeddings")
	print("• View statistics")
	print("• Image preview in results")
	print("=" * 50)

	# Check if images directory exists
	images_dir = os.path.join(os.getcwd(), 'images')
	if os.path.exists(images_dir):
		image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
		print(f"Found {image_count} images in ./images directory")
	else:
		print("Warning: ./images directory not found - image preview may not work")
		os.makedirs(images_dir, exist_ok=True)

	print("\nStarting server...")
	print("Access the web interface at: http://localhost:5001")
	print("API endpoints available at: http://localhost:5001/api/")
	print("\nPress Ctrl+C to stop the server")

	# Run the Flask app
	app.run(host='0.0.0.0', port=5001, debug=True)

if __name__ == '__main__':
	main()