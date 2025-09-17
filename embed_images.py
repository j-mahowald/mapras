#!/usr/bin/env python3

import os
import glob
import json
from tool import VisualSearchTool

def load_all_images(url_mapping_file, embeddings_file='embeddings.pkl'):
	"""Load all images from the images directory into the RAG system."""

	# Initialize the tool with a specific corpus path
	tool = VisualSearchTool(corpus_path="./visual_corpus", embeddings_file=embeddings_file)

	# Initialize the tool
	tool.setup(load_existing=False)

	# Load URL mapping if it exists
	url_mapping = {}
	if url_mapping_file and os.path.exists(url_mapping_file):
		try:
			with open(url_mapping_file, 'r') as f:
				url_mapping = json.load(f)
			print(f"Loaded URL mapping for {len(url_mapping)} images")
		except Exception as e:
			print(f"Warning: Could not load URL mapping: {e}")

	# Get all image files from the images directory
	images_dir = "./images"
	image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]

	image_paths = []
	for ext in image_extensions:
		image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
		image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))

	# Sort for consistent ordering
	image_paths.sort()

	print(f"Found {len(image_paths)} images to process")

	if not image_paths:
		print("No images found in the images directory!")
		return

	# Process images in batches to avoid memory issues
	batch_size = 10  # Adjust based on your system's memory

	for i in range(0, len(image_paths), batch_size):
		batch_paths = image_paths[i:i + batch_size]
		batch_num = i // batch_size + 1
		total_batches = (len(image_paths) + batch_size - 1) // batch_size

		print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_paths)} images)")

		# Get URLs for this batch
		batch_urls = []
		for path in batch_paths:
			filename = os.path.basename(path)
			url = url_mapping.get(filename)
			batch_urls.append(url)

		try:
			# Add images with auto-generated titles based on filename and original URLs
			tool.add_images(batch_paths, original_urls=batch_urls, auto_save=False)
			# Save to file after each batch
			tool.save_corpus()
			print(f"Batch {batch_num} saved to file.")
		except Exception as e:
			print(f"Error processing batch {batch_num}: {e}")
			# Continue with the next batch
			continue

	# Print final database info
	info = tool.get_database_info()
	print(f"\n✅ Successfully loaded {info['total_documents']} images into the RAG system")
	print(f"Corpus saved to: {tool.corpus_path}")

	return tool

def load_multiple_mappings():
	"""Load all images from multiple URL mapping files into the RAG system."""
	# Initialize the tool with a specific corpus path
	tool = VisualSearchTool(corpus_path="./visual_corpus", embeddings_file='embeddings.pkl')

	# Clear any existing database to start fresh
	tool.clear_database(remove_files=True)

	# Initialize the tool
	tool.setup(load_existing=False)

	# List of URL mapping files to process
	url_mapping_files = ['url_mapping.json', 'url_mapping_2000.json', 'url_mapping_2500.json']

	# Combined URL mapping from all files
	combined_url_mapping = {}

	for mapping_file in url_mapping_files:
		if os.path.exists(mapping_file):
			try:
				with open(mapping_file, 'r') as f:
					mapping = json.load(f)
				combined_url_mapping.update(mapping)
				print(f"Loaded {len(mapping)} URLs from {mapping_file}")
			except Exception as e:
				print(f"Warning: Could not load {mapping_file}: {e}")

	print(f"Combined URL mapping has {len(combined_url_mapping)} total images")

	# Get all image files from the images directory
	images_dir = "./images"
	image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]

	image_paths = []
	for ext in image_extensions:
		image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
		image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))

	# Sort for consistent ordering
	image_paths.sort()

	print(f"Found {len(image_paths)} images to process")

	if not image_paths:
		print("No images found in the images directory!")
		return

	# Process images in batches to avoid memory issues
	batch_size = 10  # Adjust based on your system's memory

	for i in range(0, len(image_paths), batch_size):
		batch_paths = image_paths[i:i + batch_size]
		batch_num = i // batch_size + 1
		total_batches = (len(image_paths) + batch_size - 1) // batch_size

		print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_paths)} images)")

		# Get URLs for this batch
		batch_urls = []
		for path in batch_paths:
			filename = os.path.basename(path)
			url = combined_url_mapping.get(filename)
			batch_urls.append(url)

		try:
			# Add images with auto-generated titles based on filename and original URLs
			tool.add_images(batch_paths, original_urls=batch_urls, auto_save=False)
			# Save to file after each batch
			tool.save_corpus()
			print(f"Batch {batch_num} saved to file.")
		except Exception as e:
			print(f"Error processing batch {batch_num}: {e}")
			# Continue with the next batch
			continue

	# Print final database info
	info = tool.get_database_info()
	print(f"\n✅ Successfully loaded {info['total_documents']} images into the RAG system")
	print(f"Corpus saved to: {tool.corpus_path}")

	return tool

if __name__ == "__main__":
	tool = load_multiple_mappings()

	# Test a simple search
	if tool and tool.get_database_info()['total_documents'] > 0:
		print("\n🔍 Testing search functionality...")
		results = tool.search("example query", k=3)
		print("Search test completed!")
