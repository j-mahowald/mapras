#!/usr/bin/env python3
"""
Test script for the EmbeddingSearchTool to demonstrate its functionality.
"""

from embedding_search_tool import EmbeddingSearchTool

def test_original_search():
	"""Test searching original embeddings."""
	print("=== Testing Original Embeddings Search ===")
	tool = EmbeddingSearchTool()

	# Test different queries
	queries = ["city skyline", "mountain landscape", "ocean view", "forest trees"]

	for query in queries:
		print(f"\nSearching for: '{query}'")
		results = tool.search_original_embeddings(query, k=2)
		if results:
			print(f"Found {len(results)} results:")
			for r in results:
				print(f"  - {r['title']} (rank: {r['rank']})")
		else:
			print("No results found")

def test_url_addition():
	"""Test adding images from URLs."""
	print("\n=== Testing URL Image Addition ===")
	tool = EmbeddingSearchTool()

	# Example URLs (replace with actual image URLs)
	test_urls = [
		"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
		"https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/120px-React-icon.svg.png"
	]

	print(f"Attempting to add {len(test_urls)} images from URLs...")
	try:
		new_data = tool.add_images_from_urls(test_urls, titles=["Test PNG", "React Logo"])
		print(f"Successfully processed {len(new_data)} images")

		# Create merged embeddings
		print("Creating merged embeddings...")
		tool.create_merged_embeddings(new_data)

		# Test searching merged embeddings
		print("Testing search on merged embeddings...")
		results = tool.search_merged_embeddings("logo", k=3)
		if results:
			for r in results:
				source_type = "🌐" if r['type'] == 'url' else "📁"
				print(f"  {source_type} {r['title']} (rank: {r['rank']})")

	except Exception as e:
		print(f"URL test failed: {e}")

def test_stats():
	"""Test getting statistics."""
	print("\n=== Testing Statistics ===")
	tool = EmbeddingSearchTool()
	stats = tool.get_stats()

	print(f"Original embeddings: {stats['original_embeddings']}")
	print(f"Merged embeddings: {stats['merged_embeddings']}")
	print(f"Merged file exists: {stats['merged_file_exists']}")

def main():
	"""Run all tests."""
	print("ColPali Embedding Search Tool - Test Suite")
	print("=" * 50)

	try:
		test_original_search()
		test_stats()

		# Note: URL test requires internet connection and may fail
		# Uncomment to test URL functionality:
		# test_url_addition()

	except Exception as e:
		print(f"Test failed with error: {e}")

	print("\n" + "=" * 50)
	print("Test completed!")

if __name__ == "__main__":
	main()