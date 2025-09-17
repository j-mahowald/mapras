"""
Embedding Search Tool for ColPali visual document search.

This module provides functionality to search existing embeddings and add new images from URLs,
creating merged embedding files for comprehensive visual document search.
"""

import os
import json
import pickle
from io import BytesIO
from typing import List, Union, Tuple, Optional

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor

class EmbeddingSearchTool:
	"""A dedicated search tool for visual document embeddings with URL support."""

	def __init__(self, corpus_path: str = "./visual_corpus"):
		self.model_name = "vidore/colqwen2-v1.0"
		self.corpus_path = corpus_path

		# File paths
		self.original_embeddings_file = os.path.join(corpus_path, "embeddings.pkl")
		self.original_metadata_file = os.path.join(corpus_path, "metadata.json")
		self.merged_embeddings_file = os.path.join(corpus_path, "merged_embeddings.pkl")
		self.merged_metadata_file = os.path.join(corpus_path, "merged_metadata.json")

		# Model components
		self.is_initialized = False
		self.model = None
		self.processor = None
		self.device = None

		# Data storage
		self.original_embeddings = []
		self.original_documents = []
		self.merged_embeddings = []
		self.merged_documents = []

	def _init_models(self):
		"""Initialize the ColQwen2 model and processor."""
		if self.is_initialized:
			return

		self.device = "mps" if torch.backends.mps.is_available() else "cpu"
		print(f'Initializing model on device: {self.device}')

		if self.device == "mps":
			self.model = ColQwen2.from_pretrained(
				self.model_name,
				torch_dtype=torch.float16,
				device_map=None,
				low_cpu_mem_usage=True
			).eval().to(self.device)
		else:
			self.model = ColQwen2.from_pretrained(
				self.model_name,
				torch_dtype=torch.bfloat16,
				device_map="auto"
			).eval()

		self.processor = ColQwen2Processor.from_pretrained(self.model_name)
		self.is_initialized = True
		print("Model initialized successfully")

	def load_original_embeddings(self):
		"""Load the original embeddings and metadata."""
		print("Loading original embeddings...")

		# Load embeddings
		if os.path.exists(self.original_embeddings_file):
			with open(self.original_embeddings_file, 'rb') as f:
				raw_embeddings = pickle.load(f)
			print(f"Loaded {len(raw_embeddings)} original embeddings")
		else:
			print("No original embeddings file found")
			self.original_embeddings = []
			return

		# Load metadata
		if os.path.exists(self.original_metadata_file):
			with open(self.original_metadata_file, 'r') as f:
				raw_documents = json.load(f)
			print(f"Loaded {len(raw_documents)} original documents")
		else:
			print("No original metadata file found")
			self.original_embeddings = []
			return

		# Filter out embeddings with NaN values and ensure consistency
		print("Filtering out corrupted embeddings...")
		filtered_embeddings = []
		filtered_documents = []

		min_size = min(len(raw_embeddings), len(raw_documents))
		nan_count = 0

		for i in range(min_size):
			embedding = raw_embeddings[i]
			document = raw_documents[i]

			# Check for NaN values
			if torch.isnan(embedding).any():
				nan_count += 1
				continue

			filtered_embeddings.append(embedding)
			filtered_documents.append(document)

		self.original_embeddings = filtered_embeddings
		self.original_documents = filtered_documents

		print(f"Filtered out {nan_count} corrupted embeddings")
		print(f"Using {len(self.original_embeddings)} valid embeddings")

	def search_original_embeddings(self, query: str, k: int = 5) -> List[dict]:
		"""Search the original embeddings for the most similar documents."""
		if not self.is_initialized:
			self._init_models()

		if not self.original_embeddings:
			self.load_original_embeddings()

		if not self.original_embeddings:
			print("No original embeddings available for search")
			return []

		k = min(k, len(self.original_embeddings))

		# Embed the query
		with torch.no_grad():
			batch_query = self.processor.process_queries([query]).to(self.device)
			query_embedding = self.model(**batch_query)
			query_embedding = list(torch.unbind(query_embedding.to("cpu")))[0]

		# Calculate similarities - use CPU to avoid MPS issues with scoring
		try:
			# Always use CPU for scoring to avoid NaN issues with MPS
			cpu_embeddings = [emb.to("cpu") for emb in self.original_embeddings]
			scores = self.processor.score([query_embedding.to("cpu")], cpu_embeddings, device="cpu")[0]
		except Exception as e:
			print(f"Error with processor scoring: {e}")
			print("Falling back to manual cosine similarity calculation...")
			# Manual similarity calculation as fallback
			scores = torch.zeros(len(self.original_embeddings))
			query_flat = query_embedding.flatten()
			for i, doc_emb in enumerate(self.original_embeddings):
				doc_flat = doc_emb.flatten()
				if len(query_flat) == len(doc_flat):
					# Use cosine similarity if shapes match
					similarity = torch.cosine_similarity(query_flat.unsqueeze(0), doc_flat.unsqueeze(0))
					scores[i] = similarity.item()
				else:
					# If shapes don't match, skip this embedding
					scores[i] = 0.0
		top_k_indices = scores.topk(k).indices.tolist()
		top_k_scores = scores.topk(k).values.tolist()

		# Get results
		results = []
		print(f"\nTop {k} matches for '{query}' in original embeddings:")
		for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
			doc = self.original_documents[idx]
			result = {
				'rank': i + 1,
				'score': float(score),
				'title': doc['title'],
				'source': doc['source'],
				'image_path': doc.get('image_path', ''),
				'original_url': doc.get('original_url', ''),
				'type': 'original'
			}
			results.append(result)
			print(f"{i+1}. {doc['title']} (Score: {score:.3f}) - Source: {doc['source']}")

		return results

	def download_image_from_url(self, url: str, timeout: int = 30) -> Image.Image:
		"""Download an image from a URL."""
		try:
			headers = {
				'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
			}
			response = requests.get(url, timeout=timeout, headers=headers)
			response.raise_for_status()

			image = Image.open(BytesIO(response.content))
			if image.mode != 'RGB':
				image = image.convert('RGB')
			return image
		except (requests.RequestException, OSError) as e:
			raise ValueError(f"Failed to download image from {url}: {str(e)}") from e

	def _compute_embeddings(self, images: List[Image.Image]) -> List:
		"""Compute embeddings for a list of images."""
		if not self.is_initialized:
			self._init_models()

		# Create dataloader for batch processing
		dataloader = DataLoader(
			images,
			batch_size=4,
			shuffle=False,
			collate_fn=lambda x: self.processor.process_images(x).to(self.device),
		)

		embeddings = []
		for batch in tqdm(dataloader, desc="Computing embeddings"):
			with torch.no_grad():
				batch = {k: v.to(self.device) for k, v in batch.items()}
				batch_embeddings = self.model(**batch)
			embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))

		return embeddings

	def add_images_from_urls(self, urls: List[str], titles: List[str] = None) -> List[dict]:
		"""Add images from URLs and return their metadata."""
		if not urls:
			return []

		new_documents = []
		new_images = []

		print(f"Downloading and processing {len(urls)} images...")

		for i, url in enumerate(urls):
			try:
				# Download image
				image = self.download_image_from_url(url)
				title = titles[i] if titles and i < len(titles) else f"URL_Image_{i+1}"

				# Create document metadata
				doc = {
					'title': title,
					'source': url,
					'image_path': url,
					'type': 'url'
				}

				new_documents.append(doc)
				new_images.append(image)
				print(f"✓ Downloaded: {title}")

			except Exception as e:
				print(f"✗ Failed to download {url}: {str(e)}")
				continue

		if not new_images:
			print("No images were successfully downloaded")
			return []

		# Compute embeddings for new images
		print("Computing embeddings for new images...")
		new_embeddings = self._compute_embeddings(new_images)

		return list(zip(new_embeddings, new_documents))

	def create_merged_embeddings(self, new_url_data: List[Tuple] = None) -> bool:
		"""Create merged embeddings file with original + new embeddings."""
		# Load original embeddings if not already loaded
		if not self.original_embeddings:
			self.load_original_embeddings()

		# Start with original data
		self.merged_embeddings = self.original_embeddings.copy()
		self.merged_documents = self.original_documents.copy()

		# Add new data if provided
		if new_url_data:
			for embedding, document in new_url_data:
				self.merged_embeddings.append(embedding)
				self.merged_documents.append(document)

		print(f"Created merged dataset with {len(self.merged_embeddings)} embeddings")

		# Save merged embeddings and metadata
		os.makedirs(self.corpus_path, exist_ok=True)

		# Save embeddings
		with open(self.merged_embeddings_file, 'wb') as f:
			pickle.dump(self.merged_embeddings, f)
		print(f"Saved merged embeddings to {self.merged_embeddings_file}")

		# Save metadata
		with open(self.merged_metadata_file, 'w') as f:
			json.dump(self.merged_documents, f, indent=2)
		print(f"Saved merged metadata to {self.merged_metadata_file}")

		return True

	def search_merged_embeddings(self, query: str, k: int = 5) -> List[dict]:
		"""Search the merged embeddings for the most similar documents."""
		if not self.is_initialized:
			self._init_models()

		if not self.merged_embeddings:
			print("No merged embeddings available. Create merged embeddings first.")
			return []

		k = min(k, len(self.merged_embeddings))

		# Embed the query
		with torch.no_grad():
			batch_query = self.processor.process_queries([query]).to(self.device)
			query_embedding = self.model(**batch_query)
			query_embedding = list(torch.unbind(query_embedding.to("cpu")))[0]

		# Calculate similarities - force CPU for scoring if MPS causes nan
		try:
			scores = self.processor.score([query_embedding], self.merged_embeddings, device=self.device)[0]
			# Check if scores contain nan
			if torch.isnan(scores).any():
				print("Warning: NaN scores detected, retrying on CPU...")
				# Ensure embeddings are on CPU for CPU scoring
				cpu_embeddings = [emb.to("cpu") for emb in self.merged_embeddings]
				scores = self.processor.score([query_embedding.to("cpu")], cpu_embeddings, device="cpu")[0]
		except Exception as e:
			print(f"Error with {self.device} scoring, falling back to CPU: {e}")
			# Ensure embeddings are on CPU for CPU scoring
			cpu_embeddings = [emb.to("cpu") for emb in self.merged_embeddings]
			scores = self.processor.score([query_embedding.to("cpu")], cpu_embeddings, device="cpu")[0]
		top_k_indices = scores.topk(k).indices.tolist()
		top_k_scores = scores.topk(k).values.tolist()

		# Get results
		results = []
		print(f"\nTop {k} matches for '{query}' in merged embeddings:")
		for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
			doc = self.merged_documents[idx]
			result = {
				'rank': i + 1,
				'score': float(score),
				'title': doc['title'],
				'source': doc['source'],
				'image_path': doc.get('image_path', ''),
				'original_url': doc.get('original_url', ''),
				'type': doc.get('type', 'original')
			}
			results.append(result)
			source_type = "🌐" if doc.get('type') == 'url' else "📁"
			print(f"{i+1}. {source_type} {doc['title']} (Score: {score:.3f}) - Source: {doc['source']}")

		return results

	def get_stats(self) -> dict:
		"""Get statistics about the embeddings."""
		if not self.original_embeddings:
			self.load_original_embeddings()

			# Get corpus size from metadata if available
		corpus_size = len(self.original_documents) if self.original_documents else 0
		if corpus_size == 0 and os.path.exists(self.original_metadata_file):
			try:
				with open(self.original_metadata_file, 'r') as f:
					raw_documents = json.load(f)
					corpus_size = len(raw_documents)
			except:
				pass

		stats = {
			'original_embeddings': len(self.original_embeddings),
			'merged_embeddings': len(self.merged_embeddings),
			'corpus_size': corpus_size,
			'merged_file_exists': os.path.exists(self.merged_embeddings_file)
		}

		return stats

	def interactive_search(self):
		"""Interactive search interface."""
		print("\n=== Embedding Search Tool ===")
		print("Commands:")
		print("  search <query>          - Search original embeddings")
		print("  add <url1> [url2] ...   - Add images from URLs")
		print("  merge                   - Create merged embeddings")
		print("  search_merged <query>   - Search merged embeddings")
		print("  stats                   - Show statistics")
		print("  quit                    - Exit")

		while True:
			try:
				cmd = input("\n> ").strip()
				if not cmd:
					continue

				if cmd == "quit":
					break

				self._handle_command(cmd)

			except KeyboardInterrupt:
				print("\nExiting...")
				break
			except Exception as e:
				print(f"Error: {e}")

	def _handle_command(self, cmd: str):
		"""Handle individual commands."""
		if cmd == "stats":
			stats = self.get_stats()
			print(f"Original embeddings: {stats['original_embeddings']}")
			print(f"Merged embeddings: {stats['merged_embeddings']}")
			print(f"Merged file exists: {stats['merged_file_exists']}")
		elif cmd.startswith("search "):
			query = cmd[7:]
			self.search_original_embeddings(query)
		elif cmd.startswith("search_merged "):
			query = cmd[14:]
			self.search_merged_embeddings(query)
		elif cmd.startswith("add "):
			urls = cmd[4:].split()
			if urls:
				new_data = self.add_images_from_urls(urls)
				if new_data:
					print(f"Added {len(new_data)} new images. Use 'merge' to create combined embeddings.")
				else:
					print("No images were added.")
		elif cmd == "merge":
			self.create_merged_embeddings()
		else:
			print("Unknown command. Type 'quit' to exit.")


def main():
	"""Main function to run the embedding search tool."""
	tool = EmbeddingSearchTool()
	tool.interactive_search()


if __name__ == "__main__":
	main()