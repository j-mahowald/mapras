import os
import time
import random
import requests
import json
import pickle
from typing import Dict, List, Optional, Tuple

# Set tokenizers parallelism before importing transformers/tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import torch
from tool import VisualSearchTool
from PIL import Image


class ImageDownloadEmbeddingPipeline:
	"""
	Pipeline class to handle image downloading, embedding, and indexing.

	Manages the complete workflow of:
	1. Loading and filtering CSV data
	2. Downloading images from IIIF URLs with rate limiting
	3. Computing embeddings in batches
	4. Saving metadata and index mappings
	"""

	def __init__(
		self,
		csv_path: str = "merged_files.csv",
		subset_size: int = 1000,
		batch_size: int = 50,
		rate_limit: int = 150,
		max_retries: int = 5,
		num_processes: int = 5,
		output_dir: str = "images/train/",
		corpus_path: str = "./visual_corpus",
		random_seed: int = 46
	):
		"""
		Initialize the pipeline with configuration parameters.

		Args:
			csv_path: Path to the CSV file containing image URLs
			subset_size: Number of images to process
			batch_size: Number of images to download per batch
			rate_limit: Maximum requests per minute
			max_retries: Maximum retry attempts for failed downloads
			num_processes: Number of parallel download processes
			output_dir: Directory to save downloaded images
			corpus_path: Path for visual corpus (used in embedding)
			random_seed: Random seed for reproducibility
		"""
		self.csv_path = csv_path
		self.subset_size = subset_size
		self.batch_size = batch_size
		self.rate_limit = rate_limit
		self.max_retries = max_retries
		self.num_processes = num_processes
		self.output_dir = output_dir
		self.corpus_path = corpus_path
		self.random_seed = random_seed

		# Calculated parameters
		self.interval = 60 / rate_limit  # seconds between requests

		# State variables
		self.subset = None
		self.all_metadata = []
		self.index_mapping = {}
		self.embedding_tool = None  # Will be initialized once

		# Set random seed
		random.seed(self.random_seed)

		# Ensure output directory exists
		os.makedirs(self.output_dir, exist_ok=True)

	def load_and_filter_data(self, file_count_threshold: int = 100) -> pd.DataFrame:
		"""
		Load CSV data and filter based on file count.

		Args:
			file_count_threshold: Maximum file count to include in subset

		Returns:
			Filtered DataFrame subset
		"""
		print(f"📂 Loading data from {self.csv_path}...")
		merged_files = pd.read_csv(self.csv_path)

		# Filter by file count
		merged_files['file_count'] = merged_files['file_count'].fillna(0)
		self.subset = merged_files[merged_files['file_count'] < file_count_threshold]

		print(f"✅ Loaded {len(self.subset)} rows after filtering (file_count < {file_count_threshold})")
		return self.subset

	def get_image_data(self, row_index: int) -> Dict[str, str]:
		"""
		Extract image URL and metadata from CSV row.

		Args:
			row_index: Index of the row in the subset DataFrame

		Returns:
			Dictionary containing iiif_url and resource_url
		"""
		row_data = self.subset.iloc[row_index]
		iiif_url = row_data['file_url'].replace("pct:100", "750,")
		resource_url = row_data['resource_url']

		return {
			'iiif_url': iiif_url,
			'resource_url': resource_url
		}

	def download_image(self, row_index: int) -> Optional[Dict[str, str]]:
		"""
		Download a single image from IIIF URL with retry logic.
		Args: row_index: Index of the row to download

		Returns: Dictionary with filepath, original_url, and resource_id, or None if failed
		"""
		retry_count = 0

		image_data = self.get_image_data(row_index)
		iiif_url = image_data['iiif_url']
		resource_url = image_data['resource_url']

		while retry_count < self.max_retries:
			try:
				response = requests.get(iiif_url, timeout=5)
				if response.status_code == 200:
					save_path = iiif_url.split("/")[-5].replace(":", "_")
					image_path = os.path.join(self.output_dir, f"{save_path}.jpg")

					with open(image_path, "wb") as file:
						file.write(response.content)

					return {
						'filepath': image_path,
						'original_url': iiif_url,
						'resource_id': resource_url
					}

				print(f"Retrying row {row_index} due to status code: {response.status_code}")

			except Exception as e:
				print(f"Retrying row {row_index} due to error: {e}")

			retry_count += 1
			print(f"Retry: {retry_count}")
			time.sleep(2 ** retry_count)

		return None

	def rate_limited_download(self, row_index: int) -> Optional[Dict[str, str]]:
		"""Download image with rate limiting.
		Args: row_index: Index of the row to download
		Returns: Download result or None"""
		result = self.download_image(row_index)
		time.sleep(self.interval)
		return result

	def _initialize_embedding_tool(self):
		"""Initialize the embedding tool once for reuse across batches."""
		if self.embedding_tool is None:
			print("🔧 Initializing embedding model (one-time setup)...")
			self.embedding_tool = VisualSearchTool(corpus_path=self.corpus_path)
			self.embedding_tool.setup(load_existing=False)
			print(f"✅ Model initialized on device: {self.embedding_tool.device}")
			print("✅ Model ready for embedding")

	def _compute_embeddings_fast(self, images: List[Image.Image], batch_size: int = 4) -> List:
		"""
		Compute embeddings with configurable batch size for better performance.

		Args:
			images: List of PIL images
			batch_size: Number of images to process per forward pass (default 8)

		Returns:
			List of embeddings
		"""
		from torch.utils.data import DataLoader

		# Create dataloader with batch size
		dataloader = DataLoader(
			images,
			batch_size=batch_size,
			shuffle=False,
			collate_fn=lambda x: self.embedding_tool.processor.process_images(x).to(self.embedding_tool.device),
		)

		embeddings = []
		for batch in tqdm(dataloader, desc="Computing embeddings"):
			with torch.no_grad():
				batch = {k: v.to(self.embedding_tool.device) for k, v in batch.items()}
				batch_embeddings = self.embedding_tool.model(**batch)

				# Immediately move to CPU and unbind to free GPU memory
				batch_embeddings_cpu = batch_embeddings.to("cpu")
				embeddings.extend(list(torch.unbind(batch_embeddings_cpu)))

				# Clear GPU cache
				del batch_embeddings, batch_embeddings_cpu
				if self.embedding_tool.device == "mps":
					torch.mps.empty_cache()
				elif torch.cuda.is_available():
					torch.cuda.empty_cache()

		return embeddings

	def embed_batch(self, image_paths: List[str], batch_num: int, embedding_batch_size: int = 2) -> Optional[List]:
		"""
		Compute embeddings for a batch of images and save to file.
		Args:
			image_paths: List of image file paths
			batch_num: Batch number for naming the output file
			embedding_batch_size: Number of images to process per GPU forward pass
		Returns:
			List of embeddings or None if failed
		"""
		print(f"\n🔄 Embedding batch {batch_num}...")

		# Ensure embedding tool is initialized (only happens once)
		self._initialize_embedding_tool()

		# load images
		images = []
		valid_paths = []
		for path in image_paths:
			try:
				img = Image.open(path)
				images.append(img)
				valid_paths.append(path)
			except Exception as e:
				print(f"Warning: Could not load image {path}: {e}")

		if not images:
			print(f"⚠️  No valid images to embed in batch {batch_num}")
			return None

		# Compute embeddings with batch size
		try:
			embeddings = self._compute_embeddings_fast(images, batch_size=embedding_batch_size)

			# Check for NaN values in each embedding (variable-length tensors)
			nan_count = 0
			for i, emb in enumerate(embeddings):
				if torch.isnan(emb).any():
					nan_count += 1
					# Replace NaN embedding with zeros of same shape
					embeddings[i] = torch.zeros_like(emb)

			if nan_count > 0:
				print(f"⚠️  Warning: Found {nan_count} embeddings with NaN values, replaced with zeros")

			# Save embeddings to batch-specific file immediately
			embeddings_file = f"embeddings_{batch_num}.pkl"
			with open(embeddings_file, 'wb') as f:
				pickle.dump(embeddings, f)

			print(f"✅ Saved {len(embeddings)} embeddings to {embeddings_file}")

			# Clear embeddings from memory after saving
			del embeddings, images
			if self.embedding_tool.device == "mps":
				torch.mps.empty_cache()
			elif torch.cuda.is_available():
				torch.cuda.empty_cache()

			return True  # Return success flag instead of embeddings

		except Exception as e:
			print(f"❌ Error embedding batch {batch_num}: {e}")
			import traceback
			traceback.print_exc()
			return None

	def process_batch(
		self,
		chunk_indices: List[int],
		batch_num: int,
		use_multiprocessing: bool = True
	) -> Tuple[List[Dict], Optional[List]]:
		"""
		Download and embed a batch of images.

		Args:
			chunk_indices: List of row indices to process
			batch_num: Batch number for tracking
			use_multiprocessing: Whether to use parallel downloads

		Returns:
			Tuple of (metadata_list, embeddings)
		"""
		# Download images
		if use_multiprocessing:
			with mp.Pool(processes=self.num_processes) as pool:
				try:
					results = pool.map(self.rate_limited_download, chunk_indices)
				except Exception as e:
					print(f"Error encountered: {e}")
					return [], None
		else:
			results = [self.rate_limited_download(idx) for idx in chunk_indices]

		# Filter out failed downloads
		results = [result for result in results if result is not None]

		if not results:
			print(f"⚠️  No images downloaded in batch {batch_num}")
			return [], None

		# Update index mapping and metadata
		for result in results:
			self.index_mapping[len(self.all_metadata)] = result['resource_id']
			self.all_metadata.append(result)

		# Save metadata incrementally
		with open("metadata_all.json", 'a') as f:
			for item in results:
				f.write(json.dumps(item) + "\n")

		print(f"Batch {batch_num}: {len(results)} files downloaded")

		return results, None  # Return None for embeddings, will be done separately

	def run(self, use_multiprocessing: bool = True, skip_embedding: bool = False) -> Dict:
		"""
		Execute the complete pipeline: load data, download images, and create embeddings.

		Args:
			use_multiprocessing: Whether to use parallel downloads
			skip_embedding: If True, only download images without embedding

		Returns:
			Dictionary with pipeline statistics
		"""
		start_time = time.time()

		# Load and filter data
		self.load_and_filter_data()

		# Select random subset
		rand_indices = random.sample(
			range(len(self.subset)),
			min(self.subset_size, len(self.subset))
		)

		total_batches = (len(rand_indices) + self.batch_size - 1) // self.batch_size

		print(f"\n🚀 Starting pipeline for {len(rand_indices)} images in {total_batches} batches")
		print(f"⚙️  Batch size: {self.batch_size}, Rate limit: {self.rate_limit} req/min\n")

		# Track batches for embedding
		batch_info = []

		# Process in batches (download only)
		for i in tqdm(range(0, len(rand_indices), self.batch_size), desc="Downloading batches"):
			chunk_indices = rand_indices[i:i + self.batch_size]
			batch_num = i // self.batch_size + 1

			results, _ = self.process_batch(chunk_indices, batch_num, use_multiprocessing)

			if results:
				batch_info.append({
					'batch_num': batch_num,
					'image_paths': [r['filepath'] for r in results]
				})

		# Now embed all batches (after multiprocessing is done)
		if not skip_embedding and batch_info:
			print("\n" + "="*60)
			print("📊 Download complete. Starting embedding phase...")
			print("="*60 + "\n")

			for batch in tqdm(batch_info, desc="Embedding batches"):
				self.embed_batch(batch['image_paths'], batch['batch_num'])

		# Save index mapping
		with open('index_mapping.json', 'w') as f:
			json.dump(self.index_mapping, f, indent=2)

		elapsed_time = time.time() - start_time

		# Print summary
		print(f"\n{'='*60}")
		print(f"✅ Pipeline completed!")
		print(f"{'='*60}")
		print(f"Total time: {elapsed_time:.2f} seconds")
		print(f"Images processed: {len(self.all_metadata)}")
		print(f"Index mapping entries: {len(self.index_mapping)}")
		print(f"Metadata saved to: metadata_all.json")
		print(f"Index mapping saved to: index_mapping.json")
		print(f"{'='*60}\n")

		return {
			'elapsed_time': elapsed_time,
			'images_processed': len(self.all_metadata),
			'index_entries': len(self.index_mapping),
			'batches_completed': total_batches
		}


def main():
	"""Example usage of the ImageDownloadEmbeddingPipeline."""
	pipeline = ImageDownloadEmbeddingPipeline(
		csv_path="merged_files.csv",
		subset_size=10000,
		batch_size=50,
		rate_limit=150,
		num_processes=5
	)

	stats = pipeline.run(use_multiprocessing=True)
	return stats


if __name__ == '__main__':
	main()
