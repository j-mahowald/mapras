"""unified visual search tool for colpali embeddings."""

import os
import pickle
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import glob

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader

from colpali_engine.models import ColQwen2, ColQwen2Processor

try:
	from pdf2image import convert_from_path
	PDF_SUPPORT = True
except ImportError:
	PDF_SUPPORT = False


class VisualSearchTool:
	"""
	tool for visual document search with ColPali.
	Supports:
	- loading distributed embeddings (multiple .pkl files)
	- adding images from URLs, local files, or PDFs
	- semantic search across all documents
	- single-file storage format: List[{'embedding': Tensor, 'iiif_url': str, 'title': str}]
	"""

	def __init__(self, corpus_path: str = "./visual_corpus", corpus_file: str = "corpus.pkl"):
		self.model_name = "vidore/colqwen2-v1.0"
		self.corpus_path = corpus_path
		self.corpus_file = os.path.join(corpus_path, corpus_file)

		self.is_initialized = False
		self.model = None
		self.processor = None
		self.device = None

		self.embeddings = []
		self.documents = []  # List of dicts: {'title': str, 'source': str, 'type': str}

	def _init_models(self):
		"""initialize colqwen2 model and processor"""
		if self.is_initialized:
			return

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		print(f"Initializing model on {self.device}...")

		torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
		self.model = ColQwen2.from_pretrained(
			self.model_name,
			torch_dtype=torch_dtype
		).eval().to(self.device)

		self.processor = ColQwen2Processor.from_pretrained(self.model_name)
		self.is_initialized = True
		print("Model initialized successfully")

	def load_corpus(self):
		"""load corpus from single pkl file (usually use the batch loader)"""
		if not os.path.exists(self.corpus_file):
			print(f"No corpus found at {self.corpus_file}")
			return

		print(f"Loading corpus from {self.corpus_file}...")
		try:
			with open(self.corpus_file, 'rb') as f:
				data = pickle.load(f)

			self.embeddings = []
			self.documents = []

			for item in data:
				if torch.isnan(item['embedding']).any():
					continue

				self.embeddings.append(item['embedding'])
				self.documents.append({
					'title': item.get('title', 'Untitled'),
					'source': item.get('iiif_url', item.get('source', 'Unknown')),
					'type': item.get('type', 'original')
				})

			print(f"Loaded {len(self.embeddings)} embeddings")

		except Exception as e:
			print(f"Error loading corpus: {e}")

	def load_distributed_corpus(self):
		"""load embeddings from distributed pkl files"""
		print("Loading distributed embeddings")

		all_embeddings = []
		all_documents = []

		for emb_file in glob.glob(os.path.join(self.corpus_path, "batch_*.pkl")):
		# for emb_file in [os.path.join(self.corpus_path, f"batch_{i}.pkl") for i in range(50)]:

			if not os.path.exists(emb_file):
				print('Path for distributed embeddings does not exist:', emb_file)
				continue

			try:
				with open(emb_file, 'rb') as f:
					data_list = pickle.load(f)

				for item in data_list:
					if isinstance(item, dict):
						embedding = item['embedding']
						iiif_url = item.get('iiif_url', item.get('source', 'Unknown'))
						title = item.get('title', f'Document_{len(all_embeddings)}')
					else:
						embedding = item
						iiif_url = f'Document_{len(all_embeddings)}'
						title = iiif_url

					if torch.isnan(embedding).any():
						continue

					all_embeddings.append(embedding)
					all_documents.append({
						'title': title,
						'source': iiif_url,
						'type': 'original'
					})

				print(f"Loaded {emb_file}: {len(data_list)} items")

			except Exception as e:
				print(f"Error loading {emb_file}: {e}")

		self.embeddings = all_embeddings
		self.documents = all_documents
		print(f"Total: {len(self.embeddings)} embeddings loaded")

	def save_corpus(self):
		"""save corpus to single pkl file"""
		os.makedirs(self.corpus_path, exist_ok=True)

		combined_data = [
			{
				'embedding': emb,
				'iiif_url': doc['source'],
				'title': doc['title'],
				'type': doc.get('type', 'original')
			}
			for emb, doc in zip(self.embeddings, self.documents)
		]

		with open(self.corpus_file, 'wb') as f:
			pickle.dump(combined_data, f)
		print(f"Saved {len(combined_data)} items to {self.corpus_file}")

	def _compute_embeddings(self, images: List[Image.Image], batch_size: int = 4) -> List[torch.Tensor]:
		"""compute embeddings for given images"""
		if not self.is_initialized:
			self._init_models()

		dataloader = DataLoader(
			images,
			batch_size=batch_size,
			shuffle=False,
			collate_fn=lambda x: self.processor.process_images(x)
		)

		embeddings = []
		for batch in dataloader:
			with torch.no_grad():
				batch = {k: v.to(self.device) for k, v in batch.items()}
				batch_embeddings = self.model(**batch)
				embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))

		return embeddings

	def download_image(self, url: str, timeout: int = 30) -> Optional[Image.Image]:
		"""download image from URL (most common)"""
		try:
			headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
			response = requests.get(url, timeout=timeout, headers=headers)
			response.raise_for_status()

			image = Image.open(BytesIO(response.content))
			if image.mode != 'RGB':
				image = image.convert('RGB')
			return image

		except Exception as e:
			print(f"Failed to download {url}: {e}")
			return None

	def add_images_from_urls(self, urls: List[str], titles: List[str] = None):
		"""add images from URLs to corpus"""
		images = []
		valid_urls = []
		valid_titles = []

		print(f"Downloading {len(urls)} images...")
		for i, url in enumerate(urls):
			image = self.download_image(url)
			if image:
				images.append(image)
				valid_urls.append(url)
				valid_titles.append(titles[i] if titles and i < len(titles) else f"URL_Image_{i+1}")

		if not images:
			print("No images downloaded")
			return

		print(f"Computing embeddings for {len(images)} images...")
		embeddings = self._compute_embeddings(images)

		for emb, url, title in zip(embeddings, valid_urls, valid_titles):
			self.embeddings.append(emb)
			self.documents.append({
				'title': title,
				'source': url,
				'type': 'url'
			})

		print(f"Added {len(images)} images. Total: {len(self.embeddings)}")

	def add_images_from_files(self, filepaths: List[str], titles: List[str] = None):
		"""add images from local files (use when adding URLs gives OOM)"""
		images = []
		valid_paths = []
		valid_titles = []

		for i, path in enumerate(filepaths):
			try:
				img = Image.open(path)
				if img.mode != 'RGB':
					img = img.convert('RGB')
				images.append(img)
				valid_paths.append(path)
				valid_titles.append(titles[i] if titles and i < len(titles) else os.path.basename(path))
			except Exception as e:
				print(f"Failed to load {path}: {e}")

		if not images:
			print("No images loaded")
			return

		embeddings = self._compute_embeddings(images)

		for emb, path, title in zip(embeddings, valid_paths, valid_titles):
			self.embeddings.append(emb)
			self.documents.append({
				'title': title,
				'source': path,
				'type': 'file'
			})

		print(f"Added {len(images)} images. Total: {len(self.embeddings)}")

	def add_pdf(self, pdf_path: str, title_prefix: str = None):
		"""add all pages from PDF as separate documents"""
		if not PDF_SUPPORT:
			print("PDF support not available. Install pdf2image.")
			return

		print(f"Converting PDF: {pdf_path}")
		images = convert_from_path(pdf_path, thread_count=4)

		prefix = title_prefix or os.path.basename(pdf_path).replace('.pdf', '')
		embeddings = self._compute_embeddings(images)

		for i, emb in enumerate(embeddings):
			self.embeddings.append(emb)
			self.documents.append({
				'title': f"{prefix} - Page {i+1}",
				'source': pdf_path,
				'type': 'pdf'
			})

		print(f"Added {len(images)} pages. Total: {len(self.embeddings)}")

	def search(self, query: str, k: int = 5) -> List[Dict]:
		"""search for most similar documents (returns a list)"""
		if not self.is_initialized:
			self._init_models()

		if not self.embeddings:
			print("No embeddings in corpus")
			return []

		k = min(k, len(self.embeddings))

		with torch.no_grad(): # embed query
			batch_query = self.processor.process_queries([query]).to(self.device)
			query_embedding = self.model(**batch_query)
			query_embedding = list(torch.unbind(query_embedding.to("cpu")))[0]

		try: 
			scores = self.processor.score(
				qs=[query_embedding],
				ps=self.embeddings,
				device=self.device
			)[0]
		except Exception as e:
			print(f"Scoring error: {e}")
			return []

		top_k_indices = scores.topk(k).indices.tolist()
		top_k_scores = scores.topk(k).values.tolist()

		results = [] # format results
		print(f"\nTop {k} matches for '{query}':")
		for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
			doc = self.documents[idx]

			type_emoji = {
				'url': '🌐',
				'file': '📁',
				'pdf': '📄',
				'original': '📚'
			}.get(doc['type'], '📄')

			result = {
				'rank': i + 1,
				'score': float(score),
				'title': doc['title'],
				'source': doc['source'],
				'type': doc['type'],
				'doc_index': idx  # Add index for image serving
			}
			results.append(result)
			print(f"{i+1}. {type_emoji} {doc['title']} (Score: {score:.3f})")
			print(f"   Source: {doc['source']}")

		return results

	def search_by_image(self, image: Image.Image, k: int = 5) -> List[Dict]:
		"""search for most similar documents (returns a list)"""
		if not self.is_initialized:
			self._init_models()
		
		# Get query image embedding (reuse existing method)
		query_embedding = self._compute_embeddings([image])[0]
		
		# Score against corpus (same as text search)
		scores = self.processor.score(
			qs=[query_embedding],
			ps=self.embeddings,
			device=self.device
		)[0]

		top_k_indices = scores.topk(k).indices.tolist()
		top_k_scores = scores.topk(k).values.tolist()

		results = [] # format results
		print(f"\nTop {k} matches for '{image}':")
		for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
			doc = self.documents[idx]

			type_emoji = {
				'url': '🌐',
				'file': '📁',
				'pdf': '📄',
				'original': '📚'
			}.get(doc['type'], '📄')

			result = {
				'rank': i + 1,
				'score': float(score),
				'title': doc['title'],
				'source': doc['source'],
				'type': doc['type'],
				'doc_index': idx  # Add index for image serving
			}
			results.append(result)
			print(f"{i+1}. {type_emoji} {doc['title']} (Score: {score:.3f})")
			print(f"   Source: {doc['source']}")

		return results

	def get_stats(self) -> Dict:
		"""corpus statistics"""
		type_counts = {}
		for doc in self.documents:
			doc_type = doc.get('type', 'unknown')
			type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

		return {
			'total_embeddings': len(self.embeddings),
			'corpus_file': self.corpus_file,
			'corpus_exists': os.path.exists(self.corpus_file),
			'type_counts': type_counts
		}

	def interactive_mode(self):
		"""cli interface (DEPRECATED) (just use the website)"""
		print("\n=== Unified Visual Search Tool ===")
		print("Commands:")
		print("  search <query>              - Search corpus")
		print("  add_urls <url1> [url2] ...  - Add images from URLs")
		print("  add_files <path1> [path2] ...- Add local images")
		print("  add_pdf <path>              - Add PDF")
		print("  load                        - Load corpus from file")
		print("  load_distributed <ranges>   - Load distributed files (e.g., 1-200,201-400)")
		print("  save                        - Save corpus to file")
		print("  stats                       - Show statistics")
		print("  quit                        - Exit\n")

		while True:
			try:
				cmd = input("> ").strip()
				if not cmd:
					continue

				if cmd == "quit":
					break
				elif cmd == "load":
					self.load_corpus()
				elif cmd.startswith("load_distributed "):
					ranges_str = cmd.split(maxsplit=1)[1]
					ranges = []
					for r in ranges_str.split(','):
						start, end = map(int, r.split('-'))
						ranges.append((start, end))
					self.load_distributed_corpus(ranges)
				elif cmd == "save":
					self.save_corpus()
				elif cmd == "stats":
					stats = self.get_stats()
					print(f"Total embeddings: {stats['total_embeddings']}")
					print(f"Corpus file: {stats['corpus_file']}")
					print(f"Exists: {stats['corpus_exists']}")
					print(f"Type counts: {stats['type_counts']}")
				elif cmd.startswith("search "):
					query = cmd[7:]
					self.search(query)
				elif cmd.startswith("add_urls "):
					urls = cmd[9:].split()
					self.add_images_from_urls(urls)
				elif cmd.startswith("add_files "):
					files = cmd[10:].split()
					self.add_images_from_files(files)
				elif cmd.startswith("add_pdf "):
					pdf_path = cmd[8:].strip()
					self.add_pdf(pdf_path)
				else:
					print("Unknown command")

			except KeyboardInterrupt:
				print("\nExiting...")
				break
			except Exception as e:
				print(f"Error: {e}")


def main():
	tool = VisualSearchTool()
	tool.interactive_mode()

if __name__ == "__main__":
	main()