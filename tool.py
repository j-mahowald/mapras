import os
import json
import pickle
from smolagents import Tool
from typing import List, Union, Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import DataLoader
from colpali_engine.models import ColQwen2, ColQwen2Processor
from tqdm import tqdm

from pdf2image import convert_from_path


class VisualSearchTool(Tool):
	name = "visual_search"
	description = "Performs similarity search on visual documents and images. Returns the most similar documents to a query."
	output_type = "string"
	inputs = {
		"query": {
			"type": "string",
			"description": "The query to search for. This should describe what you're looking for.",
		},
		"k": {
			"type": "integer",
			"description": "The number of most similar documents to retrieve.",
			"default": 3,
			"nullable": True,
		}
	}

	def __init__(self, corpus_path: str = "./visual_corpus", embeddings_file: str = 'original_embeddings.pkl', *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_name = "vidore/colqwen2-v1.0"  # Use ColQwen2-v1.0 model
		self.corpus_path = corpus_path
		self.embeddings_file = os.path.join(corpus_path, embeddings_file)
		self.metadata_file = os.path.join(corpus_path, "metadata.json")
		self.is_initialized = False
		self.documents = []
		self.embeddings = []


	class Document:
		def __init__(self, image: Image.Image, title: str = None, source: str = None, image_path: str = None, original_url: str = None):
			self.image = image
			self.title = title or "Untitled"
			self.source = source or "Unknown"
			self.image_path = image_path  # Store file path for persistence
			self.original_url = original_url  # Store original URL

		def __repr__(self):
			return f"Document(title='{self.title}', source='{self.source}')"

		def to_dict(self):
			"""Convert document metadata to dictionary for JSON serialization."""
			return {
				'title': self.title,
				'source': self.source,
				'image_path': self.image_path,
				'original_url': self.original_url
			}

		@classmethod
		def from_dict(cls, data: dict):
			"""Create document from dictionary, loading image if path exists."""
			image = None
			if data.get('image_path') and os.path.exists(data['image_path']):
				try:
					image = Image.open(data['image_path'])
				except Exception as e:
					print(f"Warning: Could not load image from {data['image_path']}: {e}")
					image = None

			return cls(
				image=image,
				title=data['title'],
				source=data['source'],
				image_path=data.get('image_path'),
				original_url=data.get('original_url')
			)

	def _init_models(self, model_name: str) -> None:
		self.device = "mps" if torch.backends.mps.is_available() else "cpu"
		print('model name used: ', model_name)

		# For MPS, avoid device_map="auto" and use float16 instead of bfloat16
		if self.device == "mps":
			self.model = ColQwen2.from_pretrained(
				model_name,
				torch_dtype=torch.float16,
				device_map=None,
				low_cpu_mem_usage=True
			).eval().to(self.device)
		else:
			self.model = ColQwen2.from_pretrained(
				model_name,
				torch_dtype=torch.bfloat16,
				device_map="auto"
			).eval()

		self.processor = ColQwen2Processor.from_pretrained(model_name)

	def setup(self, load_existing: bool = True):
		"""Initialize the models and optionally load existing corpus."""
		self._init_models(self.model_name)

		if load_existing:
			self.load_corpus()
		else:
			self.documents = []
			self.embeddings = []

		self.is_initialized = True

	def save_corpus(self):
		"""Save embeddings and metadata to disk."""
		# Create directory if it doesn't exist
		os.makedirs(self.corpus_path, exist_ok=True)

		# Save embeddings
		if self.embeddings:
			with open(self.embeddings_file, 'wb') as f:
				pickle.dump(self.embeddings, f)
			print(f"Saved {len(self.embeddings)} embeddings to {self.embeddings_file}")

		# Save metadata
		metadata = [doc.to_dict() for doc in self.documents]
		with open(self.metadata_file, 'w') as f:
			json.dump(metadata, f, indent=2)
		print(f"Saved metadata for {len(self.documents)} documents to {self.metadata_file}")

	def load_corpus(self):
		"""Load existing embeddings and metadata from disk."""
		self.documents = []
		self.embeddings = []

		# Load embeddings
		if os.path.exists(self.embeddings_file):
			try:
				with open(self.embeddings_file, 'rb') as f:
					self.embeddings = pickle.load(f)
				print(f"Loaded {len(self.embeddings)} embeddings from {self.embeddings_file}")
			except Exception as e:
				print(f"Warning: Could not load embeddings: {e}")
				self.embeddings = []

		# Load metadata
		if os.path.exists(self.metadata_file):
			try:
				with open(self.metadata_file, 'r') as f:
					metadata = json.load(f)

				self.documents = [self.Document.from_dict(data) for data in metadata]
				print(f"Loaded metadata for {len(self.documents)} documents from {self.metadata_file}")

				# Verify embeddings and documents match
				if len(self.embeddings) != len(self.documents):
					print(f"Warning: Mismatch between embeddings ({len(self.embeddings)}) and documents ({len(self.documents)})")
					# Truncate to smaller size to maintain consistency
					min_size = min(len(self.embeddings), len(self.documents))
					self.embeddings = self.embeddings[:min_size]
					self.documents = self.documents[:min_size]
					print(f"Truncated to {min_size} items for consistency")

			except Exception as e:
				print(f"Warning: Could not load metadata: {e}")
				self.documents = []

		if self.documents:
			print(f"Successfully loaded corpus with {len(self.documents)} documents")

	def _compute_embeddings(self, images: List[Image.Image]) -> List:
		"""Compute embeddings for a list of images."""

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

	def add_image(self, image: Union[Image.Image, str], title: str = None, source: str = None, original_url: str = None, auto_save: bool = True) -> int:
		"""Add a single image to the database."""
		if not self.is_initialized:
			self.setup()

		image_path = None
		# Handle file path input
		if isinstance(image, str):
			image_path = image
			image = Image.open(image)
			if title is None:
				title = os.path.basename(image_path)
			if source is None:
				source = image_path

		# Create document
		doc = self.Document(image=image, title=title, source=source, image_path=image_path, original_url=original_url)

		# Compute embedding
		embedding = self._compute_embeddings([image])[0]

		# Add to database
		self.documents.append(doc)
		self.embeddings.append(embedding)

		print(f"Added image: {doc.title}")

		if auto_save:
			self.save_corpus()

		return len(self.documents)

	def add_images(self, images: List[Union[Image.Image, str]], titles: List[str] = None, sources: List[str] = None, original_urls: List[str] = None, auto_save: bool = True) -> int:
		"""Add multiple images to the database."""
		if not self.is_initialized:
			self.setup()

		processed_images = []
		documents = []

		for i, img in enumerate(images):
			image_path = None
			# Handle file path input
			if isinstance(img, str):
				image_path = img
				pil_img = Image.open(img)
				title = titles[i] if titles and i < len(titles) else os.path.basename(img)
				source = sources[i] if sources and i < len(sources) else img
			else:
				pil_img = img
				title = titles[i] if titles and i < len(titles) else f"Image_{len(self.documents) + i + 1}"
				source = sources[i] if sources and i < len(sources) else "Direct input"

			original_url = original_urls[i] if original_urls and i < len(original_urls) else None

			processed_images.append(pil_img)
			documents.append(self.Document(image=pil_img, title=title, source=source, image_path=image_path, original_url=original_url))

		# Compute embeddings for all images
		embeddings = self._compute_embeddings(processed_images)

		# Add to database
		self.documents.extend(documents)
		self.embeddings.extend(embeddings)

		print(f"Added {len(images)} images to database. Total: {len(self.documents)}")

		if auto_save:
			self.save_corpus()

		return len(self.documents)

	def add_pdf(self, pdf_path: str, title_prefix: str = None, auto_save: bool = True) -> int:
		"""Add all pages from a PDF as separate documents."""
		if not self.is_initialized:
			self.setup()

		filename = os.path.basename(pdf_path)
		title_prefix = title_prefix or filename.replace('.pdf', '')

		# Convert PDF to images
		images = convert_from_path(pdf_path, thread_count=4)

		# Create titles and sources for each page
		titles = [f"{title_prefix} - Page {i+1}" for i in range(len(images))]
		sources = [pdf_path] * len(images)

		return self.add_images(images, titles, sources, auto_save=auto_save)

	def search(self, query: str, k: int = 3) -> Tuple[List, str]:
		"""Search for the most similar documents to the query."""
		if not self.is_initialized:
			self.setup()

		if not self.documents:
			message = "No documents in database. Add some images first!"
			print(message)
			return [], message

		import torch

		k = min(k, len(self.embeddings))

		# Embed the query
		with torch.no_grad():
			batch_query = self.processor.process_queries([query]).to(self.device)
			query_embedding = self.model(**batch_query)
			query_embedding = list(torch.unbind(query_embedding.to("cpu")))[0]

		# Calculate similarities
		scores = self.processor.score([query_embedding], self.embeddings, device=self.device)[0]
		top_k_indices = scores.topk(k).indices.tolist()
		top_k_scores = scores.topk(k).values.tolist()

		# Get results
		results = []
		result_lines = [f"Top {k} matches for '{query}':"]

		for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
			doc = self.documents[idx]
			results.append({
				'document': doc,
				'score': float(score),
				'rank': i + 1
			})
			result_lines.append(f"{i+1}. {doc.title} (Score: {score:.3f}) - Source: {doc.source}")

		result_string = "\n".join(result_lines)
		print(result_string)

		return results, result_string

	def get_database_info(self) -> dict:
		"""Get information about the current database."""
		return {
			'total_documents': len(self.documents),
			'is_initialized': self.is_initialized,
			'corpus_path': self.corpus_path,
			'documents': [{'title': doc.title, 'source': doc.source} for doc in self.documents]
		}

	def clear_database(self, remove_files: bool = False):
		"""Clear all documents and embeddings."""
		self.documents = []
		self.embeddings = []
		print("In-memory database cleared.")

		if remove_files:
			if os.path.exists(self.embeddings_file):
				os.remove(self.embeddings_file)
			if os.path.exists(self.metadata_file):
				os.remove(self.metadata_file)
			print("Saved corpus files removed.")

	def get_search_results(self, query: str, k: int = 3) -> List:
		"""Get search results as a list for programmatic access."""
		results, _ = self.search(query, k)
		return results

	def forward(self, query: str, k: int = 3) -> str:
		"""Main interface for the tool."""
		assert isinstance(query, str), "Query must be a string"
		results, result_string = self.search(query, k)
		return result_string