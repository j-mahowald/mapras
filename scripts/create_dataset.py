import os
import pickle
from typing import List
import requests
from io import BytesIO
import time

import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

from src.tool import VisualSearchTool
import glob

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class EmbeddingPipeline:
	def __init__(self, csv_path: str, output_dir: str = "./embeddings"):
		self.df = pd.read_csv(csv_path)
		self.df = self.df[~self.df['p1_item_id'].str.contains('sanborn', case=False, na=False)]
		self.df = self.df[self.df['file_count'] >= 50]

		# remove rows whose file_url already appears in any embeddings/batch_*.pkl
		embedded_urls = set()
		out_dir = output_dir
		if os.path.isdir(out_dir):
			pattern = os.path.join(out_dir, "batch_*.pkl")
			for p in glob.glob(pattern):
				try:
					with open(p, "rb") as f:
						items = pickle.load(f)
					if not isinstance(items, list):
						continue
					for it in items:
						if not isinstance(it, dict):
							continue
						url = it.get("iiif_url") or it.get("file_url") or it.get("url")
						if url:
							embedded_urls.add(url)
				except Exception:
					# skip unreadable/invalid files
					continue

		if embedded_urls:
			# normalize file_url the same way embeddings were created (replace pct:100 -> 1500,)
			file_urls = self.df['file_url'].fillna('').astype(str).str.replace('pct:100', '1500,', regex=False)
			self.df = self.df[~file_urls.isin(embedded_urls)].reset_index(drop=True)
		print(f"Loaded {len(self.df)} rows after filtering")
		self.output_dir = output_dir
		self.tool = None
		self.temp_dir = "./temp_images"
		os.makedirs(self.temp_dir, exist_ok=True)
		os.makedirs(output_dir, exist_ok=True)

	def download_and_save(self, url: str, save_path: str):
		"""download image from IIIF URL with rate limiting."""
		min_interval = 60.0 / 150.0  # seconds per request (0.4s)
		now = time.time()
		last = getattr(self, "_last_download_time", None)
		if last is not None:
			wait = min_interval - (now - last)
			if wait > 0:
				time.sleep(wait)

		response = requests.get(url, timeout=30)
		response.raise_for_status()
		img = Image.open(BytesIO(response.content))
		if img.mode != 'RGB':
			img = img.convert('RGB')
		img.save(save_path)
		# record time of this download
		self._last_download_time = time.time()

	def process_indices(self, indices: List[int], output_file: str = "embeddings.pkl", batch_size: int = 1):
		"""Download, embed, and save images at given indices."""
		if self.tool is None:
			self.tool = VisualSearchTool()
			self.tool._init_models()
		
		data = []
		batch_images = []
		batch_urls = []
		batch_temp_paths = []
		
		print(f"Processing {len(indices)} images in batches of {batch_size}...")
		
		for idx in tqdm(indices):
			try:
				# download
				row = self.df.iloc[idx]
				url = row['file_url'].replace('pct:100', '1500,')
				temp_path = os.path.join(self.temp_dir, f"temp_{idx}.jpg")
				self.download_and_save(url, temp_path)
				
				# accumulate
				image = Image.open(temp_path)
				batch_images.append(image)
				batch_urls.append(url)
				batch_temp_paths.append(temp_path)
				
				# process when batch is full
				if len(batch_images) == batch_size:
					embeddings = self.tool._compute_embeddings(batch_images, batch_size=batch_size)
					for emb, url in zip(embeddings, batch_urls):
						data.append({'embedding': emb.cpu(), 'iiif_url': url})
					
					# cleanup
					for path in batch_temp_paths:
						os.remove(path)
					del batch_images, embeddings
					torch.cuda.empty_cache()
					
					batch_images = []
					batch_urls = []
					batch_temp_paths = []
					
			except Exception as e:
				print(f"✗ Failed index {idx}: {e}")
		
		# process remaining images
		if batch_images:
			try:
				embeddings = self.tool._compute_embeddings(batch_images, batch_size=len(batch_images))
				for emb, url in zip(embeddings, batch_urls):
					data.append({'embedding': emb.cpu(), 'iiif_url': url})
				for path in batch_temp_paths:
					os.remove(path)
				torch.cuda.empty_cache()
			except Exception as e:
				print(f"✗ Failed final batch: {e}")
		
		# save all
		output_path = os.path.join(self.output_dir, output_file)
		with open(output_path, 'wb') as f:
			pickle.dump(data, f)
		print(f"Saved {len(data)} embeddings to {output_path}")

pipeline = EmbeddingPipeline("merged_files.csv")

for i in range(50):
	start_idx = i * 500
	end_idx = start_idx + 500
	indices = list(range(start_idx, end_idx))
	pipeline.process_indices(indices, output_file=f"batch_{191+i+1}.pkl")