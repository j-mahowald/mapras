import requests
import os
import random
import pandas as pd
from PIL import Image
from io import BytesIO
import time
import json
pd.set_option('display.max_colwidth', None)


def main():
	merged_files = pd.read_csv('merged_files.csv')
	filtered = (
		merged_files
		.query('file_count == 1')
		.loc[~merged_files['p1_item_id'].str.contains('sanborn', na=False)]
	)
	sample = filtered.sample(n=4000, random_state=None)
	list_to_load_2 = sample['file_url'].tolist()
	# save list_to_load_2
	with open('list_to_load_2.txt', 'w') as f:
		for item in list_to_load_2:
			f.write("%s\n" % item)

	os.makedirs('./images', exist_ok=True)
	os.makedirs('./images_new', exist_ok=True)
	url_mapping = {}
	for idx, url in enumerate(list_to_load_2):
		start_time = time.time()
		try:
			response = requests.get(url)
			response.raise_for_status()
			img = Image.open(BytesIO(response.content))
			max_dim = 2048
			w, h = img.size
			if max(w, h) > max_dim:
				if w >= h:
					new_w = max_dim
					new_h = int(h * max_dim / w)
				else:
					new_h = max_dim
					new_w = int(w * max_dim / h)
				img = img.resize((new_w, new_h), Image.LANCZOS)
			filename = os.path.join('./images_new', f"{idx+2500}.jpg")
			img.save(filename)
			url_mapping[f"{idx+2500}.jpg"] = url
		except Exception as e:
			print(f"Failed to download {url}: {e}")

		# Dump mapping every 100 images
		if (idx + 1) % 100 == 0:
			with open('url_mapping_2500.json', 'w') as f:
				json.dump(url_mapping, f, indent=2)
			print(f"Saved URL mapping for {len(url_mapping)} images to url_mapping_2500.json")

	# Final save in case total is not a multiple of 100
	with open('url_mapping_2500.json', 'w') as f:
		json.dump(url_mapping, f, indent=2)
	print(f"Saved URL mapping for {len(url_mapping)} images to url_mapping_2500.json")

if __name__ == '__main__':
	main()
