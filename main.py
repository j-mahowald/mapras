from tool import VisualSearchTool
import os
from PIL import Image
from contextlib import contextmanager
import warnings

import os
from PIL import Image

@contextmanager
def allow_large_images():
	"""Context manager to temporarily allow loading large images without warnings"""
	# Store original settings
	original_limit = Image.MAX_IMAGE_PIXELS

	# Remove limit and suppress warnings
	Image.MAX_IMAGE_PIXELS = None

	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", ".*decompression bomb.*", UserWarning)
		try:
			yield
		finally:
			# Always restore original limit
			Image.MAX_IMAGE_PIXELS = original_limit

def resize_large_images_simple(input_dir="./images", output_dir="./images_resized", max_pixels=89000000, max_dim=4096):
	"""Resize images so that total pixels <= max_pixels and largest dimension <= max_dim"""
	import os
	print('beginning resizing')

	os.makedirs(output_dir, exist_ok=True)

	image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
	image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

	resized_count = 0

	with allow_large_images():
		for filename in image_files:
			input_path = os.path.join(input_dir, filename)
			output_path = os.path.join(output_dir, filename)

			try:
				with Image.open(input_path) as img:
					width, height = img.size
					total_pixels = width * height

					# Calculate scale factors for both constraints
					scale_pixel = (max_pixels / total_pixels) ** 0.5 if total_pixels > max_pixels else 1.0
					scale_dim = max_dim / max(width, height) if max(width, height) > max_dim else 1.0
					scale_factor = min(scale_pixel, scale_dim)

					if scale_factor < 1.0:
						new_width = int(width * scale_factor)
						new_height = int(height * scale_factor)

						# Ensure at least 1 pixel in each dimension
						new_width = max(1, new_width)
						new_height = max(1, new_height)

						# Resize and save
						resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
						resized_img.save(output_path, optimize=True, quality=85)

						print(f"Resized {filename}: {width}x{height} → {new_width}x{new_height}")
						resized_count += 1
					else:
						# Copy as-is if under limit
						img.save(output_path)

			except Exception as e:
				print(f"Error processing {filename}: {e}")

	print(f"\nProcessed {len(image_files)} images, resized {resized_count}")
	return output_dir

# resize_large_images_simple(input_dir="./images", output_dir="./images_resized_again", max_pixels=15000000, max_dim=2048)

any_too_large = False
count=0
for fname in os.listdir("./images"):
	if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
		path = os.path.join("./images", fname)
		with Image.open(path) as img:
			if img.width * img.height > 89478485:
				count+=1
				print(f"{fname} has more than 89478485 pixels: {img.width * img.height}")
				any_too_large = True
print(count)
if not any_too_large:
	print("No images in images_resized have more than 89478485 pixels")
# resize_large_images_simple(input_dir="./images_resized", output_dir="./images_resized", max_pixels=89478485, max_dim=4096)

tool = VisualSearchTool(corpus_path = './corpus')

list_to_load = [os.path.join("./images", fname) for fname in os.listdir("./images") if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
# add images using list_to_load in batches
tool.add_images(list_to_load)
# Search for similar documents
results = tool.search("Panoramic image of Kansas City", k=3)

# Results contain the actual documents with similarity scores
print('results: ', results[0])
for result in results[0]:
	print(f"Rank {result['rank']}: {result['document'].title}")
	print(f"Score: {result['score']:.3f}")
	# Access the actual image: result['document'].image
