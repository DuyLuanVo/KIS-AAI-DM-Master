import numpy as np

# Read npy file, folder has 307 images so shape will be 307, 512
# because it's extracted from clip vit32b model so it has 512 dimensions
clip_features = np.load('L21_V001.npy', mmap_mode=None)  # shape: (307, 512)


def load_clip_features(npy_path):
    # Use mmap to save RAM if file is large
    return np.load(npy_path, mmap_mode='r')


def get_vector_for_image(features, image_name):
    # image_name format "001.jpg" → index = 0
    idx = int(image_name.split('.')[0]) - 1
    return features[idx]


# Example usage
features = load_clip_features('L21_V001.npy')
img_name = "L21_V001_001.jpg"
vector = get_vector_for_image(features, img_name)
print(f"Vector for {img_name}:", vector[:8], "...")  # Print first 8 numbers
print(vector.shape)
