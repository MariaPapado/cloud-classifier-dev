import numpy as np
from PIL import Image
import glob
import os
import random
import cv2
from tqdm import tqdm


with open('./data_full_pkl/train.txt', 'r') as f:
    ids = f.read().splitlines()

images = []
#ids = os.listdir('./data_full_pkl/images/')

# Accumulators
sum_pixels = np.zeros(3, dtype=np.float64)
sum_squared_diff = np.zeros(3, dtype=np.float64)
num_pixels = 0

# First pass: Compute mean
for _, id in enumerate(tqdm(ids)):
    image = Image.open('./data_full_pkl/images/{}'.format(id)) #[32:].reshape(511, 511, 13) 
    img = np.array(image)/255.

    h, w, _ = img.shape
    num_pixels += h * w
    sum_pixels += np.sum(img, axis=(0, 1))

# Compute mean
mean = sum_pixels / num_pixels
print(mean)

# Second pass: Compute variance
for _,id in enumerate(tqdm(ids)):
    image = Image.open('./data_full_pkl/images/{}'.format(id)) #[32:].reshape(511, 511, 13) 
    img = np.array(image)/255.

    # Compute squared differences from the mean
    squared_diff = (img - mean) ** 2
    sum_squared_diff += np.sum(squared_diff, axis=(0, 1))

# Compute variance
variance = sum_squared_diff / num_pixels

# Compute standard deviation
std = np.sqrt(variance)

print("Dataset Mean (R, G, B):", mean)
print("Dataset Std (R, G, B):", std)
