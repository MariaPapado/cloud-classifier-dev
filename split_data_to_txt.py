import os
import numpy as np
import random


# Folder containing images
folder_path = './data_full_pkl/images/'


classes = ['GOOD', 'OK', 'SEMIBAD', 'BAD']


# Get all image IDs (assuming file names are the IDs)
image_ids = [f for f in os.listdir(folder_path) if f.endswith('.png')]

GOOD_samples = []
for id in image_ids:
    idx = id.find('_')
    match = id[idx+1:-4]

    if match=='GOOD':
        GOOD_samples.append(id)


random.shuffle(GOOD_samples)
GOOD_samples_delete = GOOD_samples[:3000]
#GOOD_samples = list(set(GOOD_samples) - set(GOOD_samples_delete))
image_ids = list(set(image_ids) - set(GOOD_samples_delete))
image_class_ids = [[],[],[],[]]

for id in image_ids:
    idx = id.find('_')
    match = id[idx+1:-4]

    idx_class = classes.index(match)
    image_class_ids[idx_class].append(id)

for i in image_class_ids:
    print(len(i))

all_train_ids = []
all_val_ids = []
all_test_ids = []

for imids in image_class_ids:
    print('lll', len(imids))


    # Split the dataset into train, val, test (80%, 10%, 10%)
    train_size = int(0.9 * len(imids))
    val_size = int(0.05 * len(imids))
    test_size = len(imids) - train_size - val_size

    # Shuffle the image IDs
    random.shuffle(imids)

    # Create splits
    train_ids = imids[:train_size]
    val_ids = imids[train_size:train_size + val_size]
    test_ids = imids[train_size + val_size:]

    all_train_ids.extend(train_ids)
    all_val_ids.extend(val_ids)
    all_test_ids.extend(test_ids)

random.shuffle(all_train_ids)
random.shuffle(all_val_ids)
random.shuffle(all_test_ids)


print(len(all_train_ids))
print(len(all_val_ids))
print(len(all_test_ids))


# Write to text files
with open('./data_full_pkl/train.txt', 'w') as f:
    f.write("\n".join(all_train_ids))

with open('./data_full_pkl/val.txt', 'w') as f:
    f.write("\n".join(all_val_ids))

with open('./data_full_pkl/test.txt', 'w') as f:
    f.write("\n".join(all_test_ids))

print("Data split into train, val, and test and saved to text files.")
