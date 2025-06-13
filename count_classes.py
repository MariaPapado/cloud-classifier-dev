import os
import numpy as np
from PIL import Image
import re

with open('./data_full_pkl/train.txt', 'r') as f:
    ids = f.read().splitlines()

classes = ['GOOD', 'OK', 'SEMIBAD', 'BAD']

counts = [0]*len(classes)

for id in ids:
    idx = id.find('_')
    match = id[idx+1:-4]
    
    idx_class = classes.index(match)
    counts[idx_class] = counts[idx_class] + 1

print(counts)
counts_sum = (sum(counts))

weights = []

for c in counts:
    weights.append(round(1-((c/counts_sum)), 2))

print(weights)    
    
