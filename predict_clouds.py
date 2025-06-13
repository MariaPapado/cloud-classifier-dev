#!/usr/bin/env python
# coding: utf-8

# In[1]:
import base64
import requests
import cv2
import os
import os.path
import pickle
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pimsys.regions.RegionsDb import RegionsDb
from skimage.transform import resize
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis, entropy
from sklearn.metrics import classification_report
#from sklearn.metrics import plot_confusion_matrix, confusion_matrix, f1_score
from sklearn.preprocessing import normalize
from sklearn.utils import class_weight
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from pathlib import Path
import random
import orbital_vault as ov

from pathlib import Path
import gc
import psycopg2 as psycopg
from shapely import wkb
from shapely.geometry import mapping
import json
import random
from PIL import Image

def normalise_bands(image, percentile_min=2, percentil_max=98):
    tmp = []

    for i in range(image.shape[0]):
        c = image[i,:,:]
        min_val = np.nanmin(c[np.nonzero(c)])
        max_val = np.nanmax(c[np.nonzero(c)])
        c_2 = (c-min_val)/(max_val-min_val)
        c_valid = c_2[np.logical_and(c_2 > 0, c_2 < 1.0)]
        perc_2 = np.nanpercentile(c_valid, 2)
        perc_98 = np.nanpercentile(c_valid, 98)
        c_scaled = np.clip((c_2 - perc_2) / (perc_98 - perc_2),0,1)
        tmp.append(c_scaled)
    org_img = np.stack(tmp, axis=0)

    return org_img

def load_tif_image(layer):
    # Define mapserver path
    # Get path to after image


    bands = list(layer['band_type'])
    indexes = [bands.index("R")+1, bands.index("G")+1, bands.index("B")+1]        

    robject = rasterio.open(layer['img_path'])

    r, g, b = robject.read(indexes[0]), robject.read(indexes[1]), robject.read(indexes[2])

    img = np.stack([r, g, b], axis=0)
    #print('minmax', np.min(img), np.max(img))


#    print(path_tif)
    # Load image

    # Normalize bands on min and max
    img = normalise_bands(img)

    # Normalize
    # img = exposure.equalize_adapthist(np.moveaxis(img, 0, -1), 100)
    # img = np.moveaxis(img, -1, 0)
    # Get tif bounds
#        image_bounds = list(robject.bounds)
#        image_poly = geometry.Polygon.from_bounds(image_bounds[0], image_bounds[1], image_bounds[2], image_bounds[3])
    return img #, robject.transform, image_bounds, image_poly

with open("./data_full_pkl/data_full_new.pkl", "rb") as f:
    results = pickle.load(f)

# In[6]:

print(len(results))
results = [dict(t) for t in {tuple(d.items()) for d in results}]
print('clean', len(results))

quality = [res['label'] for res in results]
quality = np.array(quality)
print('GOOD: ',np.sum(quality=='GOOD'))
print('OK: ',np.sum(quality=='OK'))
print('SEMIBAD: ',np.sum(quality=='SEMIBAD'))
print('BAD: ',np.sum(quality=='BAD'))

addr = 'http://0.0.0.0:8001'
test_url = addr + '/predict'


ids = os.listdir('./data_full_pkl/images/')

for _, res in enumerate(tqdm(results)):

    image_before = load_tif_image(res)


    image_before = (image_before).astype(np.float32)
    image_before = np.transpose(image_before, (1,2,0))  #(1547, 2410, 3)
    width, height = image_before.shape[0], image_before.shape[1]
    #print('wh', width, height)
    image_before_d = base64.b64encode(np.ascontiguousarray(image_before).tobytes()).decode("utf-8")

    response = requests.post(test_url, json={'imageData':image_before_d, 'width': width, 'height': height})


    if response.ok:
        #print('ok')
        response_result = json.loads(response.text)
        response_result_data = base64.b64decode(response_result['result'])
        result = np.frombuffer(response_result_data,dtype=np.uint8)
        #print('rrr', result.shape, np.unique(result))
        mask = result.reshape(image_before.shape[:2])
    else:
        print('no')

    mask = cv2.resize(mask, (384,384), cv2.INTER_NEAREST)
    mask = Image.fromarray(mask)

    mask.save('./data_full_pkl/cloud_masks/{}_{}.png'.format(res['img_id'], res['label']))


#        img.save('./data_full_pkl/images/{}_{}.png'.format(img_id, label))



#with open("./data_full_pkl/images_labels_new.pkl", "wb") as f:
#    pickle.dump(final_data, f)
        
