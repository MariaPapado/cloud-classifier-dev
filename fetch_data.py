#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import myUNF
import requests
import torch

#for res in results:
#    if 'SkySat_SKYWATCH_20250520_072601_ssc2_u0001_pansharpened_clip_1_18900618_uviehcIY5d' in res['filepath']:

#        feature = {
#            "type": "Feature",
#            "geometry": mapping(res['valid_area']),
#            "properties": {}
#        }

#        with open("multipolygon.geojson", "w") as f:
#            json.dump(feature, f)

#        feature = {
#            "type": "Feature",
#            "geometry": mapping(res['bounds']),
#            "properties": {}
#        }

#        with open("multipolygon_bounds.geojson", "w") as f:
#            json.dump(feature, f)



#data_tmp = cur.fetchall()

#desc = cur.description
#col_names = [x[0] for x in desc]
#results = [dict(zip(col_names, x)) for x in data_tmp] 




def normalize_img(img, mean=[0.467, 0.504, 0.493], std=[0.306, 0.307, 0.322]):  #

    img_array = np.asarray(img, dtype=np.float32)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[i, ...] = (img_array[i, ... ] - mean[i]) / std[i]
    
    return normalized_img

def postprocess(image):
    # Find contours
    contours_first, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for con in contours_first:
        area = cv2.contourArea(con)
        if area>500:
            contours.append(con)


    output = cv2.drawContours(np.zeros((image.shape[0], image.shape[1],3)), contours, -1, (255,255,255), thickness=cv2.FILLED)

    # Smooth the mask
    blurred_mask = cv2.GaussianBlur(output, (25, 25), 0)

    # Threshold back to binary
    _, smoothed_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

    return smoothed_mask

def pad_left(arr, n=256):
    deficit_x = n - arr.shape[1] % n
    deficit_y = n - arr.shape[2] % n
    if not (arr.shape[1] % n):
        deficit_x = 0
    if not (arr.shape[2] % n):
        deficit_y = 0
    arr = np.pad(arr, ((0, 0), (deficit_x, 0), (deficit_y, 0)), mode="reflect")
    return arr, deficit_x, deficit_y




def normalise_bands(image, percentile_min=2, percentil_max=98):
    tmp = []

    for i in range(image.shape[0]):
        c = image[i,:,:]
#        min_val = np.nanmin(c[np.nonzero(c)])
#        max_val = np.nanmax(c[np.nonzero(c)])

        min_val = np.nanmin(c)
        max_val = np.nanmax(c)

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

model = myUNF.UNetFormer(num_classes=2)

model.load_state_dict(torch.load('vhr_cloud_net_18.pt', weights_only=True))
model=model.cuda()
model = model.eval()

final_data = []

for _, r in enumerate(tqdm(results)):

        mydict = {}
        img_ = load_tif_image(r)
        #img = np.transpose(img, (1,2,0))  #/255.

        img = (img_).astype(np.float32) #[:,:,:3]
        img = normalize_img(img)

#        img = np.transpose(img, (2,0,1))
        img, dx, dy = pad_left(img)
        img = torch.from_numpy(img).float().unsqueeze(0).cuda()


        with torch.no_grad():
            output = model(img).data.cpu().numpy()[0]
        #        output = np.argmax(output[[0,3,2,1]],axis=0)
        output = np.argmax(output, axis=0)
        output = output[dx:, dy:]
        output = postprocess(output.astype(np.uint8))
        output = output[:,:,0]
        output = output.astype(np.uint8)

        output = cv2.resize(output, (384,384), cv2.INTER_NEAREST)
        output = Image.fromarray(output)

        img_save = cv2.resize(np.transpose(img_, (1,2,0)), (384,384), cv2.INTER_NEAREST)
        img_save = img_save*255

        img_save = np.array(img_save, dtype=np.uint8)

        img_save = Image.fromarray(img_save)

        label = r['label']
        img_id = r['img_id']


#        mydict['img'] = img
#        mydict['label'] = label
#        mydict['img_id'] = img_id
#        final_data.append(mydict)

        img_save.save('./data_full_pkl/images/{}_{}.png'.format(img_id, label))
        output.save('./data_full_pkl/cloud_masks/{}_{}.png'.format(img_id, label))




#with open("./data_full_pkl/images_labels_new.pkl", "wb") as f:
#    pickle.dump(final_data, f)
        
