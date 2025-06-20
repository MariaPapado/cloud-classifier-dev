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
import torchvision
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
from sklearn.metrics import confusion_matrix



def normalize_img_clouds(img, mean=[0.454, 0.493, 0.482], std=[0.300, 0.300, 0.316]):  #

    img_array = np.asarray(img, dtype=np.float32)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[i, ...] = (img_array[i, ... ] - mean[i]) / std[i]
    
    return normalized_img


def normalize_img_classifier(img, mean=[0.321,  0.338, 0.317], std=[0.276, 0.273, 0.273]):  # 128x128 (datasetFPv1)

    """Normalize image by subtracting mean and dividing by std."""
    normalized_img = np.empty_like(img, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img[..., i] - mean[i]) / std[i]
    
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
    return img 

with open("./data_full_pkl/data_full_new.pkl", "rb") as f:
    results = pickle.load(f)

with open('/home/maria/mycloud-classifier/data_full_pkl/test.txt', 'r') as f:
    test_ids = f.read().splitlines()
#test_ids = test_ids[:20]

# In[6]:

model = myUNF.UNetFormer(num_classes=2)

model.load_state_dict(torch.load('net_23.pt', weights_only=True))
model=model.cuda()
model = model.eval()


model_classifier = torchvision.models.resnet50(pretrained=True).cuda()
model_classifier.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
model_classifier.fc = torch.nn.Linear(2048, 4).cuda()
model_classifier.load_state_dict(torch.load('./train_new_model/saved_models_success/net_14.pt', weights_only=True))
model_classifier = model_classifier.eval()

classes = ['GOOD', 'OK', 'SEMIBAD', 'BAD']

NumClasses = 4
c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)


for m, res in enumerate(tqdm(results)):
#    layer_1 = '/cephfs' + res['filepath']
    #print(res)
#    try:
    for imgid in test_ids:
        idx = imgid.find('_')
        match = imgid[:idx]

        if match==str(res['img_id']):
#          if '362063'==match:

            print(res['img_id'], match)

            y = classes.index(res['label'])
            y_true = np.array([y])

            img_ = load_tif_image(res)
           # img_ = (img_).astype(np.float32) #[:,:,:3]
            #img = np.transpose(img, (1,2,0))  #/255.
            img_clouds = img_.copy()
            img_clouds = img_clouds.astype(np.float32)
            img_classifier = img_.copy()

            img_clouds = normalize_img_clouds(img_clouds)

    #        img = np.transpose(img, (2,0,1))
            img_clouds, dx, dy = pad_left(img_clouds)
            img_clouds = torch.from_numpy(img_clouds).float().unsqueeze(0).cuda()

            img_classifier = np.transpose(img_classifier, (1,2,0))

            img_save = img_classifier.copy()
            img_save = cv2.resize(img_save, (384,384), cv2.INTER_LINEAR)

            img_classifier = cv2.resize(img_classifier, (384,384), cv2.INTER_LINEAR)
            img_classifier = img_classifier*255
            img_classifier = np.array(img_classifier, dtype=np.uint8)
            img_classifier_pil = Image.fromarray(img_classifier)
            img_classifier_pil.save('./check_img_scratch.png')

            img_classifier = img_classifier/255.
            img_classifier = normalize_img_classifier(img_classifier)
            img_classifier = np.transpose(img_classifier, (2,0,1))

            with torch.no_grad():
                output = model(img_clouds).data.cpu().numpy()[0]
            #        output = np.argmax(output[[0,3,2,1]],axis=0)
            output = np.argmax(output, axis=0)

            output = output[dx:, dy:]
            output = postprocess(output.astype(np.uint8))
            output = output[:,:,0]

            output = output.astype(np.uint8)
            output = output/255.

            output = cv2.resize(output, (384,384), cv2.INTER_LINEAR)
            idx1 = np.where(output>=0.2)
            idx0 = np.where(output<0.2)
            output[idx0] = 0.
            output[idx1] = 1.

            #print(np.unique(output))

            output = np.expand_dims(output, 0)

            Xfin = np.concatenate((img_classifier, output), 0)


#            Xfin = np.transpose(Xfin, (2,0,1))
            Xfin = torch.from_numpy(Xfin).unsqueeze(0).float().cuda()

            out = model_classifier(Xfin)
#            np.save('checknp_scratch.npy', out.data.cpu().numpy()) ####to elegksa me to docker mexri edw einai idio!!!!!!

            prob = torch.nn.functional.softmax(out, 1)
            print(prob)
            y_pred = out.argmax(dim=1)
            print('ypred', y_pred)
            y_pred = y_pred.data.cpu().numpy()

            c_matrix = c_matrix + confusion_matrix(y_true, y_pred, labels=np.arange(NumClasses))

#            cv2.imwrite('./PREDS/IMGS/{}_{}_{}.png'.format(res['img_id'], res['label'], classes[int(y_pred[0])]), img_save*255)
#            cv2.imwrite('./PREDS/CLOUDS/{}_{}_{}.png'.format(res['img_id'], res['label'], classes[int(y_pred[0])]), output[0]*255)

print('VAL_c_matric')
print(c_matrix)


#VAL_c_matric epoch 14 success
#[[105   5  12   0]
# [  2  81  32   2]
# [  0   1  28   2]
# [  0   1  17  57]]

'''
##NOW
#docker
[[103   4  15   0]
 [  2  74  39   2]
 [  0   2  27   2]
 [  0   2  19  54]]

gpu2
[[105   5  12   0]
 [  2  81  32   2]
 [  0   1  28   2]
 [  0   1  17  57]]
 '''
