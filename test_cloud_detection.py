##test of highres cloud detection model alone can act as a cloud classifier by calculating cloud percentages

from fastapi import FastAPI
import uvicorn
import cv2
from pydantic import BaseModel
from pimsys.regions.RegionsDb import RegionsDb
import os
import torch
import cv2
import json
import base64
import numpy as np
from shapely import geometry
import myUNF
import rasterio
import requests
import pickle


def process_file(res):
    #org_img = read_file(res['filepath_tif'], None, None)
    bands = list(res['band_type'])
    indexes = [bands.index("R")+1, bands.index("G")+1, bands.index("B")+1]        
    robject = rasterio.open('/cephfs'+res['filepath_tif'])
    if len(robject.indexes) == 4:
        r, g, b = robject.read(indexes[0]), robject.read(indexes[1]), robject.read(indexes[2])
        scaled = []
        # scale channels and reorder
        for c in [r, g , b]:
            min_val = np.nanmin(c[np.nonzero(c)])
            max_val = np.nanmax(c[np.nonzero(c)])
            c_2 = (c-min_val)/(max_val-min_val)
            c_valid = c_2[np.logical_and(c_2 > 0, c_2 < 1.0)]
            perc_2 = np.nanpercentile(c_valid, 2)
            perc_98 = np.nanpercentile(c_valid, 98)
            c_scaled = np.clip((c_2 - perc_2) / (perc_98 - perc_2),0,1)
            scaled.append(c_scaled)
        org_img = np.stack(scaled, axis=-1)
    else:
        org_img = None

    res['img'] = org_img

    return res

def normalise_bands(image, percentile_min=2, percentil_max=98):
    tmp = []

#    for i in range(image.shape[0]):
#        perc_2 = np.percentile(image[i, :, :], percentile_min)
#        perc_98 = np.percentile(image[i, :, :], percentil_max)
#        band = (image[i, :, :] - perc_2) / (perc_98 - perc_2)
#        band[band < 0] = 0.
#        band[band > 1] = 1.
#        tmp.append(band)

        # scale channels and reorder
    for i in range(image.shape[0]):
        c = image[i,:,:]
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

    robject = rasterio.open(layer['filepath'])

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
    image_bounds = list(robject.bounds)
    image_poly = geometry.Polygon.from_bounds(image_bounds[0], image_bounds[1], image_bounds[2], image_bounds[3])
    return img, robject.transform, image_bounds, image_poly



addr = 'http://0.0.0.0:8001'
test_url = addr + '/predict'
#test_url = 'http://cloud-detection.stg.orbitaleye-dev.nl/predict'


with open("data.pkl", "rb") as f:
    data = pickle.load(f)
print(data[0])

def find_class(res):
    idx = np.where(res==255)
    count_pixels = len(idx[0])
    area = res.shape[0]*res.shape[1]
    perc = (count_pixels/area)*100

    if perc>=80:
        pred_class = 'BAD'
    elif 30<perc<80:
        pred_class = 'SEMIBAD'
    elif 10<perc<=30:
        pred_class = 'OK'
    else:
        pred_class = 'GOOD'

    return pred_class, perc


from tqdm import tqdm
#/mapserver/data/20250520/SkySat_SKYWATCH_20250520_125238_ssc1_u0001_pansharpened_clip_0_11716934_zXe2CkWW03.tif
#data = [i for i in data if '/mapserver/data/20250520/SkySat_SKYWATCH_20250520_125238_ssc1_u0001_pansharpened_clip_0_11716934_zXe2CkWW03.tif'==i['filepath']]
data = [dict(t) for t in {tuple(d.items()) for d in data}]
print('lennnnnnnnnnnn', len(data))
quality = [res['user_classification'] for res in data if (res['user_classification']!=None)]
quality = np.array(quality)
print('GOOD: ',np.sum(quality=='GOOD'))
print('OK: ',np.sum(quality=='OK'))
print('SEMIBAD: ',np.sum(quality=='SEMIBAD'))
print('BAD: ',np.sum(quality=='BAD'))
#print(data)
#data = data[:10]
from sklearn.metrics import confusion_matrix
y_pred = []
y_true = []
for m, res in enumerate(tqdm(data)):
#    layer_1 = '/cephfs' + res['filepath']
    #print(res)
    try:
        image_before, tif_transform, image_bounds, image_poly = load_tif_image(res)

        image_before = (image_before).astype(np.float32)[:3,:,:] 
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
        pred_class, perc = find_class(mask)
        y_pred.append(pred_class)
        y_true.append(res['user_classification'])
    except:
        pass
#    cv2.imwrite('./check_new/img_{}.png'.format(m), image_before[:,:,[2,1,0]]*255)
#    cv2.imwrite('./check_new/mask_{}_{}_{}_{}.png'.format(m, perc, pred_class, res['user_classification']), mask)

cm = confusion_matrix(y_true, y_pred, labels=['GOOD', 'OK', 'SEMIBAD', 'BAD'])
    #cv2.imwrite('mask_{}.png'.format(m), res)
print(cm)

