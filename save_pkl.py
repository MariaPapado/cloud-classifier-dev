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


creds_mapserver = ov.get_image_broker_credentials()

creds = ov.get_sarccdb_credentials()

config = {
    "regions_database": {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
    }
}


query = """
SELECT *
FROM public.optical_image_base
WHERE source = 'SkySat'
  AND capture_timestamp > '2025-04-01'
  AND capture_timestamp < '2025-05-30'
  AND width < 3000
  AND height < 3000

"""

#  AND capture_timestamp > '2025-05-01'
#  AND band_type = 'BGRN';


#  AND capture_timestamp < '2025-04-30'
#  AND capture_timestamp > '2023-01-01'




connection_string = f"dbname={creds['database']} host=sar-ccd-db.orbitaleye.nl password={creds['password']} port={creds['port']} user={creds['user']}"
conn = psycopg.connect(connection_string)

# In[2]:

cur = conn.cursor()
cur.execute(query)

columns = [desc[0] for desc in cur.description]
rows = cur.fetchall()

#results = []
#for row in rows:
#    row_dict = dict(zip(columns, row))
#    if 'valid_area' in row_dict and isinstance(row_dict['valid_area'], str):
#        row_dict['valid_area'] = wkb.loads(bytes.fromhex(row_dict['valid_area']))
#    results.append(row_dict)

#    if 'bounds' in row_dict and isinstance(row_dict['bounds'], str):
#        row_dict['bounds'] = wkb.loads(bytes.fromhex(row_dict['bounds']))
#    results.append(row_dict)


#    if 'valid_pixels' in row_dict and isinstance(row_dict['valid_pixels'], str):
#        row_dict['valid_pixels'] = wkb.loads(bytes.fromhex(row_dict['valid_pixels']))
#    results.append(row_dict)

#print(results[0])

results = []
for row in rows:
    row_dict = dict(zip(columns, row))
    results.append(row_dict)


cur.close()
conn.close()


results = [res for res in results if res['filepath']!=None]
results = [res for res in results if res['filepath'].split('.')[-1]=='tif']

results2= []

for res in results:
    mydict = {}

    try:

        if res['user_classification']==None:
            if res['classification'] in ['GOOD', 'OK']:
                label = res['classification']


        if res['user_classification']in ['GOOD', 'OK']:
            label = res['user_classification']

        if res['user_classification'] in ['SEMIBAD', 'BAD']:
            label = res['user_classification']

    #    if res['classification'] in ['GOOD', 'OK']:
    #        if res['user_classification']==None:
    #            label = res['classification']
    #        else:
    #            label = res['user_classification']
    #    
    #    if (res['classification'] in ['SEMIBAD', 'BAD']):
    #        if res['user_classification'] != None:
    #            label = res['user_classification']
        mydict['img_path'] = '/cephfs' + res['filepath']    
        mydict['img_id'] = res['id']
        mydict['band_type'] = res['band_type']
        mydict['label'] = label
        results2.append(mydict)
    except:
        print('ok')
        pass





# In[6]:

def check_tif_files(res):
    tif_path = res['filepath'].split('.')[0]+'.tif'
    if os.path.isfile('/cephfs'+tif_path):
        res['filepath_tif'] = tif_path
    else:
        res['filepath_tif'] = None
    
    return res

#results = Parallel(n_jobs=16, prefer='processes')(delayed(check_tif_files)(res) for res in tqdm(results))

#results = [check_tif_files(res) for res in tqdm(results)]

print(len(results2))
results2 = [dict(t) for t in {tuple(d.items()) for d in results2}]
print('clean', len(results2))

#quality = [res['user_classification'] for res in results2 if (res['user_classification']!=None)]
quality = [res['label'] for res in results2]
quality = np.array(quality)
print('GOOD: ',np.sum(quality=='GOOD'))
print('OK: ',np.sum(quality=='OK'))
print('SEMIBAD: ',np.sum(quality=='SEMIBAD'))
print('BAD: ',np.sum(quality=='BAD'))


with open("./data_full_pkl/data_full_new.pkl", "wb") as f:
    pickle.dump(results2, f)
