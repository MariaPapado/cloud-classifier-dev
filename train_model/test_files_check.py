import requests
import pickle
import os
from pathlib import Path
import json
import ast
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm

# url of the model in production
# test_url = 'https://cloud-classifier-ml.orbitaleye.nl/predict-proba'
# url of the model in the test server, replace server IP address with your server!!
#test_url = 'http://10.10.100.4:8000/predict-proba'
test_url = 'http://0.0.0.0:8000/predict-proba'

######################################################################################################################################
data_dir = '/home/maria/mycloud-classifier/data_full_pkl/images/'

with open("/home/maria/mycloud-classifier/data_full_pkl/data_full_new.pkl", "rb") as f:
    results = pickle.load(f)

with open('/home/maria/mycloud-classifier/data_full_pkl/test.txt', 'r') as f:
    test_ids = f.read().splitlines()
#test_ids = test_ids[:20]

NumClasses = 4
c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)
#classes = ['GOOD', 'OK', 'SEMIBAD', 'BAD']
classes = ['BAD', 'SEMIBAD', 'OK', 'GOOD']
for _, res in enumerate(tqdm(results)):
    for tid in test_ids:
        if str(res['img_id']) in tid:
#          if '362063' in tid:
            print(res['img_path'])

            path = Path(res['img_path'])
            relative = path.as_posix().split("/data", 1)[1]

            file_path = '/img' + relative
            response = requests.get(test_url, json={'file_path':file_path, 'url': None, 'band_type': 'BGRN'})
#            print(response, response.text)


            data = json.loads(response.text)  # replace `your_str` with your actual string variable
            classification = ast.literal_eval(data["classification"])
            print('cl', classification)
            highest = max(classification, key=classification.get)
            y_pred = classes.index(highest)
            y_true = classes.index(res['label'])
            y_pred, y_true = np.array([y_pred]), np.array([y_true])
#            print(highest)  # e.g., 'OK'
            c_matrix = c_matrix + confusion_matrix(y_true, y_pred, labels=np.arange(NumClasses))


print(c_matrix)

#apotelesma tou paliou cloud-classifier...hmmm......
#[[109   7   2   4]
# [  0 110   4   3]
# [  0  13  12   6]
# [  0   7   4  64]]

# example image which should be GOOD but is classified as BAD
#file_path = '/img/20250511/SkySat_SKYWATCH_20250511_085520_ssc13_u0002_pansharpened_clip_0_5526330_dLgDPozUhU.tif'
# /mapserver/data/
#response = requests.get(test_url, json={'file_path':file_path, 'url': None, 'band_type': 'BGRN'})

#print(response, response.text) 

######################################################################################################################################



'''
# example image which should be GOOD but is classified as BAD
file_path = '/img/20250511/SkySat_SKYWATCH_20250511_085520_ssc13_u0002_pansharpened_clip_0_5526330_dLgDPozUhU.tif'
# /mapserver/data/
response = requests.get(test_url, json={'file_path':file_path, 'url': None, 'band_type': 'BGRN'})

print(response, response.text) 
'''
