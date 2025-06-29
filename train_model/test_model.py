###kanei testing me vash tis testing eikones pou exw swsei..

import torchvision
import torch
from torch.utils.data import DataLoader
from dataloader import DataClassifier
import pickle
from tqdm import tqdm
import torchvision.transforms as transforms
import torchmetrics
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import shutil
from PIL import Image
import cv2

with open('/home/maria/mycloud-classifier/data_full_pkl/test.txt', 'r') as f:
    test_ids = f.read().splitlines()

test_ids = test_ids[:20]


data_dir = '/home/maria/mycloud-classifier/data_full_pkl/images2/'

model = torchvision.models.resnet50().cuda()
model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
model.fc = torch.nn.Linear(2048, 4).cuda()
#model.load_state_dict(torch.load('./saved_models_success12/net_12.pt', weights_only=True))
model.load_state_dict(torch.load('./saved_models_success/net_14.pt', weights_only=True))

NumClasses = 4
c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)

save_folder = './PREDS/'
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)

# Create the directory
os.mkdir(save_folder)
save_folder_imgs = save_folder + 'IMGS/'
save_folder_clouds = save_folder + 'CLOUDS/'
os.mkdir(save_folder_imgs)
os.mkdir(save_folder_clouds)

def normalize_img(img, mean=[0.321,  0.338, 0.317], std=[0.276, 0.273, 0.273]):  # 128x128 (datasetFPv1)

    """Normalize image by subtracting mean and dividing by std."""
    #img_array = np.asarray(img, dtype=np.uint8)
    normalized_img = np.empty_like(img, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img[..., i] - mean[i]) / std[i]
    
    return normalized_img


model.eval()    

classes = ['GOOD', 'OK', 'SEMIBAD', 'BAD']



with torch.no_grad():
    for i, id in enumerate(tqdm(test_ids)):
#      if '363926' in id:
#        print(id)

        idx = id.find('_')
        match = id[idx+1:-4]
        y = classes.index(match)
        y_true = np.array([y])

        Ximg_0 = Image.open('/home/maria/mycloud-classifier/data_full_pkl/images2/{}'.format(id))
        Ximg = np.array(Ximg_0)/255.
        Ximg = normalize_img(Ximg)

        Xcloud = Image.open('/home/maria/mycloud-classifier/data_full_pkl/cloud_masks2/{}'.format(id))
        Xcloud.save('./PREDS/CLOUDS/{}'.format(id))
        Xcloud = np.array(Xcloud)/255.
        Xcloud = np.expand_dims(Xcloud, 2)

        Xfin = np.concatenate((Ximg,Xcloud), 2)

        Xfin = np.transpose(Xfin, (2,0,1))
        Xfin = torch.from_numpy(Xfin).unsqueeze(0).float().cuda()
        out = model(Xfin)
        prob = torch.nn.functional.softmax(out, 1)
        print(prob)
        y_pred = out.argmax(dim=1)
        y_pred = y_pred.data.cpu().numpy()

        c_matrix = c_matrix + confusion_matrix(y_true, y_pred, labels=np.arange(NumClasses))

        Ximg_0.save('./PREDS/IMGS/{}_{}.png'.format(id[:-4], classes[y_pred[0]]))

print('VAL_c_matric')
print(c_matrix)



