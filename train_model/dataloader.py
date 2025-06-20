import cv2
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import rasterio
from shapely import geometry
from PIL import Image

class DataClassifier(Dataset):

    def __init__(self, data_dir, cloud_dir, data_ids, mode):
        super(DataClassifier, self).__init__()
        self.data_dir = data_dir
        self.data_ids = data_ids
        self.labels = ['GOOD', 'OK', 'SEMIBAD', 'BAD']
        self.mode = mode
        self.cloud_dir = cloud_dir


    def normalize_img(self, img, mean=[0.321,  0.338, 0.317], std=[0.276, 0.273, 0.273]):  # 128x128 (datasetFPv1)

        """Normalize image by subtracting mean and dividing by std."""
        normalized_img = np.empty_like(img, np.float32)

        for i in range(3):  # Loop over color channels
            normalized_img[..., i] = (img[..., i] - mean[i]) / std[i]
        
        return normalized_img


    def random_fliplr(self, img1):
        if random.random() > 0.5:
            img1 = np.fliplr(img1)

        return img1


    def random_flipud(self, img1):
        if random.random() > 0.5:
            img1 = np.flipud(img1)

        return img1


    def random_rot(self, img1):
        k = random.randrange(3) + 1

        img1 = np.rot90(img1, k).copy()

        return img1


    def __getitem__(self, index):

        data_ids_element = self.data_ids[index]
        Ximg = Image.open(self.data_dir + data_ids_element)
        Xcloud = Image.open(self.cloud_dir + data_ids_element)
        Xcloud = np.array(Xcloud)/255.
        Xcloud = np.expand_dims(Xcloud, 2)
        Ximg = np.array(Ximg)/255.
#        print('aaaaaaaaaaaaaaaaaa', Ximg.shape)

        idx = data_ids_element.find('_')
        match = data_ids_element[idx+1:-4]
        
        Yimg = self.labels.index(match)

        Ximg = self.normalize_img(Ximg)

        Xfin = np.concatenate((Ximg,Xcloud), 2)            
        #Xfin = Ximg
        if self.mode == 'train': 
            dice = random.randrange(0,3)
            if dice==0:
                Xfin = self.random_fliplr(Xfin)
            if dice==1:
                Xfin = self.random_flipud(Xfin)
            else:
                Xfin = self.random_rot(Xfin)
        Xfin = np.transpose(Xfin, (2,0,1))

        Xfin = Xfin.astype(np.float32)   

        return Xfin, Yimg

    def __len__(self):
        return len(self.data_ids)
