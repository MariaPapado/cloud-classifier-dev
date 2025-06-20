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
import torch.nn.functional as F

with open('/home/maria/mycloud-classifier//data_full_pkl/train.txt', 'r') as f:
    train_lines = f.read().splitlines()


with open('/home/maria/mycloud-classifier//data_full_pkl/val.txt', 'r') as f:
    val_lines = f.read().splitlines()

data_dir = '/home/maria/mycloud-classifier//data_full_pkl/images2/'
cloud_dir = '/home/maria/mycloud-classifier//data_full_pkl/cloud_masks2/'

batch_size = 8
trainset = DataClassifier(data_dir, cloud_dir, train_lines, 'train')
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)

valset = DataClassifier(data_dir, cloud_dir, val_lines, 'val')
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=False)

model = torchvision.models.resnet50(pretrained=True).cuda()
model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
model.fc = torch.nn.Linear(2048,4).cuda()



focal_loss = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
	model='focal_loss',
	alpha=[0.05, 0.10, .80, .15],
	gamma=2,
	reduction='mean',
	device='cpu',
	dtype=torch.float32,
	force_reload=False
)

focal_loss = focal_loss.cuda()

criterion = focal_loss # FocalLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

epochs = 20

NumClasses = 4
c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)
iter_ = 0


save_folder = 'saved_models'
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)

# Create the directory
os.mkdir(save_folder)

for epoch in range(1, epochs):

    model.train()
    train_loss = []
    correct = 0
    total = 0


    for i, batch in enumerate(tqdm(trainloader)):
        Ximg, y = batch
        Ximg, y = Ximg.float().cuda(), y.cuda()
        optimizer.zero_grad()

        out = model(Ximg)
        y_pred = out.argmax(dim=1)

        label_1hot = torch.nn.functional.one_hot(y, num_classes=NumClasses).cuda()

        loss = criterion(out, y)
#        loss = criterion(out, label_1hot.float())
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        c_matrix = c_matrix + confusion_matrix(y.data.cpu().numpy(), y_pred.data.cpu().numpy(), labels=np.arange(NumClasses))

        iter_ += 1

        if iter_ % 50 == 0:
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, epochs, i, len(trainloader),100.*i/len(trainloader), loss.item()))

    print('TRAIN_c_matric')
    print(c_matrix)
    print('TRAIN_mean_loss: ', np.mean(train_loss)) 



    model.eval()    
    c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)
    with torch.no_grad():
        val_loss = []
        for i, batch in enumerate(tqdm(valloader)):
            Ximg, y = batch
            Ximg, y = Ximg.float().cuda(), y.cuda()
            out = model(Ximg)
        
            loss = criterion(out, y)
#            loss = criterion(out, label_1hot.float())

            y_pred = out.argmax(dim=1)

            label_1hot = torch.nn.functional.one_hot(y, num_classes=NumClasses).cuda()


            val_loss.append(loss.item())

            c_matrix = c_matrix + confusion_matrix(y.data.cpu().numpy(), y_pred.data.cpu().numpy(), labels=np.arange(NumClasses))


    print('VAL_c_matric')
    print(c_matrix)
    print('VAL_mean_loss: ', np.mean(val_loss)) 

    torch.save(model.state_dict(), './' + save_folder + '/net_{}.pt'.format(epoch))
