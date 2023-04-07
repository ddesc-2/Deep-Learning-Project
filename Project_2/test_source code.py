import numpy as np, pandas as pd, gc
import cudf, cuml, cupy
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv('/kaggle/input/shopee-product-matching/train.csv')
test = pd.read_csv('/kaggle/input/shopee-product-matching/test.csv')
tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
train['target'] = train.label_group.map(tmp)

test.insert(test.shape[1] , 'label_group' , 0)
tmp = test.groupby('image_phash').posting_id.agg('unique').to_dict()
test['phash'] = test.image_phash.map(tmp)

labelencoder = LabelEncoder()
train=pd.DataFrame(train)
train['label_group'] = labelencoder.fit_transform(train['label_group'])
test['image_preds'] = test['image_phash']

def getImagePaths(path):
    image_names = []
    for dirname, _, imagenames in os.walk(path):
        for image in imagenames:
            fullpath = os.path.join(dirname, image)
            image_names.append(fullpath)
    return image_names

train_images_path = getImagePaths('/kaggle/input/shopee-product-matching/train_images')
test_images_path = getImagePaths('/kaggle/input/shopee-product-matching/test_images')



transform= transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
#dataset+dataloader處理
class Dataset(Dataset):
    def __init__(self, train_file, img_dir, image_fullpath, transform=transform):

        self.img_labels = train_file
        self.img_dir = img_dir
        self.image_fullpath = image_fullpath
        self.transform = transform

    def __len__(self):
        return len(self.image_fullpath)

    def __getitem__(self, idx):
        image = read_image(self.image_fullpath[idx])
        label = self.img_labels.iloc[idx, 4]
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = Dataset(train, '/kaggle/input/shopee-product-matching/train_images'
                              , train_images_path, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32768, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(2048, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

CNN = CNN(11014).to(device)

test_dataset = Dataset(test, '/kaggle/input/shopee-product-matching/test_images'
                              , test_images_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


weight = '/kaggle/input/mymodel/model_state_dict_final.pt'


CNN.load_state_dict(torch.load(weight))

CNN.eval()
with torch.no_grad():
    for i , (images, labels) in enumerate(test_dataloader):
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        outputs = CNN(images)
        _, predicted = torch.max(outputs, 1)
        preds = int(predicted)
        test.iloc[i,4] = preds
        

tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
test['img_pred'] = test.label_group.map(tmp)

def combine_for_sub(row):
    x = np.concatenate([row.phash,row.img_pred])
    return ' '.join( np.unique(x) )

test['matches'] = test.apply(combine_for_sub,axis=1)
test[['posting_id','matches']].to_csv('submission.csv',index=False)
sub = pd.read_csv('submission.csv')
print(sub)
