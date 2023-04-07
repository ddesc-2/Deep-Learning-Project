import numpy as np, pandas as pd, gc
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
print(device)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
train['target'] = train.label_group.map(tmp)

labelencoder = LabelEncoder()
train=pd.DataFrame(train)
train['label_group'] = labelencoder.fit_transform(train['label_group'])


tmp = test.groupby('image_phash').posting_id.agg('unique').to_dict()
test['phash'] = test.image_phash.map(tmp)
test.head()

def getImagePaths(path):
    image_names = []
    for dirname, _, imagenames in os.walk(path):
        for image in imagenames:
            fullpath = os.path.join(dirname, image)
            image_names.append(fullpath)
    return image_names

train_images_path = getImagePaths('train_images')
test_images_path = getImagePaths('test_images')




transform= transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
#dataset處理
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

#train&test dataloader
train_dataset = Dataset(train, 'train_images'
                              , train_images_path, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


test.insert(test.shape[1] , 'labels' , 0)
test['labels'] = test['phash']
for i in range(3):
    test.iloc[i,4] = i
test_dataset = ShopeeDataset(test, 'test_images'
                              , test_images_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#model
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

    
# model training
CNN = CNN(11014).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNN.parameters(), lr=0.001, weight_decay=0.1)

CNN.train()

num_epochs = 100
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    the_last_loss = 100
    patience = 6
    trigger_times = 0
    for i, (images, labels) in enumerate(train_dataloader):  

        images = Variable(images.to(device))
        labels = Variable(labels.to(device))
        
        outputs = CNN(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    path = '/home/server/mdfk/'
    the_current_loss = loss.item()


    if the_current_loss > the_last_loss:
        trigger_times += 1
        print('trigger times:', trigger_times)

    if trigger_times >= patience:
        print('Early stopping!\nStart to test process.')
        file = open(path + 'model_state_dict_final.pt','w')
        dict_path = path + 'model_state_dict_final.pt'
        torch.save(CNN.state_dict(), dict_path)
    else:
        print('trigger times: 0')
        trigger_times = 0

        the_last_loss = the_current_loss
    idx +=1

print("OK!")
