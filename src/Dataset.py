import numpy as np 
from torch.utils.data import Dataset 
import torchvision.transforms as transforms 
from PIL import Image 
import pandas as pd 
from glob import glob 
import os 
import cv2 

class MVtecADDataset(Dataset):
    def __init__(self,cfg,img_dirs,labels=None,Augmentation=None):
        super(MVtecADDataset,self).__init__()
        self.cfg = cfg 
        self.dirs = img_dirs 
        self.augmentation = self.__init_aug__(Augmentation)
        self.labels = self.__init_labels__(labels)

    def __len__(self):
        return len(self.dirs)

    def __init_labels__(self,labels):
        if np.sum(labels) !=None:
            return labels 
        else:
            return np.zeros(len(self.dirs))
    
    def __init_aug__(self,Augmentation):
        if Augmentation == None:
            augmentation = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize([256, 256]),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        else: 
            augmentation = Augmentation 
        return augmentation                                      

    def __getitem__(self,idx):
        img_dir = self.dirs[idx]
        img = Image.open(img_dir)
        if np.array(img).shape[-1] != 3:
            img = cv2.cvtColor(np.array(img),cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img)
        img = self.augmentation(img)

        if np.sum(self.labels) !=None:
            return img,self.labels[idx] 
        else:
            return img

class Datadir_init:
    def __init__(self,root='./Dataset',cls='hazelnut'):
        self.Dataset_dir = os.path.join(root,cls) 
        
    def test_load(self):
        test_label_unique = pd.Series(sorted(glob(f'{self.Dataset_dir}/test/*'))).apply(lambda x : x.split('/')[-1]).values
        test_label_unique = np.delete(test_label_unique,np.where(test_label_unique=='good')[0])
        test_label_unique = np.append(test_label_unique,'good')
        test_label_unique = {key:value for value,key in enumerate(test_label_unique)}
        self.test_label_unique = test_label_unique 

        test_dir = f'{self.Dataset_dir}/test/'
        label = list(test_label_unique.keys())[0]

        test_img_dirs = [] 
        test_img_labels = [] 
        for label in list(test_label_unique.keys()):
            img_dir = sorted(glob(test_dir +f'{label}/*'))
            img_label = np.full(len(img_dir),test_label_unique[label])
            test_img_dirs.extend(img_dir)
            test_img_labels.extend(img_label)
        
        test_img_labels = np.array(test_img_labels )
        test_img_labels = np.where(test_img_labels==test_label_unique['good'],test_img_labels,1)
        test_img_labels = np.where(test_img_labels==1,test_img_labels,0)
        return np.array(test_img_dirs),test_img_labels 

    def train_load(self):
        train_img_dirs = sorted(glob(f'{self.Dataset_dir}/train/good/*.png'))
        return np.array(train_img_dirs) 