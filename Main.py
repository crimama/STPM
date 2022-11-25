import os 
import warnings 
warnings.filterwarnings('ignore')
from tqdm import tqdm 
import numpy as np 
import cv2 
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from torchvision import transforms 
from IPython.display import clear_output
import wandb 
import argparse 

from src import Datadir_init,MVtecADDataset
from src import ResNet18,get_networks
from src import STPM_detection, mkd_detection
from src import STPMLoss,MKDLoss


def load_gt(root, cls):
    gt = []
    gt_dir = os.path.join(root, cls, 'ground_truth')
    sub_dirs = sorted(os.listdir(gt_dir))
    for sb in sub_dirs:
        for fname in sorted(os.listdir(os.path.join(gt_dir, sb))):
            temp = cv2.imread(os.path.join(gt_dir, sb, fname), cv2.IMREAD_GRAYSCALE)
            temp = cv2.resize(temp, (256, 256)).astype(np.bool)[None, ...]
            gt.append(temp)
    gt = np.concatenate(gt, 0)
    return  gt

def preprocess(cfg,augmentation=None):
    #mk save dir 
    try:
        os.mkdir(f"./Save_models/{cfg['model']}_{cfg['loss_function']}/{cfg['class']}")
    except:
        pass
    #Seed fix 
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    #Data load 
    Data_dir = Datadir_init(cfg['root'],cfg['class'])
    train_dirs = Data_dir.train_load()
    test_dirs,test_labels = Data_dir.test_load()
    
    gt = load_gt(cfg['root'],cfg['class'])
    true_gt = np.zeros((len(test_labels), 256, 256), dtype=np.bool)
    true_gt[np.where(test_labels==1)[0]]= gt


    indx = int(len(train_dirs)*0.8)
    train_dset = MVtecADDataset(cfg,train_dirs[:indx],Augmentation=augmentation)
    valid_dset = MVtecADDataset(cfg,train_dirs[indx:])
    test_dset = MVtecADDataset(cfg,test_dirs,test_labels)

    train_loader = DataLoader(train_dset,batch_size=cfg['batch_size'],shuffle=True)
    valid_loader = DataLoader(valid_dset,batch_size=cfg['batch_size'],shuffle=False)
    test_loader = DataLoader(test_dset,batch_size=cfg['batch_size'],shuffle=False)
    return train_loader,valid_loader,test_loader,true_gt,test_labels 

   
def make_transform():
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def make_lossfunction(loss_function):
    if loss_function == 'mkd':
        return MKDLoss
    else:
        return STPMLoss         



def train_epoch(student,teacher,train_loader,criterion,optimizer,cfg):
    teacher.eval()
    student.train()
    train_loss = [] 
    for batch_imgs,_ in train_loader:
        batch_imgs = batch_imgs.to(cfg['device']).type(torch.float32)

        with torch.no_grad():
            feat_t = teacher(batch_imgs)
        feat_s = student(batch_imgs)

        loss = criterion(feat_t,feat_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        train_loss.append(loss.detach().cpu().numpy())
    return train_loss 

def valid_epoch(student,teacher,valid_loader,criterion,cfg):
    teacher.eval()
    student.eval()
    valid_loss = [] 
    for batch_imgs,_ in valid_loader:
        batch_imgs = batch_imgs.to(cfg['device']).type(torch.float32)

        with torch.no_grad():
            feat_t = teacher(batch_imgs)
        feat_s = student(batch_imgs)

        loss = criterion(feat_t,feat_s)


        valid_loss.append(loss.detach().cpu().numpy())
    return valid_loss     



def parse_arguments():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('-Class')
    parser.add_argument('-model')
    parser.add_argument('-loss_function')
    args = parser.parse_args() 
    return args 

if __name__ == "__main__":

#Configuration 
    cfg = {} 
    cfg['img_size']= 256 
    cfg['class_name'] = 'bottle'
    cfg['batch_size']= 32 
    cfg['lr'] = 0.4
    cfg['Epochs'] = 100 
    cfg['device'] = 'cuda:0'
    cfg['seed'] = 0 
    cfg['root'] = './Dataset'
    cfg['class'] = 'bottle'
    cfg['lambda'] = 0.01
    cfg['model'] = 'STPM'
    cfg['loss_function'] = 'MKD'
    device = cfg['device']

    args = parse_arguments()
    cfg['class'] = args.Class
    cfg['model'] = args.model
    cfg['loss_function'] = args.loss_function

#Wandb 
    wandb.init(project=f"{cfg['model']}_{cfg['loss_function']}",name=cfg['class'])

#Call 
    student,teacher = get_networks(cfg['model'])
    transform = make_transform()
    train_loader,valid_loader,test_loader,true_gt,test_labels = preprocess(cfg,transform)



    criterion = make_lossfunction(cfg['loss_function'])(cfg['model'])
    optimizer = torch.optim.Adam(student.parameters(),lr=cfg['lr'])
    detector = STPM_detection(test_loader,cfg['model'],cfg)
    clear_output()


    total_train_loss = [] 
    total_valid_loss = [] 
    best_valid_loss = np.inf 
    print('Training start')

    for epoch in tqdm(range(cfg['Epochs'])):
        train_loss = train_epoch(student,teacher,train_loader,criterion,optimizer,cfg)
        valid_loss = valid_epoch(student,teacher,valid_loader,criterion,cfg)

        print(f'\t epoch : {epoch+1} train loss : {np.mean(train_loss):.3f}')
        print(f'\t epoch : {epoch+1} valid loss : {np.mean(valid_loss):.3f}')

        total_train_loss.append(train_loss)
        total_valid_loss.append(valid_loss)

        
        #image_auroc = detector.auroc(teacher,student)
        pixel_auroc,image_auroc  = detector.test_inference(teacher,student,test_loader,true_gt,test_labels)
        print(f"\t Pixel AUROC : {pixel_auroc:.3f}")
        print(f"\t Image AUROC : {image_auroc:.3f}")


    #check point 
        if np.mean(valid_loss) < best_valid_loss:
            torch.save(student,f"./Save_models/{cfg['model']}_{cfg['loss_function']}/{cfg['class']}/best.pt")
            best_valid_loss = np.mean(valid_loss) 
            print(f'\t Model save : {epoch} | best loss : {best_valid_loss :.3f}')

        train_loss = np.mean(train_loss)
        valid_loss = np.mean(valid_loss)
        wandb.log({ 'loss'   : train_loss,
                    'valid_loss'  : valid_loss,
                    'Pixel AUROC' : pixel_auroc,
                    'image_auroc' : image_auroc})