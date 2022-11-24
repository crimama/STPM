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
import wandb 
import argparse 


from src import Datadir_init,MVtecADDataset
from src import ResNet18

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
        os.mkdir(f"./Save_models/{cfg['class']}")
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

def loss_fn(f_t,f_s):
    total_loss = 0 
    for t,s in zip(f_t,f_s):
        t,s = F.normalize(t,dim=1),F.normalize(s,dim=1)
        total_loss += torch.sum((t.type(torch.float32) - s.type(torch.float32)) ** 2, 1).mean()
    return total_loss

    
def make_transform():
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def train_epoch(student,teacher,train_loader,optimizer,cfg):
    teacher.eval()
    student.train()
    train_loss = [] 
    for batch_imgs,_ in train_loader:
        batch_imgs = batch_imgs.to(cfg['device']).type(torch.float32)

        with torch.no_grad():
            feat_t = teacher(batch_imgs)
        feat_s = student(batch_imgs)

        loss = loss_fn(feat_t,feat_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        train_loss.append(loss.detach().cpu().numpy())
    return train_loss 

def test_score(student,teacher,loader):
    teacher.eval()
    student.eval()
    loss_map = np.zeros((len(loader.dataset), 64, 64))
    i = 0 
    for batch_imgs,_ in loader:
        batch_imgs = batch_imgs.to(cfg['device']).type(torch.float32)
        with torch.no_grad():
            feat_s = student(batch_imgs)
            feat_t = teacher(batch_imgs)
        score_map = 1.
        for t,s in zip(feat_t,feat_s):
            t,s = F.normalize(t,dim=1),F.normalize(s,dim=1)
            sm = torch.sum((t - s) ** 2, 1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            # aggregate score map by element-wise product
            score_map = score_map * sm
        loss_map[i: i + batch_imgs.size(0)] = score_map.squeeze().cpu().data.numpy()
        i += batch_imgs.size(0)
    return loss_map 


def roc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc    
def test_inference(teacher,student,test_loader,true_gt,test_labels):
    scores = test_score(teacher,student,test_loader)
    scores = [cv2.resize(i,dsize=(256,256)) for i in scores]
    scores = np.stack(scores)
    pixel_auroc = roc(true_gt.flatten(),scores.flatten())
    image_auroc = roc(test_labels,scores.max(-1).max(-1))
    return pixel_auroc,image_auroc 

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('-Class')
    args = parser.parse_args() 
    return args 

if __name__ == "__main__":
    cfg = {} 
    cfg['img_size']= 256 
    cfg['class_name'] = 'bottle'
    cfg['batch_size']= 32 
    cfg['lr'] = 0.4
    cfg['Epochs'] = 100 
    cfg['device'] = 'cuda:0'
    cfg['seed'] = 0 
    cfg['root'] = './Dataset'
    cfg['class'] = 'transistor'
    
    args = parse_arguments()   
    cfg['class'] = args.Class
    wandb.init(project='STPM2',name=cfg['class'])

    device = cfg['device']
    student = ResNet18(Pretrained=False).to(device)
    teacher = ResNet18(Pretrained=True).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(),lr=cfg['lr'])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0)
    #scaler = torch.cuda.amp.GradScaler()

    transform = make_transform()
    train_loader,valid_loader,test_loader,true_gt,test_labels = preprocess(cfg,transform)

    total_train_loss = [] 
    total_valid_loss = [] 
    best_valid_loss = np.inf 
    print('Training start')

    for epoch in tqdm(range(cfg['Epochs'])):
        train_loss = train_epoch(student,teacher,train_loader,optimizer,cfg)
        valid_loss = test_score(student,teacher,train_loader).mean()

        print(f'\t epoch : {epoch+1} train loss : {np.mean(train_loss):.3f}')
        print(f'\t epoch : {epoch+1} valid loss : {valid_loss.item():.3f}')

        total_train_loss.append(train_loss)
        total_valid_loss.append(valid_loss)

    #check point 
        if valid_loss < best_valid_loss:
            torch.save(student,f"./Save_models/{cfg['class']}/best.pt")
            best_valid_loss = valid_loss 
            print(f'\t Model save : {epoch} | best loss : {best_valid_loss :.3f}')

        
        pixel_auroc,image_auroc = test_inference(teacher,student,test_loader,true_gt,test_labels)
        print(f"\t Pixel AUROC : {pixel_auroc:.3f}")
        print(f"\t Image AUROC : {image_auroc:.3f}")

        wandb.log({"train_loss":train_loss,
                    "valid_loss":valid_loss,
                    'Pixel AUROC' :pixel_auroc,
                    'image_auroc' : image_auroc})
    