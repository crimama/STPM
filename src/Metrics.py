import torch 
import torch.nn.functional as F 
import cv2 
from sklearn.metrics import roc_curve,auc 
import numpy as np 

class mkd_detection:
    def __init__(self,test_loader,model,cfg):
        self.test_loader = test_loader 
        self.model = model 
        self.cfg = cfg 
    def cal_loss(self,output_pred,output_real):
        similarity_loss = torch.nn.CosineSimilarity()
        lamda = self.cfg['lambda']
        if self.model == 'mkd':
            y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
            y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]
        else:
            y_pred_1, y_pred_2, y_pred_3 = output_pred[0],output_pred[1],output_pred[2]
            y_1, y_2, y_3 = output_real[0],output_real[1],output_real[2]
        abs_loss_1 = torch.mean((y_pred_1 - y_1) ** 2, dim=(1, 2, 3))
        loss_1 = 1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1))
        abs_loss_2 = torch.mean((y_pred_2 - y_2) ** 2, dim=(1, 2, 3))
        loss_2 = 1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1))
        abs_loss_3 = torch.mean((y_pred_3 - y_3) ** 2, dim=(1, 2, 3))
        loss_3 = 1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1))
        total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)
        return total_loss 

    def auroc(self,student,teacher):
        student.eval()
        teacher.eval()
        scores = [] 
        total_labels = [] 
        for imgs,labels in self.test_loader:
            imgs = imgs.to('cuda:0')
            output_pred = student(imgs)
            output_real = teacher(imgs)

            score = self.cal_loss(output_pred,output_real)
            scores.extend(score.detach().cpu().numpy())
            total_labels.extend(labels.detach().cpu().numpy())
        fpr,tpr,thr = roc_curve(total_labels,scores)
        auroc = auc(fpr,tpr)
        return auroc 

    
class STPM_detection:
    def __init__(self,test_loader,model,cfg):
        self.test_loader = test_loader
        self.model = model 
        self.cfg = cfg  

    def test_score(self,student,teacher,loader):
        teacher.eval()
        student.eval()
        loss_map = np.zeros((len(loader.dataset), 64, 64))
        i = 0 
        for batch_imgs,_ in loader:
            batch_imgs = batch_imgs.to(self.cfg['device']).type(torch.float32)
            feat_s = student(batch_imgs)
            feat_t = teacher(batch_imgs)
            if self.model =='MKD':
                feat_s = [feat_s[6], feat_s[9], feat_s[12]]
                feat_t = [feat_t[6], feat_t[9], feat_t[12]]
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
        
    def roc(self,labels, scores):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        return roc_auc    
    def test_inference(self,teacher,student,test_loader,true_gt,test_labels):
        scores = self.test_score(teacher,student,test_loader)
        scores = [cv2.resize(i,dsize=(256,256)) for i in scores]
        scores = np.stack(scores)
        pixel_auroc = self.roc(true_gt.flatten(),scores.flatten())
        image_auroc = self.roc(test_labels,scores.max(-1).max(-1))
        return pixel_auroc,image_auroc 