import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class MKDLoss(nn.Module):
    def __init__(self,model,lamda=0.01):
        super(MKDLoss, self).__init__()
        self.lamda = lamda
        self.criterion = nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()
        self.model = model 

    def feature(self,output_pred,output_real):
        if self.model == 'MKD':
            y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[3], output_pred[6], output_pred[9], output_pred[12]
            y_0, y_1, y_2, y_3 = output_real[3], output_real[6], output_real[9], output_real[12]
            return [y_pred_0, y_pred_1, y_pred_2, y_pred_3],[y_0, y_1, y_2, y_3]
        else:
            return [i for i in output_pred],[j for j in output_real]


    def forward(self, output_pred, output_real):
        feature_s,feature_t = self.feature(output_pred,output_real)
        
        total_abs_loss = 0 
        total_sim_loss = 0
        for y_p,y_r in zip(feature_s,feature_t):
            abs_loss = self.criterion(y_p,y_r)
            sim_loss = torch.mean(1 - self.similarity_loss(y_p.view(y_p.shape[0], -1), y_r.view(y_r.shape[0], -1)))

            total_sim_loss += sim_loss 
            total_abs_loss += abs_loss 
        total_loss = total_sim_loss + self.lamda*total_abs_loss

        return total_loss

class STPMLoss(nn.Module):
    def __init__(self,model):
        super(STPMLoss, self).__init__()
        self.model = model 

    def loss_fn(self,f_s,f_t):
        total_loss = 0 
        for t,s in zip(f_t,f_s):
            t,s = F.normalize(t,dim=1),F.normalize(s,dim=1)
            total_loss += torch.sum((t.type(torch.float32) - s.type(torch.float32)) ** 2, 1).mean()
        return total_loss

    def feature(self,f_s,f_t):
        if self.model == 'MKD':
            y_pred_0, y_pred_1, y_pred_2, y_pred_3 = f_s[3], f_s[6], f_s[9], f_s[12]
            y_0, y_1, y_2, y_3 = f_t[3], f_t[6], f_t[9], f_t[12]
            return [y_pred_0, y_pred_1, y_pred_2, y_pred_3],[y_0, y_1, y_2, y_3]
        else:
            return [i for i in f_s],[j for j in f_t]

    def forward(self, f_s,f_t):
        
        f_s,f_t = self.feature(f_s,f_t)
        total_loss = self.loss_fn(f_s,f_t)
        return total_loss