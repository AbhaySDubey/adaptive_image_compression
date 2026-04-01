import torch
import torch.nn as nn
import torch.nn.functional as F


# class AdaptiveCompressorLoss(nn.Module):
#     def __init__(self, lmda):
#         super(AdaptiveCompressorLoss, self).__init__()
#         self.lmda = lmda
#         # self.shape = shape
        
#     def forward(self, probability, x, x_hat, shape):
#         rate = -torch.log2(probability)
#         # hardcoded bits per pixel computation; for now
#         # shape = 
#         R = torch.sum(rate)/(shape[0]*shape[2]*shape[3])
        
#         D = nn.functional.mse_loss(x_hat, x)
        
#         loss = R + self.lmda*D
        
#         return loss,R,D

class AdaptiveCompressorLoss(nn.Module):
    def __init__(self, lmda, alpha):
        super().__init__()
        self.lmda = lmda
        self.alpha = alpha
    
    def reinitialize_la(self, LMDA, ALPHA):
        print(f"changing lambda from {self.lmda} to {LMDA}")
        print(f"changing alpha from {self.alpha} to {ALPHA}")
        self.lmda = LMDA
        self.alpha = ALPHA
        
    def forward(self, probability, x, x_hat, shape):
        # Rate: average bits per symbol
        # shape = x.shape
        R = ((-torch.log2(probability)).sum()/(shape[0]*shape[2]*shape[3]))
        scaled_r = R*self.alpha

        # Distortion
        x_hat = torch.clamp(x_hat, 0.0, 1.0)
        x = torch.clamp(x, 0.0, 1.0)
        
        # for the first train: 80% F, 10% Cr, 10% Cb
        # for the second train: 60% F, 20% Cr, 20% Cb
        # for the third train: 40% F, 30% Cr, 30% Cb
        D = (0.6*F.mse_loss(x_hat[0], x[0], reduction="mean")+0.2*F.mse_loss(x_hat[1], x[1], reduction="mean")+0.2*F.mse_loss(x_hat[2], x[2], reduction="mean"))
        
        scaled_d = D*self.lmda

        loss = scaled_r + scaled_d
        
        return loss,R,scaled_r,D,scaled_d