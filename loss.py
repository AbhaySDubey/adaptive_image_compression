import torch
import torch.nn as nn


class AdaptiveCompressorLoss(nn.Module):
    def __init__(self, lmda):
        super(AdaptiveCompressorLoss, self).__init__()
        self.lmda = lmda
        # self.shape = shape
        
    def forward(self, probability, x, x_hat, shape):
        rate = -torch.log2(probability)
        # hardcoded bits per pixel computation; for now
        # shape = 
        R = torch.sum(rate)/(shape[0]*shape[2]*shape[3])
        
        D = nn.functional.mse_loss(x_hat, x)
        
        loss = R + self.lmda*D
        
        return loss,R,D
