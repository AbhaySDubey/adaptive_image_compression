import torch
import torch.nn as nn
from utils import CNNBlock, _create_CNN_block

architecture = [
    [(3,1,1,32, "C"), 2],
    "M",
    [(3,1,1,64, "C"), 2],
    "M",
    [(3,1,1,128, "C"), 2],
    "M",
]

class Encoder(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture
        
        self.encoder = self._create_encoder(self.in_channels, architecture)
        
    def forward(self, x):
        return torch.tanh(self.encoder(x))*10.0
    
    def _create_encoder(self, in_channels, architecture):
        layers = []

        for archi_x in architecture:
            if type(archi_x) == list:
                blk_cfg, num_repeats = archi_x
                for _  in range(num_repeats):
                    layers.append(_create_CNN_block(
                        in_channels, blk_cfg
                    ))
                    in_channels = blk_cfg[-2]
            elif archi_x == "M":
                layers.append(nn.MaxPool2d(
                    kernel_size=2, stride=2
                ))

        return nn.Sequential(*layers)
    
    def test(self, x):
        y_ = self.forward(x)
        print(f"Encoder-Model(test for {x.shape}):",y_.shape)

# def test(x):
#     encoder_model = Encoder()
#     y_ = encoder_model(x)
#     print("\noutput_head:", y_.shape)


# if __name__ == "__main__":
#     test(torch.zeros((1,3,256,256)))