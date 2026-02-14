import torch
import torch.nn as nn
from utils import CNNBlock, TCNNBlock, _create_CNN_block

architecture = [
    (2,0,2,128, "U"),
    [(3,1,1,128, "C"), 2],
    (2,0,2,64, "U"),
    [(3,1,1,64, "C"), 2],
    (2,0,2,32, "U"),
    [(3,1,1,32, "C"), 2],
    (3,1,1,3, "C")
]

class Decoder(nn.Module):
    def __init__(self, in_channels=128, **kwargs):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture
        # self.out_channels = out_channels
        
        self.decoder = self._create_decoder(self.in_channels, self.architecture)

    def forward(self, x):
        return self.decoder(x)

    def _create_decoder(self, in_channels, architecture):
        layers = []

        for archi_x in architecture:
            if type(archi_x) == list:
                blk_cfg, num_repeats = archi_x
                for _  in range(num_repeats):
                    layers.append(_create_CNN_block(
                        in_channels, blk_cfg
                    ))
                    in_channels = blk_cfg[-2]
            elif type(archi_x) == tuple:
                layers.append(_create_CNN_block(
                    in_channels, archi_x
                ))
                in_channels = archi_x[-2]

        return nn.Sequential(*layers)
    
    def test(self, x):
        y_ = self.forward(x)
        print(f"Decoder-Model(test for {x.shape}):",y_.shape)
    
def test(x):
    decoder_model = Decoder()
    y_ = decoder_model(x)
    print("\noutput_head:", y_.shape)


if __name__ == "__main__":
    test(torch.zeros((1,128,256,256)))