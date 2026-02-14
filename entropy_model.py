import torch
import torch.nn as nn
from utils import CNNBlock, TCNNBlock, _create_CNN_block


# the u-net like architecture below will predict the probability distribution for the latent produced by the entropy model
# this probability distribution is used to estimate the bit-rate during training for loss computation and by the arithmetic coder for the actual compression of the quantized image``

# okay, so i need to implement a u-net like architecture that predicts per-pixel (for the latent) probability distributions that can be used to infer the entropy, quantization scale, bitrate, etc
# now, the typical architecture of a u-net involves down-sampling path -> bottleneck -> up-sampling path
# the down-sampling path behaves similar to a typical convolution network that is to compress the abstract information present in an image
# the up-sampling path involves spatial reconstruction of the image through transposed-convolutions that utilize the compressed, abstracted information from the original image
# and the information (feature maps) obtained (read, directly concatenated to the feature maps at the current level) from the corresponding down-sampling layer

# typically, we'd call the DOWN-SAMPLER an ENCODER
# and the UP-SAMPLER a DECODER
# the BOTTLENECK that conncets the ENCODER and DECODER and acts as a bridge between the 2 is the area of highest abstraction of contextual information from the image with lowest spatial information

"""
    LEGEND:
     1. conv (3x3 + ReLU) -> c3
     2. max-pool (2x2) (downsample) -> d
     3. up-conv (2 x 2) (upsample) -> u
     4. copy-concatenate -> cc

     c->c       =cc=>       u->c->c
      d                    u
       c->c     =cc=>     u->c->c
        d                u   
         c->c =cc=> c->c
          d         u
            c->c->c
           
              ⇧
          ```U-NET``` (or something like that)
"""

# nn.ConvTranspose2d()

# tuples: (kernel_size,padding,stride,num_filters, type_of_convolution) -> type_of_convolution: "U" — Transposed Convolution (up-sampling), "C" — Standard Convolution
# lists: (tuple_as_described_above), num_repeat]
architecture = {
    
    # downsampler
    "downsampler": [
        [(3,1,1,128, "C"), 2],
        "M",
        [(3,1,1,256, "C"), 2],
        "M",
        [(3,1,1,512, "C"), 2],
        "M",
    ],

    # bottleneck
    "bottleneck": [
        [(3,1,1,1024, "C"), 3],
    ],

    # upsampler
    "upsampler": [
        (2,0,2,512, "U"),
        [(3,1,1,512, "C"), 2],
        (2,0,2,256, "U"),
        [(3,1,1,256, "C"), 2],
        (2,0,2,128, "U"),
        [(3,1,1,128, "C"), 2],
    ],
}


class UNet(nn.Module):
    def __init__(self, latent_in_channels, in_channels=3, **kwargs):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture
        self.latent_in_channels = latent_in_channels
        self.output_head_archi = (3,1,1,2*self.latent_in_channels, "C")

        # sections of u-net
        # the blocks in encoder and decoder sections would be stored as a module list instead of a stack of modules
        # so that i can run them separately and access the outputs and concatenate them wherever necessary
        
        self.encoder, self.bottleneck_in_chnls = self._create_downsampler_path(self.in_channels, architecture['downsampler'])
        self.bottleneck, self.decoder_in_chnls = self._create_bottleneck(self.bottleneck_in_chnls, architecture['bottleneck'])
        self.decoder, self.output_head_in_chnls = self._create_upsampler_path(self.decoder_in_chnls, architecture['upsampler'])
        self.output_head, self.out_channels = self._create_output_head(self.output_head_in_chnls, self.output_head_archi)

        # concat feature maps
        # self.concat_feature_maps = []

    def forward(self, x):
        # print("encoder:", len(self.encoder))
        x1 = self.encoder[0](x)
        # self.concat_feature_maps.append(x1)
        x1_pooled = self.encoder[1](x1)
        # print(x1.shape, x1_pooled.shape)
        
        x2 = self.encoder[2](x1_pooled)
        # self.concat_feature_maps.append(x2)
        x2_pooled = self.encoder[3](x2)
        # print(x2.shape, x2_pooled.shape)
        
        x3 = self.encoder[4](x2_pooled)
        # self.concat_feature_maps.append(x3)
        x3_pooled = self.encoder[5](x3)
        # # print(x3.shape, x3_pooled.shape)

        # print("\nbottleneck:", len(self.bottleneck))
        x4 = self.bottleneck(x3_pooled)
        # print(x4.shape)

        # print("\ndecoder:", len(self.decoder))
        x5 = self.decoder[0](x4)
        x6 = self.decoder[1](torch.cat((x3,x5), dim=1))
        # print(x5.shape, x6.shape)
        
        x7 = self.decoder[2](x6)
        x8 = self.decoder[3](torch.cat((x2, x7), dim=1))
        # print(x7.shape, x8.shape)

        x9 = self.decoder[4](x8)
        x10 = self.decoder[5](torch.cat((x1,x9), dim=1))
        # print(x9.shape, x10.shape)

        y_ = self.output_head(x10)

        return y_

    def _create_downsampler_path(self, in_channels, architecture):
        layers = nn.ModuleList()

        for archi_x in architecture:
            if type(archi_x) == list:
                blk_cfg, num_repeats = archi_x
                layer = []
                for _  in range(num_repeats):
                    layer.append(_create_CNN_block(
                        in_channels, blk_cfg
                    ))
                    in_channels = blk_cfg[-2]
                layers.append(nn.Sequential(*layer))
            elif archi_x == "M":
                layers.append(nn.MaxPool2d(
                    kernel_size=2, stride=2
                ))

        return layers, in_channels
            
    def _create_upsampler_path(self, in_channels, architecture):
        layers = nn.ModuleList()

        for archi_x in architecture:
            if type(archi_x) == list:
                blk_cfg, num_repeats = archi_x
                layer = []
                for _  in range(num_repeats):
                    layer.append(_create_CNN_block(
                        in_channels, blk_cfg
                    ))
                    in_channels = blk_cfg[-2]
                layers.append(nn.Sequential(*layer))
            elif type(archi_x) == tuple:
                layers.append(_create_CNN_block(
                    in_channels, archi_x
                ))
                in_channels = archi_x[-2]*2

        return layers, in_channels
    
    def _create_bottleneck(self, in_channels, architecture):
        layers = []
        blk_cfg, num_repeats = architecture[-1]
        for _ in range(num_repeats):
            layers.append(_create_CNN_block(
                in_channels, blk_cfg
            ))
            in_channels = blk_cfg[-2]

        return nn.Sequential(*layers), in_channels
    
    def _create_output_head(self, in_channels, architecture):
        return _create_CNN_block(in_channels=in_channels, blk_cfg=architecture), architecture[-2]

    def test(self, x):
        y_ = self.forward(x)
        print(f"Entropy-Model(test for {x.shape}):",y_.shape)


def test(x):
    unet_model = UNet(latent_in_channels=128, in_channels=x.shape[1]).to(device="cuda")
    y_ = unet_model(x)
    print("\noutput_head:", y_.shape)


if __name__ == "__main__":
    test(torch.zeros((1,128,256,256), device="cuda"))