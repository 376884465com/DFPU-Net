import torch
import torch.nn as nn  
from nets.resnet import resnet50      
from nets.vgg import VGG16      
from nets.attention import ChannelAttention,cbam_block, eca_block, se_block  

attention_block = [cbam_block, cbam_block, eca_block]   

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
     
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)  
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.ConvTranspose2d(in_channels=in_size-out_size, out_channels=in_size-out_size, kernel_size=4, stride=2, padding=1)  
        self.relu   = nn.ReLU(inplace = True)    
        self.phi = 1
        self.up_concat_att = attention_block[self.phi - 1](in_size - out_size)  

    def forward(self, inputs1, inputs2):
        
        outputs = torch.cat([inputs1, (self.up(inputs2))*(self.up_concat_att(self.up(inputs2)))], 1) 
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg', phi = 0): 
        super(Unet, self).__init__()

        self.phi = phi   

        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)  
            in_filters  = [192, 384, 768, 1024]  
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

     
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])  
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(  
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

        if 1 <= self.phi and self.phi <= 3:
            self.up_concat4_att = attention_block[self.phi - 1](128)  
            self.up_concat3_att = attention_block[self.phi - 1](256)
            self.up_concat2_att = attention_block[self.phi - 1](512)
            self.up_concat1_att = attention_block[self.phi - 1](512)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)  
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)


        up4 = self.up_concat4(feat4, feat5) 
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

      

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
