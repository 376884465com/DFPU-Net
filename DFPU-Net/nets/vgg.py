import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
import torchvision

class VGG(nn.Module): 
    def __init__(self, features, num_classes=1000):  
        super(VGG, self).__init__()
        self.features = features    
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))   
        self.classifier = nn.Sequential(     
            nn.Linear(512 * 7 * 7, 4096),    
            
            nn.ReLU(True),   
            nn.Dropout(),    
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights() 
        aspp_channel = [64, 128, 256, 512]
        self.aspp1 = ASPP(64, 64)
        self.aspp2 = ASPP(128, 128)
        self.aspp3 = ASPP(256, 256)
        self.aspp4 = ASPP(512, 512)
        self.aspp5 = PPM(512, 512 // 4, [2, 3, 6, 9])

    def forward(self, x):  
        feat1 = self.features[  :4 ](x) 
        feat1 = self.aspp1(feat1)
        # feat1 = torchvision.transforms.ToTensor(feat1)
        feat2 = self.features[4 :9 ](feat1)
        feat2 = self.aspp2(feat2)
        feat3 = self.features[9 :16](feat2)
        feat3 = self.aspp3(feat3)
        feat4 = self.features[16:23](feat3)
        feat4 = self.aspp4(feat4)
        feat5 = self.features[23:-1](feat4)
        feat5 = self.aspp5(feat5)

        return [feat1, feat2, feat3, feat4, feat5]  

    def _initialize_weights(self):
        for m in self.modules():    
            if isinstance(m, nn.Conv2d):     
               
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')   
                if m.bias is not None:    
                    nn.init.constant_(m.bias, 0)   
            elif isinstance(m, nn.BatchNorm2d):   
                nn.init.constant_(m.weight, 1)   
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):       
                nn.init.normal_(m.weight, 0, 0.01)   
                nn.init.constant_(m.bias, 0)

import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        self.max  = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.up = nn.ConvTranspose2d(in_channel, depth, kernel_size=4, stride=2, padding=1)  
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2) 
        self.covself = nn.Conv2d(in_channel, depth, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
        # ========================================================== #
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=5, dilation=5)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=7, dilation=7)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
       

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True) 

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
       
        return net

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
       
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                
                nn.AdaptiveAvgPool2d(bin),
                
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))

        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = []  
        for f in self.features:
            temp = f(x)  
            temp = F.interpolate(temp, x_size[2:], mode='bilinear', align_corners=True)
            out.append(temp)
        return torch.cat(out, 1)  


def make_layers(cfg, batch_norm=False, in_channels = 6):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  
        else:
            '''
            在这个案例中，可以沿着图像边缘再填充一层像素
            那么6×6的图像就被填充成了一个8×8的图像
            如果用3×3的图像对这个8×8的图像卷积，得到的输出是6×6的图像
            就得到了一个尺寸和原始图像相同的图像
            '''
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)    
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)   
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}   



def VGG16(pretrained, in_channels = 6, **kwargs):     
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

    del model.avgpool
    del model.classifier
    return model