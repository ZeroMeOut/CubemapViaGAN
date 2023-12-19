import math
import torch
import torch.nn
import torch.nn.functional as F 
from torch.nn import Conv2d
from tensor_type import Tensor4d
# from torch_pconv import PConv2d


# weights_init and PartialConv I got from a repository of github.com/naoto0804. They bascially did what I did here.
# Arcihitecture https://github.com/ayulockin/deepimageinpainting/blob/master/images/model_partial_conv.png

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    return init_fun

class PartialConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class down_layers(torch.nn.Module):
    def __init__(self, type: str, in_channels: int, out_channels: int):
        super().__init__()

        self.relu = torch.nn.ReLU()

        if type == "seven":
            self.pconv = PartialConv(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        elif type == "five":
            self.pconv = PartialConv(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        elif type == "three":
            self.pconv = PartialConv(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        if type != "seven":
            norm = torch.nn.BatchNorm2d(x.size(dim=1))
            x = norm(x)
        x, mask = self.relu(x), self.relu(mask)
        return x, mask

class up_layers(torch.nn.Module):
    def __init__(self, type: str, relu: str, in_channels: int, out_channels: int):
        super().__init__()
        
        self.relu = relu
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        if type == "first":
            self.pconv = PartialConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        elif type == "second":
            self.pconv = PartialConv(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        
    def forward(self, x: Tensor4d, x_mask: Tensor4d, contat_x: Tensor4d, contat_x_mask: Tensor4d):
        x = F.interpolate(x, scale_factor=2)
        x_mask = F.interpolate(x_mask, scale_factor=2)
        x, x_mask = torch.cat([x, contat_x], dim=1), torch.cat([x_mask, contat_x_mask], dim=1)
        x, x_mask = self.pconv(x, x_mask)
        if self.relu == "yes":
            x, x_mask = self.leaky_relu(x), self.leaky_relu(x_mask)
        else:
            pass
        return x, x_mask


class generator(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Down layers
        self.pconv1 = down_layers("seven", in_channels, 64)
        self.pconv2 = down_layers("five", 64, 128)
        self.pconv3 = down_layers("five", 128, 256)
        self.pconv4 = down_layers("five", 256, 512)
        self.pconv5to8 = down_layers("five", 512, 512)

        # Up layers
        self.pconv9to12 = up_layers("first", "yes", 512 + 512, 512)
        self.pconv13 = up_layers("second", "yes", 512 + 256, 256)
        self.pconv14 = up_layers("second", "yes", 256 + 128, 128)
        self.pconv15 = up_layers("second", "yes", 128 + 64, 64)
        self.pconv16 = up_layers("second", "Nu-uh", 64 + out_channels, out_channels)

    def forward(self, x, mask):
        x1, x1_mask = self.pconv1(x, mask)
        x2, x2_mask = self.pconv2(x1, x1_mask)
        x3, x3_mask = self.pconv3(x2, x2_mask)
        x4, x4_mask = self.pconv4(x3, x3_mask)
        x5, x5_mask = self.pconv5to8(x4, x4_mask)
        x6, x6_mask = self.pconv5to8(x5, x5_mask) #torch.Size([1, 512, 8, 8])
        x7, x7_mask = self.pconv5to8(x6, x6_mask) #torch.Size([1, 512, 4, 4])
        x8, x8_mask = self.pconv5to8(x7, x7_mask) #torch.Size([1, 512, 2, 2])

        x9, x9_mask = self.pconv9to12(x8, x8_mask, x7, x7_mask)
        x10, x10_mask = self.pconv9to12(x9, x9_mask, x6, x6_mask)
        x11, x11_mask = self.pconv9to12(x10, x10_mask, x5, x5_mask)
        x12, x12_mask = self.pconv9to12(x11, x11_mask, x4, x4_mask)
        x13, x13_mask = self.pconv13(x12, x12_mask, x3, x3_mask)
        x14, x14_mask = self.pconv14(x13, x13_mask, x2, x2_mask)
        x15, x15_mask = self.pconv15(x14, x14_mask, x1, x1_mask)
        x16, x16_mask = self.pconv16(x15, x15_mask, x, mask)

        return x16, x16_mask
    
def test():
    net = generator()
    x = torch.randn(1, 3, 512, 512)
    x_mask = torch.ones(1, 3, 512, 512)
    y, y_mask = net(x, x_mask)
    print(y.shape, y_mask.shape)

if __name__ == '__main__':
    test()


        
