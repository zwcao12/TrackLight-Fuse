import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRGB(nn.Module):
    """Autoencoder Encoder model."""

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.enconv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.enconv5 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.enconv1(x)) #[1,32,224,224]
        x = F.relu(self.enconv2(x)) #[1,64,112,112]
        x = F.relu(self.enconv3(x)) #[1,128,56,56]
        x = F.relu(self.enconv4(x)) #[1,128,56,56]
        x = self.enconv5(x) #[1,256,26,26]
        return x

class EncoderD(nn.Module):
    """Autoencoder Encoder model."""

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.enconv5 = nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.enconv1(x)) #[1,32,224,224]
        x = F.relu(self.enconv2(x)) #[1,32,112,112]
        x = F.relu(self.enconv3(x)) #[1, 64, 56, 56]
        x = F.relu(self.enconv4(x)) #[1,128, 56, 56]
        x = self.enconv5(x) #[1,128,26,26]
        return x

#RGB-D D-RGB
class FeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features=None, out_features=None, act_layer=torch.nn.GELU):
        super().__init__()
        self.act_fcn = act_layer()
        self.input = torch.nn.Linear(in_features, (in_features + out_features) // 2)
        self.projection = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.output = torch.nn.Linear((in_features + out_features) // 2, out_features)

    def forward(self, x):
        x = self.input(x) #[224,224,128] [224,224,256]
        x = self.act_fcn(x)
        x = self.projection(x)
        x = self.act_fcn(x)
        x = self.output(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, in_channels):
        super(MambaBlock, self).__init__()
        self.in_channels = in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.act = nn.SiLU()
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.linear2 = nn.LayerNorm(in_channels, in_channels)
        self.final_linear = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        norm_x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        linear1_out = self.linear1(norm_x.permute(0,2,3,1)).permute(0,3,1,2)
        linear2_out = self.linear2(norm_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        res = self.act(linear2_out)
        conv_out = self.conv(linear1_out)
        conv_out = self.act(conv_out)
        gate_out1 = self.conv1(conv_out)
        gate_out2 = self.conv2(conv_out)
        gate_out = (conv_out + gate_out1 + gate_out2) / 3
        gate_out = gate_out * res
        output = self.final_linear(gate_out.permute(0,2,3,1)).permute(0, 3, 1, 2)
        output = output + x
        return output

class FeatureProjectionConv(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, act_layer=torch.nn.GELU):
        super().__init__()
        self.act_fcn = act_layer()
        self.input_conv = nn.Conv2d(in_channels, (in_channels + out_channels) // 2, kernel_size=1)
        self.projection_conv = nn.Conv2d((in_channels + out_channels) // 2, (in_channels + out_channels) // 2, kernel_size=1)
        self.output_conv = nn.Conv2d((in_channels + out_channels) // 2, out_channels, kernel_size=1)
        self.mamba = MambaBlock(out_channels)

    def forward(self, x):
        #[1,256,224,224]
        x = self.input_conv(x) #[1,192,224,224]
        x = self.act_fcn(x)
        x = self.projection_conv(x)
        x = self.act_fcn(x)
        x = self.output_conv(x) #[1,128,224,224]
        x = self.mamba(x)
        return x

class DecoderRGB(nn.Module):
    """Autoencoder Decoder model.

    Args:
        out_channels (int): number of convolution output channels
        img_size (tuple): size of input images
    """

    def __init__(self, out_channels, padding, img_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.img_size = img_size
        self.last_upsample = (
            int(img_size / 4) if padding else int(img_size - 4),
            int(img_size / 4) if padding else int(img_size - 4),
        )
        self.deconv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding=2)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)

    def forward(self, x):
        #[2,256,26,26]
        x = F.interpolate(x, size=(int(self.img_size / 16) - 1, int(self.img_size / 16) - 1), mode="bilinear") #[2,256,13,13]
        x = F.relu(self.deconv1(x)) #[2,128,15,15]
        x = self.dropout1(x)

        x = F.interpolate(x, size=(int(self.img_size / 8), int(self.img_size / 8)), mode="bilinear") #[2,128,28,28]
        x = F.relu(self.deconv2(x)) #[2,64,30,30]
        x = self.dropout2(x)

        x = F.interpolate(x, size=(int(self.img_size / 4) - 1, int(self.img_size / 4) - 1), mode="bilinear") #[2,64,55,55]
        x = F.relu(self.deconv3(x)) #[2,64,57,57]
        x = self.dropout3(x)

        x = F.interpolate(x, size=(int(self.img_size / 2), int(self.img_size / 2)), mode="bilinear") #[2,64,112,112]
        x = F.relu(self.deconv4(x)) #[2,32,114,114]
        x = self.dropout4(x)

        x = F.interpolate(x, size=self.last_upsample, mode="bilinear") #[2,32,220,220]
        x = F.relu(self.deconv5(x)) #[2,32,222,222]

        x = self.deconv6(x) #[2,3,224,224]
        return x

class DecoderD(nn.Module):
    """Autoencoder Decoder model.

    Args:
        out_channels (int): number of convolution output channels
        img_size (tuple): size of input images
    """

    def __init__(self, out_channels, padding, img_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.img_size = img_size
        self.last_upsample = (
            int(img_size / 4) if padding else int(img_size - 4),
            int(img_size / 4) if padding else int(img_size - 4),
        )
        self.deconv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding=2)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)

    def forward(self, x):
        #[2,128,26,26]
        x = F.interpolate(x, size=(int(self.img_size / 16) - 1, int(self.img_size / 16) - 1), mode="bilinear") #[2,128,13,13]
        x = F.relu(self.deconv1(x)) #[2,128,15,15]
        x = self.dropout1(x)

        x = F.interpolate(x, size=(int(self.img_size / 8), int(self.img_size / 8)), mode="bilinear") #[2,128,28,28]
        x = F.relu(self.deconv2(x)) #[2,64,30,30]
        x = self.dropout2(x)

        x = F.interpolate(x, size=(int(self.img_size / 4) - 1, int(self.img_size / 4) - 1), mode="bilinear") #[2,64,55,55]
        x = F.relu(self.deconv3(x)) #[2,64,57,57]
        x = self.dropout3(x)

        x = F.interpolate(x, size=(int(self.img_size / 2), int(self.img_size / 2)), mode="bilinear") #[2,64,112,112]
        x = F.relu(self.deconv4(x)) #[2,32,114,114]
        x = self.dropout4(x)

        x = F.interpolate(x, size=self.last_upsample, mode="bilinear") #[2,32,220,220]
        x = F.relu(self.deconv5(x)) #[2,32,222,222]

        x = self.deconv6(x) #[2,3,224,224]
        return x