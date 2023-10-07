import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=5, skip_connet=True):
        super(Generator, self).__init__()
        self.skip_connet = skip_connet

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(3):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        self.encoder = nn.Sequential(*model)
        model = []

        # Upsampling
        out_features = in_features // 2
        for _ in range(3):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  #  nn.Tanh()
                  ]

        self.decoder = nn.Sequential(*model)

    def model(self, x):
        return self.decoder(self.encoder(x))

    def forward(self, x):
        if self.skip_connet:
            return self.model(x) + x
        else:
            return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # 额外添加层
        #  model += [nn.Conv2d(512, 512, 4, stride=2, padding=1),
        #  nn.InstanceNorm2d(512),
        #  nn.LeakyReLU(0.2, inplace=True)]

        #  model += [nn.Conv2d(512, 512, 4, stride=1, padding=1),
        #  nn.InstanceNorm2d(512),
        #  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, num_d=3):
        super().__init__()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.D = Discriminator(input_nc)
        self.num_d = num_d

    def forward(self, x):
        result = []
        for _ in range(self.num_d):
            result.append(self.D(x))
            x = self.downsample(x)
        result = sum(result) / len(result)
        return result


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dim=64, bilinear=True, res=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.res = res

        self.inc = DoubleConv(n_channels, dim)
        self.down1 = Down(dim, dim * 2)
        self.down2 = Down(dim * 2, dim * 4)
        self.down3 = Down(dim * 4, dim * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor)
        self.up1 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4 = Up(dim * 2, dim, bilinear)
        self.outc = OutConv(dim, n_classes)

    def encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        feature = torch.cat((x1.flatten(1), x2.flatten(1), x3.flatten(1), x4.flatten(1), x5.flatten(1)), dim=1)
        return feature

    def model(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        res = self.outc(x)
        return res

    def forward(self, x):
        if self.res:
            return self.model(x) + x
        else:
            return self.model(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TwoHeadAttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, dim=64, bilinear=True, res=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.res = res

        self.inc = DoubleConv(n_channels, dim)
        self.down1 = Down(dim, dim * 2)
        self.down2 = Down(dim * 2, dim * 4)
        self.down3 = Down(dim * 4, dim * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor)
        # decoder1
        self.up1_head1 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2_head1 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3_head1 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4_head1 = Up(dim * 2, dim, bilinear)
        self.outc_head1 = OutConv(dim, n_classes)
        # decoder2
        self.up1_head2 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2_head2 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3_head2 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4_head2 = Up(dim * 2, dim, bilinear)
        self.outc_head2 = OutConv(dim, n_classes*2)

    def backbone(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # head1
        x_head1 = self.up1_head1(x5, x4)
        x_head1 = self.up2_head1(x_head1, x3)
        x_head1 = self.up3_head1(x_head1, x2)
        x_head1 = self.up4_head1(x_head1, x1)
        x_head1 = self.outc_head1(x_head1)
        # head2
        x_head2 = self.up1_head2(x5, x4)
        x_head2 = self.up2_head2(x_head2, x3)
        x_head2 = self.up3_head2(x_head2, x2)
        x_head2 = self.up4_head2(x_head2, x1)
        x_head2 = self.outc_head2(x_head2)

        x_head2 = torch.softmax(x_head2,1)
        bg_mask = x_head2[:,0:1]
        fg_mask = x_head2[:,1:]
        fg = x_head1 * fg_mask
        bg = x * bg_mask
        result = fg + bg
        return result, fg_mask, fg, bg_mask, bg

    def forward(self,x):
        return self.backbone(x)[0]

    def model(self,x):
        return self.backbone(x)[1]


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        num_params = 1
        for size in param.size():
            num_params *= size
        total_params += num_params
    return total_params

if __name__ == "__main__":
    x = torch.randn(3, 1, 512, 512)
    G = UNet(1, 1, 64)
    out = G(x)
    total_params = count_parameters(G)
    print(f"Total number of parameters: {total_params}")
    pass
