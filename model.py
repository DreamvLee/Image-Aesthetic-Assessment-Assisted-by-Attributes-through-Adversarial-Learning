# coding:utf8
from torch import nn
from torchvision import models
from torch import functional as F


# class Linear(nn.Module):
#     def __init__(self,in_features,out_features):
#         nn.Module.__init__(self)
#         self.layer1 = Linear(in_features,512)
#         self.layer2 = Linear(512,128)
#         self.layer3 = Linear(128,out_features)
#     def forward(self, x):
#         re
#         x = F.Relu(self.layer1(x))
#
#         x2 = self.layer2(x)


class NetmyG(nn.Module):
    """
    1024维生成器定义
    """

    def __init__(self, opt):
        super(NetmyG, self).__init__()
        ngf = opt.ngf  # 生成器feature map数
        self.mylinear = nn.Sequential(
            nn.Linear(12, 1024),

           # nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.Linear(1024, 256 * 7 * 7),
            #nn.BatchNorm2d(256 * 7 * 7),
            nn.ReLU(True),
        )

        self.main = nn.Sequential(
            # 输入是一个nz维度的噪声，我们可以认为它是一个7*7*256的feature map


            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*8) x 14 x 14

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 上一步的输出形状： 64 x 32 x 32

            nn.ConvTranspose2d(ngf * 4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # 上一步的输出形状：32 x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上一步的输出形状：16 x 112 x 112
            nn.ConvTranspose2d(ngf, 3, 4, 2 ,1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x224 x 224
        )

    def forward(self, input):
        res = self.mylinear(input).view(input.size(0),256, 7,7)
        return self.main(res)


class NetG(nn.Module):
    """
    生成器定义
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = opt.ngf  # 生成器feature map数

        # 将49转换为7*7
        self.myliear = nn.Sequential(nn.Linear(12, 49))
        # changed = myliear(input).view(64, 1, 7, 7)
        self.main = nn.Sequential(

            nn.ConvTranspose2d(opt.nz, ngf * 2, 4, stride=2, padding=1, output_padding=0),

            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf / 2, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(ngf / 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf / 2, ngf / 4, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(ngf / 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf / 4, 3, 4, stride=2, padding=1, output_padding=0),
            nn.Tanh()
        )

        # self.main = nn.Sequential(
        #     # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map
        #     nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(True),
        #     # 上一步的输出形状：(ngf*8) x 4 x 4
        #
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # 上一步的输出形状： (ngf*4) x 8 x 8
        #
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # 上一步的输出形状： (ngf*2) x 16 x 16
        #
        #     nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # 上一步的输出形状：(ngf) x 32 x 32
        #     nn.S
        #     nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
        #     nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
        #     # 输出形状：3 x 96 x 96

    def forward(self, input):
        input = input.view(input.size(0), -1)
        changed = self.myliear(input)
        # Batchsize todo
        changed = changed.view(input.size(0), 1, 7, 7)
        return self.main(changed)


class NetD(nn.Module):
    """
    判别器定义
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        weidu = opt.weidu
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Linear(2048, 512)
        self.main = nn.Sequential(
            resnet,

            # nn.Linear(224, 512),
            # nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 12),
            nn.Sigmoid()

        )

        # self.main = nn.Sequential(
        #     # 输入 3 x 224 x 224
        #     nn.Conv2d(3, ndf, 4, stride= 2, padding=1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # 输出 (ndf) x 112 x 112
        #
        #     nn.Conv2d(ndf, ndf * 2, 4,stride= 2, padding=1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # 输出 (ndf*2) x 56 x 56
        #
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2,padding= 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # 输出 (ndf*4) x 28 x 28
        #
        #     nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # 输出 (ndf*8) x 14 x 14
        #
        #     nn.Conv2d(ndf * 8, ndf * 4, 4,stride= 2, padding=1, bias=False),
        #     # 输出 1x 7x 7
        #     nn.Conv2d(ndf * 4, 12, 7,stride= 2, padding=0, bias=False),
        #
        #     nn.Sigmoid()  # 输出一个数(概率)
        # )

    def forward(self, input):
        return self.main(input).view(-1, 12)
