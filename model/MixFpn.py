import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ResNet的基本Bottleneck类
class Bottleneck(nn.Module):
    expansion = 4  # 通道倍增数

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, 1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channel=3, layers=[2, 2, 2, 2]):
        super(FPN, self).__init__()
        self.inplanes = 64
        # 处理输入的C1模块（C1代表了RestNet的前几个卷积与池化层）
        self.conv1 = nn.Conv2d(in_channel, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # 搭建自下而上的C2，C3，C4，C5
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        # 对C5减少通道数，得到P5
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
        # 3x3卷积融合特征
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        # 横向连接，保证通道数相同
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    # 自上而下的采样模块
    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # 自下而上
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # 自上而下
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # 卷积的融合，平滑处理
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


class CBL(nn.Module):
    def __init__(self, in_channel, out_channel, kernal_size=3, stride=1, padding=1):
        super(CBL, self).__init__()
        self.cblblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernal_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.cblblock(x)


class BoxEmbed(nn.Module):
    def __init__(self, model_in_channel, W, num_require=25, coord_number=4):
        super(BoxEmbed, self).__init__()
        conv_list = []
        self.coord_number = coord_number
        channel = model_in_channel
        for i in range(int(channel / 32) - 1):
            W = W / 2
            if (W > 4):
                kernal_size = 2
                stride = 2
            else:
                kernal_size = 1
                stride = 1
            if (channel > 32):
                channel = int(channel / 2)
                in_channel = 2 * channel
                out_channel = channel
            else:
                in_channel = channel
                out_channel = channel
            conv_list.append(nn.Conv2d(in_channel, out_channel, kernal_size, stride))
            conv_list.append(nn.BatchNorm2d(out_channel))
            conv_list.append(nn.ReLU())
        self.conv_block = nn.Sequential(*conv_list)
        self.embed = nn.Conv1d(512, num_require, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.shape[0], -1, self.coord_number)  # [Batch,512,4]
        x = self.embed(x)
        return x


class MixFpn(nn.Module):
    def __init__(self, in_channel=3, layers=[2, 2, 2, 2], num_classes=2, num_queries=25, need_return_dict=True):
        super(MixFpn, self).__init__()
        self.fpn = FPN(in_channel, layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.hidden = 128
        self.cbl_same = CBL(self.hidden, self.hidden)
        self.cbl_down1 = CBL(2 * self.hidden, self.hidden)
        self.cbl_down = CBL(self.hidden, self.hidden, 2, 2, 0)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * self.hidden, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.num_require = num_queries
        self.need_return_dict = need_return_dict
        self.softmax = nn.Linear(1024, num_queries * (num_classes + 1) * 2)
        self.class_embed = nn.Linear(num_queries * (num_classes + 1) * 2, num_queries * (num_classes + 1))
        self.box_embed1 = BoxEmbed(self.hidden * 2, 32, num_require=num_queries)
        self.box_embed2 = BoxEmbed(self.hidden * 2, 16, num_require=num_queries)
        self.box_embed3 = BoxEmbed(self.hidden * 2, 8, num_require=num_queries)

    def _upsample(self, x, H, W):
        return F.upsample(x, size=(H, W), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return self._upsample(x, H, W) + y

    def feature(self, x):
        p2, p3, p4, p5 = self.fpn(x)
        B, C, H, W = p2.shape

        for i in range(int(C / self.hidden) - 1):
            p2 = self.conv1(p2)
            p3 = self.conv1(p3)
            p4 = self.conv1(p4)
            p5 = self.conv1(p5)

        x1 = torch.cat([self.cbl_same(p2), self._upsample(p5, H, W)], dim=1)
        feature1 = self.cbl_down1(x1)
        x2 = torch.cat([p3, self.cbl_down(feature1)], dim=1)
        feature2 = self.cbl_down1(x2)
        x3 = torch.cat([p4, self.cbl_down(feature2)], dim=1)
        return x1, x2, x3

    def build_results(self, x, y):
        return {
            "pred_logits": x,
            "pred_boxes": y,
        }

    def forward(self, x):
        x = self.feature(x)
        box_coord_1 = self.box_embed1(x[0])
        box_coord_2 = self.box_embed2(x[1])
        box_coord_3 = self.box_embed3(x[2])
        pred_coord = (box_coord_1 * box_coord_2 * box_coord_3).sigmoid()
        x = self.conv(x[-1])
        x = x.view(x.shape[0], -1)
        x = self.softmax(x)
        pred_class = self.class_embed(x)
        pred_class = pred_class.view(pred_class.shape[0], -1,9)
        return self.build_results(pred_class, pred_coord) if (self.need_return_dict) else [pred_class, pred_coord]

if __name__ == '__main__':
    model = MixFpn(in_channel=3, num_classes=8, num_queries = 25)
    image = torch.zeros((2, 3, 128, 128))
    d = model(image)
    print(d['pred_boxes'].shape, d['pred_logits'].shape)
