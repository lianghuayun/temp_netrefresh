import torch
import torch.nn as nn

# ----------------------------------------------------
# 定义基本的残差块（BasicBlock），用于 ResNet（残差网络）模型
# ----------------------------------------------------
class BasicBlock(nn.Module):
    # expansion 用于调整卷积层的输出通道数
    expansion = 1  # 没有通道数的扩展

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # 初始化类实例
        # in_channels 输入特征图的通道数
        # out_channels 输出特征图的通道数
        # stride 卷积操作的步幅，默认为 1
        # downsample 用来处理输入和输出的尺寸不匹配时情况（可选参数）
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# ----------------------------------------------------
# 定义ResNet（残差网络）模型
# ----------------------------------------------------
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        # 初始化类实例
        # block 残差块的类型
        # layers 每一层块数的列表
        # num_classes 分类任务的类别数
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, out_channels, blocks, stride=1):
        #  定义残差块的层
        # out_channels 每层输出的通道数
        # layers[i] 每层中的残差块的数量
        # stride 步幅
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # downsample 是一个由 1x1 卷积和批归一化层组成的模块，用于调整输入和输出的形状
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet50(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


# 创建模型
model = resnet50(num_classes=10)  # 修改为你自己的类别数量

# 将模型移动到设备（例如 GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(model)
