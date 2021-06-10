# 这个是python当中让print都以python3的形式进行print,即把print视为函数
from __future__ import print_function
# 使得我们能够手动输入命令行参数
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Type, Any, Callable, Union, List, Optional
from torch.autograd import Variable
from torch import Tensor
from typing import Any, List, Tuple
# 可选择的densenet模型
__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']
# 可下载的densenet预训练权重


# 定义一个denseblock(dense layer),其中growth_rate的意思是一层产生多少个特征图
from data_processor.core.load_data_processor import ICDARDataset


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        # 首先对输入做一次bn、激活、卷积
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        # 输出特征图的数量为bn_size*growth_rate，卷积、bn、激活
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *growth_rate, kernel_size=1, stride=1,bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,kernel_size=3, stride=1, padding=1,bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # 判断当前tensor是否参与梯度传播
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    # @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    # @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)   # out_channel:  bn_size * growth_rate

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))  # out_channel:  growth_rate
        # 加上dropout
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        # 随着layer层数的增加，每增加一层，输入的特征图就增加一倍growth_rate
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            # 添加一层layer
            self.add_module('denselayer%d' % (i + 1), layer)


    def forward(self, init_features: Tensor) -> Tensor:
        # 提取特征
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)        # out_channel: growth_rate
            features.append(new_features)
        # 将特征图concat在一起
        return torch.cat(features, 1)    # 通道数: init_features + num_layers * growth_rate


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        # transition层使用的是1 x 1卷积核，作用是用来改变通道数
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=1, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

        Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
            num_classes (int) - number of classification classes
            memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
              but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
        """
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 6,
        memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()

        # 第一层卷积
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # 构建每一个denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # 最后一个bn层
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # 分类器
        self.classifier = nn.Linear(num_features, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # 提取特征、激活、池化、摊平、分类
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# 加载训练权重
def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model

# 预训练权重，其中第一个参数'densenet121'代表densenet的模型名称，32代表每一层添加32个特征图，(6, 12, 24, 16)表示4个denselayer重复的次数，64表示初始特征数
def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet161(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)










parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10')
parser.add_argument('--lr', type=float, default=0.007, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# 跑多少次batch进行一次日志记录
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

# 这个是使用argparse模块时的必备行,将参数进行关联
args = parser.parse_args()
# 这个是在确认是否使用GPU的参数
args.cuda =torch.cuda.is_available()
# 设置一个随机数种子
torch.manual_seed(args.seed)
if args.cuda:
    # 为GPU设置一个随机数种子
    torch.cuda.manual_seed(args.seed)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307, ), (0.3081, ))
# ])
model = densenet121()
# 判断是否调用GPU模式
if args.cuda:
    model.cuda()
# 初始化优化器 model.train()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.9)
# train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(ICDARDataset("../data"), batch_size=args.batch_size, shuffle=True)
# test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(ICDARDataset("../data_test"), batch_size=args.test_batch_size, shuffle=False)
def train(epoch):
    """
    定义每个epoch的训练细节
    """
    # 设置为training模式
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 如果要调用GPU模式,就把数据转存到GPU
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # 优化器梯度初始化为零
        optimizer.zero_grad()
        output = model(data)
        # 负对数似然函数损失
        target = F.one_hot(target, num_classes=6).to(torch.float32)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))


def test():
    with torch.no_grad():
        # 设置为test模式
        torch.cuda.empty_cache()
        model.eval()
        # 初始化测试损失值为0
        test_loss = 0
        # 初始化预测正确的数据个数为0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            target = F.one_hot(target, num_classes=6).to(torch.float32)
            # 把所有loss值进行累加
            test_loss += F.mse_loss(output, target)
            # 获取最大对数概率值的索引
            pred = output.data.max(1, keepdim=True)[1]
            # 对预测正确的个数进行累加
            correct += pred.eq(target.data.max(1, keepdim=True)[1]).sum()

        # 因为把所有loss值进行累加,所以最后要除以总的数据长度才能得到平均loss
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
        ))


# 进行每个epoch的训练
for epoch in range(1, args.epochs + 1):
    train(epoch)

    test()