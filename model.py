import paddle
import paddle.nn as nn
from paddle.vision.models.resnet import  BasicBlock,BottleneckBlock

# 实现RBF网络
class RBF(nn.Layer):
    def __init__(self, in_features, num_centers, out_features, bias=True):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.center = self.create_parameter(shape=[num_centers, in_features], default_initializer=nn.initializer.Uniform(-1, 1))
        self.weight = self.create_parameter(shape=[out_features, in_features], default_initializer=nn.initializer.Uniform(-1, 1))
        # self.linear = nn.Linear(num_centers, out_features, bias_attr=bias)
        if bias:
            self.bias = self.create_parameter(shape=[out_features], default_initializer=nn.initializer.Uniform(-1, 1))
        else:
            self.bias = None

    def forward(self, input):
        # input: [batch_size, in_features]
        # weight: [out_features, in_features]
        # output: [batch_size, out_features]
        output = paddle.exp(-paddle.sum(paddle.square(input.unsqueeze(1) - self.center)))
        if self.bias is not None:
            output += self.bias
        return output

class MLP(nn.Layer):
    def __init__(self, in_features=5322, out_features=4):
        super(MLP, self).__init__()
        self.linears = nn.Sequential(nn.Linear(in_features, in_features//3),
                                        nn.ReLU(),
                                        nn.Linear(in_features//3, in_features//4),
                                        nn.ReLU(),
                                        nn.Linear(in_features//4, in_features//8),
                                        nn.ReLU(),
                                        # nn.Dropout(0.5),
                                        nn.Linear(in_features//8, in_features//16),
                                        nn.ReLU(),
                                        nn.Linear(in_features//16, out_features))

    def forward(self, x):
        return self.linears(x)



class ResNet(nn.Layer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        Block (BasicBlock|BottleneckBlock): Block module of model.
        depth (int, optional): Layers of ResNet, Default: 50.
        width (int, optional): Base width per convolution group for each convolution block, Default: 64.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.
        groups (int, optional): Number of groups for each convolution block, Default: 1.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet model.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

            # build ResNet with 18 layers
            resnet18 = ResNet(BasicBlock, 18)

            # build ResNet with 50 layers
            resnet50 = ResNet(BottleneckBlock, 50)

            # build Wide ResNet model
            wide_resnet50_2 = ResNet(BottleneckBlock, 50, width=64*2)

            # build ResNeXt model
            resnext50_32x4d = ResNet(BottleneckBlock, 50, width=4, groups=32)

            x = paddle.rand([1, 3, 224, 224])
            out = resnet18(x)

            print(out.shape)
            # [1, 1000]
    """

    def __init__(self,
                 block=BasicBlock,
                 depth=18,
                 width=64,
                 in_features=6,
                 out_features=4,
                 with_pool=True,
                 groups=1):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.groups = groups
        self.base_width = width
        self.out_features = out_features
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(in_features,
                               self.inplanes,
                               kernel_size=3,
                               stride=2,
                               padding=3,
                               bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        if out_features > 0:
            self.out = nn.Linear(512 * block.expansion, out_features)
            # self.out = nn.Sequential(nn.Linear(512 * block.expansion, 512),
            #                          nn.ReLU(),
            #                             nn.Linear(512, out_features))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes * block.expansion,
                          1,
                          stride=stride,
                          bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      norm_layer=norm_layer))

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

        if self.with_pool:
            x = self.avgpool(x)

        if self.out_features > 0:
            x = paddle.flatten(x, 1)
            x = self.out(x)
        return x


def build_model(model_name='resnet18', in_features=6, out_features=220):
    if model_name == 'resnet18':
        model = ResNet(BasicBlock, depth=18, in_features=in_features, out_features=out_features)
    elif model_name == 'resnet50':
        model = ResNet(BottleneckBlock, depth=50, in_features=in_features, out_features=out_features)
    elif model_name == 'mlp':
        model = MLP(in_features, out_features)
    return model