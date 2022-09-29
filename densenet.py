import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict


class DenseBlock(nn.ModuleDict):

  def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
    
    super().__init__()
    
    for i in range(num_layers):
      layer = DenseLayer(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
      self.add_module("denselayer%d" % (i + 1), layer)

  def forward(self, init_features):
    features = [init_features]
    for name, layer in self.items():
      new_features = layer(features)
      features.append(new_features)
    return torch.cat(features, 1)


class TransitionLayer(nn.Sequential):

  def __init__(self, num_input_features, num_output_features):
    super().__init__()

    self.norm = nn.BatchNorm2d(num_input_features)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    

class DenseLayer(nn.Module):

  def __init__(self, num_input_features, bn_size, growth_rate, drop_rate):
    
    super().__init__()

    self.norm1 = nn.BatchNorm2d(num_input_features)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

    self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    self.drop_rate = float(drop_rate)

  def bn_function(self, inputs):
    concated_features = torch.cat(inputs, 1)
    bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
    return bottleneck_output

  def forward(self, input):
    prev_features = [input] if isinstance(input, Tensor) else input
    bottleneck_output = self.bn_function(prev_features)
    new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

    if self.drop_rate > 0:
      new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

    return new_features
    

class DenseNet(nn.Module):

  def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=24, bn_size=32, num_classes=6, drop_rate=0):

    super().__init__()

    self.features = nn.Sequential(OrderedDict([
      ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
      ("norm0", nn.BatchNorm2d(num_init_features)),
      ("relu0", nn.ReLU(inplace=True)),
      ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    ]))

    num_features = num_init_features

    for i, num_layers in enumerate(block_config):
      
      block = DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
      self.features.add_module("denseblock%d" % (i + 1), block)
      
      num_features = num_features + num_layers * growth_rate

      if i != len(block_config) - 1:
          trans = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
          self.features.add_module("transition%d" % (i + 1), trans)
          
          num_features = num_features // 2

    self.features.add_module("norm5", nn.BatchNorm2d(num_features))
    self.classifier = nn.Linear(num_features, num_classes)

  def forward(self, x):
    features = self.features(x)
    
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)

    return out
    