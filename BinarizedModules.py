import torch
import torch.nn as nn


class BinActive(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        return input.sign()

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.le(-1)] = 0
        grad_input[input.ge(1)] = 0
        return grad_input


class BinConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=1,
                 stride=1, padding=1, dropout=0, groups=1, Linear=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.Linear = Linear

        self.dropout_ratio = dropout

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.bn(input)
        x = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x