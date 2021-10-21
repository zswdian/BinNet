import torch.nn as nn
import numpy


class BinOp():

    def __init__(self, model):

        count_conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_conv2d += 1

        start_range = 1
        end_range = count_conv2d - 2
        self.bin_range = numpy.linspace(start_range, end_range, end_range-start_range+1)\
            .astype('int').tolist()

        self.num_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []
        self.model = model

        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index += 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_params):
            neg_mean = self.target_modules[index].data.mean(1, keepdim=True).mul(-1)\
                .expand_as(self.target_modules[index].data)
            self.target_modules[index].data.add_(neg_mean)

    def clampConvParams(self):
        for index in range(self.num_params):
            self.target_modules[index].data.clamp_(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_params):
            self.target_modules[index].data = self.target_modules[index].data.sign()

    def restore(self):
        for index in range(self.num_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryWeightGrad(self):
        alpha = []
        for key, value in dict(self.model.named_parameters()):
            if 'alpha' in key:
                alpha.append(value)

        for index in range(self.num_params):
            weight = self.target_modules[index].data
            alpha = self.alpha[index].data
            alpha[weight.lt(-1.0)] = 0
            alpha[weight.gt(1.0)] = 0
            self.target_modules[index].grad.data = alpha.\
                mul(self.target_modules[index].grad.data)