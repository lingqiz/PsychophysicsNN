# scientific computing
import numpy as np
from scipy import optimize
from scipy.stats import multivariate_normal

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# torch package
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.distributions.normal as torch_normal
from torchvision import models

'''
Berardino, A., Laparra, V., Ballé, J., & Simoncelli, E. (2017)
    - Fisher Information Matrix, Distance Metric

Gal, Y., & Ghahramani, Z. (2016)
    - Deep Gaussian Process
    - Representation of aleatoric uncertainty/epistemic uncertainty
    - Dropout/"Noise" as approximate inference

Sampling/Inference when noise is presented/with efficient population

* plot the likelihood function based on gaussian internal noise
* p(r|f(theta)): "mapping" to homogeneous internal space; efficient coding (NN2015)
'''

class ConvNet(models.AlexNet):
    def __init__(self, stim_generator):
        # get layer from features = nn.Sequential()
        super(ConvNet, self).__init__()
        self.generator = stim_generator

    def feature_layer(self, x):
        # access response of the feature layer
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        return x

    # loop through layers in features = nn.Sequential()
    def feedforward(self, x, n_layer):
        for idx, layer in enumerate(self.features.children()):
            if idx >= n_layer:
                return x.view(x.size()[0], -1)
            x = layer(x)

    def orientation_response(self, theta, layer_idx):
        stimulus = self.generator(torch.tensor(theta))
        return self.feedforward(stimulus, layer_idx)

    def orientation_loss(self, response, theta, layer_idx):
        return nn.MSELoss()(response, self.orientation_response(theta, layer_idx)).item()

    # vector-valued derivative: df(layer_idx, theta)/d(theta)
    # finite difference implementation
    def derivative(self, theta, layer_idx):
        delta = 0.01
        f_delta = self.orientation_response(theta + delta, layer_idx) - \
                  self.orientation_response(theta - delta, layer_idx)
        df_dtheta = f_delta / (2 * delta)
        return df_dtheta

    # compute fisher information under Gaussian noise assumption
    def fisher(self, theta, layer_idx):
        df_dtheta = self.derivative(theta, layer_idx)
        return torch.mm(df_dtheta, df_dtheta.transpose(0, 1)).item()