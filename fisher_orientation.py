# scientific computing
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# torch package
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models

from stimulus import *

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

    # TODO: change to stimulus generator structure
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


def plot_fisher(stimulus_gen, layer_idx=0, normalize=True):
    model_urls = {'alexnet': 'http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', }
    nn_model = ConvNet(stimulus_gen)
    nn_model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    theta_range = np.linspace(0.0, np.pi, 50)
    theta_image = np.array([nn_model.fisher(theta, 0) for theta in theta_range])
    theta_fisher = np.array([nn_model.fisher(theta, layer_idx) for theta in theta_range])

    if normalize:
        theta_image = theta_image / np.sum(theta_image)
        theta_fisher = theta_fisher / np.sum(theta_fisher)

    plt.plot(theta_range / np.pi, np.sqrt(theta_image), 'o-')
    plt.plot(theta_range / np.pi, np.sqrt(theta_fisher), 'o-')
    plt.xlabel('theta (radius)')
    plt.ylabel('sqrt Fisher Information')
    plt.show()
    return theta_fisher


def plot_fisher_center_surround(surround, layer_idx=0, normalize=True):
    stim_gen = lambda theta: rgb_center_surround(theta, surround)
    return plot_fisher(stim_gen, layer_idx, normalize)


def plot_stimulus():
    theta = torch.tensor(np.pi * 0)
    sinwave = gen_sinusoid(224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0)

    plt.figure()
    plt.axis('off')
    plt.imshow(sinwave.detach().numpy(), cmap=plt.gray())
    plt.show()


if __name__ == '__main__':
    # plot_stimulus()
    # fish_info_control = np.sqrt(plot_fisher(rgb_sinusoid, layer_idx=5, normalize=False))

    theta_range = np.linspace(0.0, np.pi, 50)
    fish_info_control = np.sqrt(plot_fisher(rgb_sine_noise, layer_idx=4, normalize=False))
    fish_info_exp = np.sqrt(plot_fisher_center_surround(torch.tensor(np.pi * 0.75), layer_idx=4, normalize=False))

    plt.plot(theta_range / np.pi, fish_info_exp - fish_info_control, 'o-')
    plt.xlabel('theta (radius)')
    plt.ylabel('difference in sqrt fisher information')
    plt.show()
