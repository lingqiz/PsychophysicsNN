# scientific computing
import numpy as np
import matplotlib
import scipy.io as sio

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


def plot_fisher(stimulus_gen, layer_idx=0, normalize=True, reference=True, plot=True):
    model_urls = {'alexnet': 'http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', }
    nn_model = ConvNet(stimulus_gen)
    nn_model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    theta_range = np.linspace(0.0, np.pi, 80)
    theta_image = np.array([nn_model.fisher(theta, 0) for theta in theta_range])
    theta_fisher = np.array([nn_model.fisher(theta, layer_idx) for theta in theta_range])

    if normalize:
        theta_image = theta_image / np.sum(theta_image)
        theta_fisher = theta_fisher / np.sum(theta_fisher)

    if reference:
        plt.plot(theta_range / np.pi, np.sqrt(theta_image), 'o-')

    if plot:
        plt.plot(theta_range / np.pi, np.sqrt(theta_fisher), 'o-')
        plt.xlabel('Orientation (Radius)')
        plt.ylabel('Square Root of Fisher Information')
        plt.show()
    return theta_fisher


def plot_fisher_center_surround(surround, layer_idx=0, normalize=True, reference=True, plot=True, freq=0.35):
    stim_gen = lambda theta: rgb_center_surround(theta, surround, freq=freq)
    return plot_fisher(stim_gen, layer_idx, normalize, reference, plot)


def plot_stimulus(freq=0.35):
    theta = torch.tensor(np.pi * 0.25)
    sinwave = gen_sinusoid(224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, freq=freq)

    plt.figure()
    plt.axis('off')
    plt.imshow(sinwave.detach().numpy(), cmap=plt.gray())
    plt.show()


def fisher_frequency(freq_range, layer_idx, plot=True):
    theta_range = np.linspace(0.0, np.pi, 80)
    fisher_info = np.zeros((freq_range.size, theta_range.size))
    for idx in range(freq_range.size):
        stimulus = lambda theta: rgb_sinusoid(theta, freq_range[idx])
        fisher_info[idx, ] = np.sqrt(plot_fisher(stimulus, layer_idx=layer_idx, normalize=True, reference=False, plot=False))

    mean_fisher = fisher_info.mean(axis=0)
    std_fisher  = fisher_info.std(axis=0)

    if plot:
        plt.errorbar(theta_range, mean_fisher, std_fisher)
        plt.xlabel('Orientation (Radius)')
        plt.ylabel('Proportion of (Square Root) Fisher Information')
        plt.show()

    return (mean_fisher, std_fisher)


def fisher_center_surround(freq_range):
    theta_range = np.linspace(0.0, np.pi, 80)
    fisher_info = np.zeros((freq_range.size, theta_range.size))
    for idx in range(freq_range.size):
        stimulus = lambda theta: rgb_sine_noise(theta, freq_range[idx])
        fish_info_control = np.sqrt(plot_fisher(stimulus, layer_idx=4, normalize=False, reference=False, plot=False))
        fish_info_exp = np.sqrt(plot_fisher_center_surround(torch.tensor(np.pi * 0.5), layer_idx=4, normalize=False, reference=False, plot=False, freq=freq_range[idx]))

        fish_info_diff = fish_info_exp - fish_info_control
        fisher_info[idx, ] = fish_info_diff - fish_info_diff.mean(axis=0)


    mean_fisher = fisher_info.mean(axis=0)
    std_fisher = fisher_info.std(axis=0)

    plt.errorbar(theta_range / np.pi * 180, mean_fisher, std_fisher, errorevery = 2)
    plt.xlabel('Orientation (Degree)')
    plt.ylabel('Absolute Difference of Fisher Information')
    plt.show()


# TODO: Decoding for center-surround stimulus, bias
# TODO: Training on center patch, test with changes in surround
# TODO: Color Psychophysics; Threshold & Perceptual Organization;
#  Conditional Statistics (e.g., hue given saturation, hue given brightness)
if __name__ == '__main__':
    # # Fisher information as a function of orientation, combined plot
    # theta_range = np.linspace(0.0, np.pi, 80)
    # mean_fisher1, std_fisher1 = fisher_frequency(np.linspace(0.32, 0.40, 10), 3, False)
    # mean_fisher2, std_fisher2 = fisher_frequency(np.linspace(0.12, 0.20, 10), 6, False)
    #
    # plt.errorbar(theta_range / np.pi * 180, mean_fisher1, std_fisher1)
    # plt.errorbar(theta_range / np.pi * 180, mean_fisher2, std_fisher2)
    # plt.xlabel('Orientation (Degree)')
    # plt.ylabel('Proportion of of Fisher Information')
    # plt.show()
    #
    # sio.savemat('fisher_infomation.mat', {'theta_range': theta_range, 'mean_fisher1' : mean_fisher1, 'mean_fisher2' :  mean_fisher2,
    #                                       'std_fisher1' : std_fisher1, 'std_fisher2' : std_fisher2})
    # # Fisher information as a function of orientation
    # fisher_frequency(np.linspace(0.32, 0.4, 10), 3)
    # fisher_frequency(np.linspace(0.12, 0.20, 10), 6)
    #
    # # High SF, early layer
    # stim_freq = 0.35
    # plot_stimulus(stim_freq)
    # stimulus = lambda theta: rgb_sinusoid(theta, stim_freq)
    # np.sqrt(plot_fisher(stimulus, layer_idx=3, normalize=True, reference=False))
    #
    # # Low SF, late layer
    # stim_freq = 0.15
    # plot_stimulus(stim_freq)
    # stimulus = lambda theta: rgb_sinusoid(theta, stim_freq)
    # np.sqrt(plot_fisher(stimulus, layer_idx=6, normalize=True, reference=False))

    # Center surround stimulus
    theta_range = np.linspace(0.0, np.pi, 80)
    fish_info_control = np.sqrt(plot_fisher(rgb_sine_noise, layer_idx=4, normalize=True, reference=False))
    fish_info_exp = np.sqrt(plot_fisher_center_surround(torch.tensor(np.pi * 0.25), layer_idx=4, normalize=True, reference=False))

    plt.plot(theta_range / np.pi * 180, fish_info_exp - fish_info_control, 'o-')
    plt.xlabel('Orientation (Degree)')
    plt.ylabel('difference in sqrt fisher information')
    plt.show()
    #
    # fisher_center_surround(np.linspace(0.32, 0.42, 10))