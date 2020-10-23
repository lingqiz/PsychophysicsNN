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


def generate_gabor(sz, A, omega, theta, func=torch.cos, K=np.pi):
    radius = (int(sz[0] / 2.0), int(sz[1] / 2.0))
    [x, y] = torch.meshgrid([torch.tensor(range(-radius[0], radius[0])),
                             torch.tensor(range(-radius[1], radius[1]))])
    x = x.float()
    y = y.float()
    x1 =  x * torch.cos(theta) + y * torch.sin(theta)
    y1 = -x * torch.sin(theta) + y * torch.cos(theta)

    gauss = omega ** 2 / (4 * np.pi * K ** 2) * torch.exp(- omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    sinusoid = A * func(omega * x1) * torch.exp(torch.tensor(K ** 2 / 2))

    return gauss * sinusoid


def rgb_gabor(theta):
    output = torch.zeros(1, 3, 224, 224)
    gabor  = generate_gabor((224, 224), A=10, omega=torch.tensor(0.3), theta=theta, func=torch.cos)
    for idx in range(3):
        output[0, idx, :, :] = gabor

    return output


def gen_sinusoid(sz, A, omega, rho):
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = torch.meshgrid([torch.tensor(range(-radius[0], radius[0])),
                             torch.tensor(range(-radius[1], radius[1]))])
    x = x.float()
    y = y.float()
    stimuli = A * torch.cos(0.35 * omega[0] * x + 0.35 * omega[1] * y + rho)
    return stimuli


def rgb_sinusoid(theta):
    output = torch.zeros(1, 3, 224, 224)
    sin_stim = gen_sinusoid((224, 224), A=10, omega=[torch.cos(theta), torch.sin(theta)], rho=0)
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output


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

    def cos_similarity(self, theta, layer_idx):
        delta = 0.01
        plus  = self.orientation_response(theta + delta, layer_idx)
        minus = self.orientation_response(theta - delta, layer_idx)
        return nn.modules.distance.CosineSimilarity()(plus, minus).item()


def plot_fisher(layer_idx=0, normalize=True):
    model_urls = {'alexnet': 'http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', }
    nn_model = ConvNet(rgb_sinusoid)
    nn_model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    theta_range = np.linspace(0.0, np.pi, 50)
    theta_image = np.array([nn_model.fisher(theta, 0) for theta in theta_range])
    theta_fisher = np.array([nn_model.fisher(theta, layer_idx) for theta in theta_range])

    if normalize:
        theta_image = theta_image / np.sum(theta_image)
        theta_fisher = theta_fisher / np.sum(theta_fisher)

    plt.plot(theta_range / np.pi * 180, np.sqrt(theta_image), 'o-')
    plt.plot(theta_range / np.pi * 180, np.sqrt(theta_fisher), 'o-')
    plt.show()


def plot_cos_sim(layer_idx=0):
    model_urls = {'alexnet': 'http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', }
    nn_model = ConvNet(rgb_sinusoid)
    nn_model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    theta_range = np.linspace(0.0, np.pi, 100)
    theta_sim = [nn_model.cos_similarity(theta, layer_idx) for theta in theta_range]

    plt.plot(theta_range / np.pi, theta_sim, 'o-')
    plt.show()


class Estimator:
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx

    # return an MLE estimate of theta given response
    def mle(self, response, lb=0, ub=np.pi):
        objective = lambda theta: self.model.orientation_loss(response, theta, self.layer_idx)
        return optimize.fminbound(objective, lb, ub, disp=1)

    # calculate the (full shape of) likelihood function given a response vector
    def log_llhd(self, response, noise_var, lb, ub):
        theta_range = np.linspace(lb, ub, 200)
        llhd = np.zeros(theta_range.size)
        for idx, theta in enumerate(theta_range):
            mean_res = self.model.orientation_response(theta, self.layer_idx).detach().numpy().flatten()
            llhd_idpd = np.array([multivariate_normal.pdf(response[dim], mean_res[dim], noise_var)
                                  for dim in range(response.size)])
            llhd[idx] = np.sum(np.log(llhd_idpd))

        return theta_range, llhd


def plot_likelihood(theta=0, layer_idx=3):
    model_urls = {'alexnet': 'http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', }
    nn_model = ConvNet(rgb_sinusoid)
    nn_model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    estimator = Estimator(nn_model, layer_idx=layer_idx)

    # noisy response
    response = nn_model.orientation_response(theta, layer_idx).detach().numpy().flatten()
    theta_range, llhd = estimator.log_llhd(response, 10, theta - np.pi * 0.25, theta + np.pi * 0.25)

    plt.plot(theta_range / np.pi, llhd)
    plt.show()


def plot_res():
    theta_range = np.linspace(0.0, np.pi, 50)
    var_est = np.load('./sim_data/data_std_iter2000_01.npy')

    plt.plot(theta_range / np.pi, var_est, 'o-')
    plt.title('std of the estimate')
    plt.show()


def main():
    model_urls = {'alexnet': 'http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', }
    nn_model = ConvNet(rgb_sinusoid)
    nn_model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    estimator = Estimator(nn_model, layer_idx=3)

    theta_range = np.linspace(0.0, np.pi, 100)
    mean_est = np.zeros(theta_range.shape)
    std_est  = np.zeros(theta_range.shape)

    for idx, theta in enumerate(theta_range):
        response = nn_model.orientation_response(theta, layer_idx=3)
        noise = torch_normal.Normal(loc=torch.tensor([0.0]),
                                    scale=torch.tensor([1.0]))

        estimates = np.array([estimator.mle(response + noise.sample(response.size()).view(1, -1),
                                            lb=theta-0.25 * np.pi,
                                            ub=theta+0.25 * np.pi) for _ in range(1000)])
        mean_est[idx] = np.mean(estimates)
        std_est[idx]  = np.std(estimates)

    np.save('./data_mean', mean_est)
    np.save('./data_std', std_est)

    plt.plot(theta_range / np.pi, mean_est)
    plt.show()

    plt.plot(theta_range / np.pi, std_est, 'o-')
    plt.show()


if __name__ == '__main__':
    plot_fisher(layer_idx=3, normalize=True)
    # plot_likelihood(theta=np.pi*0.5)