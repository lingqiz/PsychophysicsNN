# torch package
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

def derivative(model, theta):
    '''
    Vector-valued derivative: df(theta)/d(theta)
    with a finite difference implementation
    '''
    delta = 0.01
    with torch.no_grad():
        f_delta = model.orientation_response(theta + delta) - \
                  model.orientation_response(theta - delta)
    df_dtheta = f_delta / (2 * delta)
    return df_dtheta

# compute fisher information under Gaussian noise assumption
def fisher(model, theta):
    df_dtheta = derivative(model, theta)
    return torch.mm(df_dtheta, df_dtheta.transpose(0, 1)).item()

def compute_fisher(model, normalize=True):
    theta_range = np.linspace(0.0, np.pi, 49)
    theta_fisher = np.array([fisher(model, theta) for theta in theta_range])
    theta_fisher = np.sqrt(theta_fisher)

    if normalize:
        # normalize FI within [0, 2 * pi]
        theta_fisher = theta_fisher / np.trapz(theta_fisher, 2 * theta_range)
        theta_range = theta_range / np.pi * 180 - 90 # convert to degrees

    return theta_range, theta_fisher

class AlexNet(models.AlexNet):
    def __init__(self, stim_generator):
        # get layer from features = nn.Sequential()
        super(AlexNet, self).__init__()
        self.generator = stim_generator
        self.layer_idx = -1

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

    def orientation_response(self, theta):
        '''
        Compute response to an orientation stimulus
        '''
        stimulus = self.generator(torch.tensor(theta))
        # compute the response to the orientation stimulus
        if self.layer_idx == -1:
            return self.feature_layer(stimulus)

        else:
            return self.feedforward(stimulus, self.layer_idx)

class VGG16(models.vgg.VGG):
    def __init__(self, stim_generator):
        # init and load pre-trained VGG16 model
        super(VGG16, self).__init__(models.vgg.make_layers(models.vgg.cfgs['D']))
        self.generator = stim_generator

    def feature_layer(self, x):
        # access response of the feature layer
        x = self.features(x)
        x = x.view(-1, 512 * 7 * 7)
        return x

    def orientation_response(self, theta):
        '''
        Compute response to an orientation stimulus
        '''
        stimulus = self.generator(torch.tensor(theta))
        return self.feature_layer(stimulus)

class ViT(nn.Module):
    def __init__(self, vit_model, stim_generator):
        super(ViT, self).__init__()
        self.vit = vit_model
        self.generator = stim_generator

    def feature_layer(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)
        return x[:, 0]

    def orientation_response(self, theta):
        '''
        Compute response to an orientation stimulus
        '''
        stimulus = self.generator(torch.tensor(theta))
        return self.feature_layer(stimulus)