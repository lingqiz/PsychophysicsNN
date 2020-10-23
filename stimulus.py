import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# torch package
import torch
from torch.distributions import normal


def gen_sinusoid(sz, A, omega, rho, freq=0.35):
    radius = int(sz / 2.0)
    [x, y] = torch.meshgrid([torch.tensor(range(-radius, radius)),
                             torch.tensor(range(-radius, radius))])
    x = x.float()
    y = y.float()
    stimuli = A * torch.cos(freq * omega[0] * x + freq * omega[1] * y + rho)
    return stimuli


def gen_sinusoid_aperture(ratio, sz, A, omega, rho, polarity, freq=0.35):
    sin_stimuli = gen_sinusoid(sz, A, omega, rho, freq=freq)
    radius = int(sz / 2.0)
    [x, y] = torch.meshgrid([torch.tensor(range(-radius, radius)),
                             torch.tensor(range(-radius, radius))])
    aperture = torch.empty(sin_stimuli.size(), dtype=torch.float)

    aperture_radius = float(radius) * ratio
    aperture[x ** 2 + y ** 2 >= aperture_radius ** 2] = 1 - polarity
    aperture[x ** 2 + y ** 2 < aperture_radius ** 2] = polarity

    return sin_stimuli * aperture


def center_surround(ratio, sz, theta_center, theta_surround, A, rho, freq=0.35):
    center = gen_sinusoid_aperture(ratio, sz, A, [torch.cos(theta_center), torch.sin(theta_center)], rho, 1, freq=freq)
    surround = gen_sinusoid_aperture(ratio, sz, A, [torch.cos(theta_surround), torch.sin(theta_surround)], rho, 0, freq=freq)
    return center + surround


def sinsoid_noise(ratio, sz, A, omega, rho, freq=0.35):
    radius = int(sz / 2.0)
    sin_aperture = gen_sinusoid_aperture(ratio, sz, A, omega, rho, 1, freq=freq)

    nrm_dist = normal.Normal(0.0, 0.12)
    noise_patch = nrm_dist.sample(sin_aperture.size())

    [x, y] = torch.meshgrid([torch.tensor(range(-radius, radius)),
                             torch.tensor(range(-radius, radius))])
    aperture = torch.empty(sin_aperture.size(), dtype=torch.float)

    aperture_radius = float(radius) * ratio
    aperture[x ** 2 + y ** 2 >= aperture_radius ** 2] = 1
    aperture[x ** 2 + y ** 2 < aperture_radius ** 2] = 0

    return noise_patch * aperture + sin_aperture


def rgb_sinusoid(theta, freq=0.35):
    output = torch.zeros(1, 3, 224, 224)
    sin_stim = gen_sinusoid(224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, freq=freq)
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output


def rgb_sine_aperture(theta):
    output = torch.zeros(1, 3, 224, 224)
    sin_stim = gen_sinusoid_aperture(0.75, 224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, polarity=1)
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output


def rgb_sine_noise(theta, freq=0.35):
    output = torch.zeros(1, 3, 224, 224)
    sin_stim = sinsoid_noise(0.75, 224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, freq=freq)
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output


def rgb_center_surround(theta_center, theta_surround, freq=0.35):
    output = torch.zeros(1, 3, 224, 224)
    stimulus = center_surround(0.75, 224, theta_center, theta_surround, A=1, rho=0, freq=freq)
    for idx in range(3):
        output[0, idx, :, :] = stimulus
    return output


def show_stimulus(I):
    plt.figure()
    plt.axis('off')
    plt.imshow(I.detach().numpy(), cmap=plt.gray())
    plt.show()


if __name__ == '__main__':
    theta = torch.tensor(np.pi * 0.65)
    sinewave = gen_sinusoid(224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, freq=0.15)
    aperture = gen_sinusoid_aperture(0.75, 224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, polarity=1)
    sinewave_noise = sinsoid_noise(0.75, 224, 1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, freq=0.15)

    show_stimulus(sinewave)
    show_stimulus(aperture)
    show_stimulus(sinewave_noise)
    show_stimulus(center_surround(0.75, 224, torch.tensor(np.pi * 0.45), torch.tensor(np.pi * 0.25), 1, rho=0))
