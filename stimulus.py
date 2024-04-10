import torch
from torch.distributions import normal

STIM_SIZE = 224

def gen_sinusoid(sz, A, omega, rho, freq=0.35):
    radius = int(sz / 2.0)
    [x, y] = torch.meshgrid([torch.tensor(range(-radius, radius)),
                             torch.tensor(range(-radius, radius))],
                             indexing='ij')
    x = x.float()
    y = y.float()
    stimuli = A * torch.cos(freq * omega[0] * x + freq * omega[1] * y + rho)
    return stimuli

def gen_sinusoid_aperture(ratio, sz, A, omega, rho, polarity, freq=0.35):
    sin_stimuli = gen_sinusoid(sz, A, omega, rho, freq=freq)
    radius = int(sz / 2.0)
    [x, y] = torch.meshgrid([torch.tensor(range(-radius, radius)),
                             torch.tensor(range(-radius, radius))],
                             indexing='ij')

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
                             torch.tensor(range(-radius, radius))],
                             indexing='ij')

    aperture = torch.empty(sin_aperture.size(), dtype=torch.float)
    aperture_radius = float(radius) * ratio
    aperture[x ** 2 + y ** 2 >= aperture_radius ** 2] = 1
    aperture[x ** 2 + y ** 2 < aperture_radius ** 2] = 0

    return noise_patch * aperture + sin_aperture

def rgb_sinusoid(theta, freq=0.35):
    output = torch.zeros(1, 3, STIM_SIZE, STIM_SIZE)
    sin_stim = gen_sinusoid(STIM_SIZE, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, freq=freq)
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output

def rgb_sine_aperture(theta):
    output = torch.zeros(1, 3, STIM_SIZE, STIM_SIZE)
    sin_stim = gen_sinusoid_aperture(0.75, STIM_SIZE, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, polarity=1)
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output

def rgb_sine_noise(theta, rho=0, freq=0.35):
    output = torch.zeros(1, 3, STIM_SIZE, STIM_SIZE)
    sin_stim = sinsoid_noise(ratio=0.75, sz=STIM_SIZE, A=1,
                             omega=[torch.cos(theta), torch.sin(theta)],
                             rho=rho, freq=freq)

    # fill in the RGB channels
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output

def rgb_center_surround(theta_center, theta_surround, rho=0, freq=0.35):
    output = torch.zeros(1, 3, STIM_SIZE, STIM_SIZE)
    stimulus = center_surround(ratio=0.60, sz=STIM_SIZE,
                               theta_center=theta_center,
                               theta_surround=theta_surround,
                               A=1, rho=rho, freq=freq)

    # fill in the RGB channels
    for idx in range(3):
        output[0, idx, :, :] = stimulus
    return output
