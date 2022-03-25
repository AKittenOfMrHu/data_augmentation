import numpy as np
import torch
import scipy.ndimage
from skimage.morphology import dilation
from skimage.measure import label, regionprops


def random_gamma(img, gamma_range=(0.5, 1.08)):
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    img = np.power(img, gamma)
    img = np.clip(img, 0., 255.)
    return img   

def random_contrast(img, value_range=(0, 2)):
    value = np.random.uniform(value_range[0], value_range[1])
    mean = np.mean(img)#, axis=(1, 2, 3)) if dim == 4 else np.mean(img, axis=(1, 2))
    img = (img - mean) * value + mean
    img = np.clip(img, 0, 255)
    return img

def random_bright(img, value_range=(-64, 64)):
    value = np.random.uniform(value_range[0], value_range[1])
    img = np.clip(img+value, 0., 255.)
    return img

def gaussian_noise(data, variance_range=(0, 32)):
    variance = np.random.uniform(variance_range[0], variance_range[1])
    new_data = data + np.random.normal(0, scale=variance, size=data.shape)
    new_data = np.clip(new_data, 0., 255.)
    return new_data

def gaussian_blur(data, sigma_range=(0.5, 1.5)):
    sigma = [np.random.uniform(np.min(sigma_range), np.max(sigma_range))
             if np.min(sigma_range) < np.max(sigma_range)
             else np.max(sigma_range) for _ in data.shape]
    data = scipy.ndimage.gaussian_filter(data, sigma=sigma, order=0)
    data = np.clip(data, 0., 255.)
    return data

def horizon_mirroring(imgs, lbls=None):
    imgs = imgs[:, :, ::-1]
    if lbls is not None:
        if len(lbls.shape) == len(imgs.shape):
            lbls = lbls[:, :, ::-1]
        else:
            lbls = lbls[:, ::-1]
    return imgs, lbls


def horizon_mirroring_with_var(imgs, lbls, vars):
    imgs = imgs[:, :, ::-1]
    vars = vars[:, :, ::-1]
    if len(lbls.shape) == len(imgs.shape):
        lbls = lbls[:, :, ::-1]
    else:
        lbls = lbls[:, ::-1]
    return imgs, lbls, vars


def elastic_transform(image, label=None, sigma_range=(13, 50), alpha_range=(10, 150)):
    region_size = image.shape[-3:]
    axis_range = [np.arange(0, i) for i in region_size]
    coords = np.meshgrid(*axis_range, indexing='ij')
    coords = np.array([coords[i].astype(np.float) - (region_size[i]-1)/2. for i in range(len(region_size))], dtype=np.float)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    offset = scipy.ndimage.gaussian_filter(np.random.random(region_size[-2:]), sigma=sigma)*2 - 1
    offset = np.stack([offset]*region_size[0], axis=0)
    coords[-2:, :] = coords[-2:, :] + offset * alpha
    coords = np.array([coords[i] + (region_size[i]-1)/2. for i in range(len(region_size))], dtype=np.float)
    image = scipy.ndimage.map_coordinates(image, coords, order=3)
    if label is not None:
        if len(label.shape) == len(image.shape):
            label = scipy.ndimage.map_coordinates(label, coords, order=1)
        else:
            label = scipy.ndimage.map_coordinates(label, coords[1:, region_size[-3]//2, :], order=1)
    return image, label

