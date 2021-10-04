import numpy as np
import scipy.ndimage


def augment_gaussian_noise(data,
                           variance_range=(0, 0.1),
                           p_noise=0.2,
                           noise_per_channel=True):
    '''
    :param data: (n, c, x, y (,z))
    :param variance_range:
    :param p_noise:
    :param noise_per_channel:
    :return:
    '''
    if np.random.random() > p_noise:
        return data
    if isinstance(variance_range, (tuple, list, np.ndarray)):
        if noise_per_channel:
            variance = [np.random.uniform(variance_range[0], variance_range[1]) for _ in range(data.shape[1])]
        else:
            variance = np.random.uniform(variance_range[0], variance_range[1])
    else:
        variance = variance_range[0]
    for ni in range(data.shape[0]):
        if isinstance(variance, (tuple, list, np.ndarray)):
            assert len(variance) == data.shape[1], '\'noise_per_channel\' requires all channel has respective variance'
            for ci in range(data.shape[1]):
                data[ni, ci] = data[ni, ci] + np.random.normal(0.0, scale=variance[ci], size=data.shape[2:])
        else:
            data[ni] = data[ni] + np.random.normal(0, scale=variance, size=data.shape[1:])
    return data


def augment_gaussian_blur(data,
                          sigma_range=(2, 4), p_blur=0.2,
                          blur_per_axis=True, blur_order=0):
    '''
    :param data: (n, c, x, y (,z))
    :param sigma_range:
    :param p_blur:
    :param blur_per_axis:
    :param blur_order:
    :return:
    '''
    if np.random.random() > p_blur:
        return data

    if isinstance(sigma_range, (tuple, list, np.ndarray)):
        if blur_per_axis:
            sigma = [np.random.uniform(np.min(sigma_range), np.max(sigma_range))
                     if np.min(sigma_range) < np.max(sigma_range)
                     else np.max(sigma_range) for _ in data.shape[2:]]
        else:
            sigma = np.random.uniform(np.min(sigma_range), np.max(sigma_range)) \
                    if np.min(sigma_range) < np.max(sigma_range) \
                    else np.max(sigma_range)
    else:
        sigma = sigma_range

    for ni in range(data.shape[0]):
        for ci in range(data.shape[1]):
            data[ni, ci] = scipy.ndimage.gaussian_filter(data[ni, ci], sigma=sigma, order=blur_order)
    return data

