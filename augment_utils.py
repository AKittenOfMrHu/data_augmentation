import numpy as np
import scipy.ndimage


def interpolate_img(img, coords, order=3, mode='constant', cval=0.0, is_lbl=False):
    if is_lbl and order != 0:
        '''
        Interpolation on all categories with order>0 in lbl will cause the regions 
        belonging to the same label with different values. Therefore, it is better to
        deal with different labels separately.
        '''
        categories = np.unique(img)
        newimg = np.zeros(coords.shape[1:], dtype=img.dtype)
        for i, category in enumerate(categories):
            mpd_img = scipy.ndimage.map_coordinates((img == category).astype(float),
                                                    coords, order=order, mode=mode, cval=cval)
            newimg[mpd_img > 0.5] = category
        return newimg
    else:
        return scipy.ndimage.map_coordinates(img, coords, order=order, mode=mode, cval=cval)


def elastic_deformation(coords, sigma=10, alpha=200):
    '''
    :param data:
    :param coords:
    :return:
    '''
    n_dim = len(coords)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            scipy.ndimage.gaussian_filter(np.random.random(coords.shape[1:]) * 2. - 1,
                                          sigma=sigma)
        )
    coords += np.array(offsets)*alpha
    return coords


def rotation_3d(coords, angles):
    matrix = np.identity(len(coords))
    matrix_x = np.dot(matrix, np.array([[1, 0, 0],
                                        [0, np.cos(angles[0]), -np.sin(angles[0])],
                                        [0, np.sin(angles[0]), np.cos(angles[0])]]))
    matrix_xy = np.dot(matrix_x, np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                                           [0, 1, 0],
                                           [-np.sin(angles[1]), 0, np.cos(angles[1])]]))
    matrix_xyz = np.dot(matrix_xy, np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                                             [np.sin(angles[2]), np.cos(angles[2]), 0],
                                             [0, 0, 1]]))
    new_coords = np.dot(coords.reshape(len(coords), -1).transpose(), matrix_xyz).transpose().reshape(coords.shape)
    return new_coords


def rotation_2d(coords, angle):
    matrix = np.array([[np.cos(angle), np.sin(angle)],
                       [-np.sin(angle), np.cos(angle)]])
    print(np.dot(coords.reshape(len(coords), -1).transpose(), matrix).transpose().shape)
    new_coords = np.dot(coords.reshape(len(coords), -1).transpose(), matrix).transpose().reshape(coords.shape)
    return new_coords


def random_scale(coords, factors):
    if isinstance(factors, (tuple, list, np.ndarray)):
        assert len(coords) == len(factors)
        for i in range(len(factors)):
            coords[i] *= factors[i]
    else:
        coords *= factors
    return coords
