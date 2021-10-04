from __future__ import absolute_import

from augment_crop_and_pad import crop_and_pad

import numpy as np

from augment_utils import rotation_3d, rotation_2d, elastic_deformation, random_scale, \
    interpolate_img


def augment_mirroring(imgs, lbls=None,
                      p_mrr_axes=(0.2, 0.2, 0.2)):
    '''
    To mirror images and lbls
    :param imgs:
    :param lbls:
    :param p_mrr_axes:
    :return:
    '''
    n_dim = len(imgs.shape[2:])
    if not isinstance(p_mrr_axes, (tuple, list, np.ndarray)):
        p_mrr_axes = [p_mrr_axes] * n_dim

    if np.random.uniform(0, 1) <= p_mrr_axes[0]:
        imgs = imgs[:, :, ::-1]
        if lbls is not None:
            lbls = lbls[:, :, ::-1]
    if np.random.uniform(0, 1) <= p_mrr_axes[1]:
        imgs = imgs[:, :, :, ::-1]
        if lbls is not None:
            lbls = lbls[:, :, :, ::-1]
    if n_dim == 3 and np.random.uniform(0, 1) <= p_mrr_axes[2]:
        imgs = imgs[:, :, :, :, ::-1]
        if lbls is not None:
            lbls = lbls[:, :, :, :, ::-1]
    return imgs, lbls


def augment_translation(imgs, lbls=None,
                        p_trs_axes=0.2, dist_trs_axes=10,
                        pad_mode='edge'):
    '''
    To translate images and labels
    :param imgs:
    :param lbls:
    :param p_trs_axes:
    :param dist_trs_axes:
    :param pad_mode:
    :return:
    '''
    n_dim = len(imgs.shape[2:])
    if not isinstance(dist_trs_axes, (tuple, list, np.ndarray)):
        dist_trs_axes = [dist_trs_axes] * n_dim
    if not isinstance(p_trs_axes, (tuple, list, np.ndarray)):
        p_trs_axes = [p_trs_axes] * n_dim

    factor_trs_axes = [np.random.uniform(0, 1) < p_trs_axes[i] for i in range(n_dim)]
    dist_trs_axes = [np.random.uniform(-dist_trs_axes[i], dist_trs_axes[i]) for i in range(n_dim)]
    dist_trs_axes = np.array(factor_trs_axes).astype(int) * np.array(dist_trs_axes).astype(int)

    crop_first_point = [max(0, -dist_trs_axes[i]) for i in range(n_dim)]
    crop_end_point = [min(imgs.shape[2+i], imgs.shape[2+i]-dist_trs_axes[i]) for i in range(n_dim)]
    pad_before_part = [max(0, dist_trs_axes[i]) for i in range(n_dim)]
    pad_after_part = [max(0, -dist_trs_axes[i]) for i in range(n_dim)]

    croped_lbls = None
    if n_dim == 2:
        croped_imgs = imgs[:, :,
                      crop_first_point[0]: crop_end_point[0],
                      crop_first_point[1]: crop_end_point[1]]
        if lbls is not None:
            croped_lbls = lbls[:, :,
                          crop_first_point[0]: crop_end_point[0],
                          crop_first_point[1]: crop_end_point[1]]
    elif n_dim == 3:
        croped_imgs = imgs[:, :,
                      crop_first_point[0]: crop_end_point[0],
                      crop_first_point[1]: crop_end_point[1],
                      crop_first_point[2]: crop_end_point[2]]
        if lbls is not None:
            croped_lbls = lbls[:, :,
                          crop_first_point[0]: crop_end_point[0],
                          crop_first_point[1]: crop_end_point[1],
                          crop_first_point[2]: crop_end_point[2]]
    else:
        croped_imgs = None
        raise Exception('The dims of \'data\' should be 2 or 3 !')

    new_imgs = np.pad(croped_imgs, np.array([[0, 0]+pad_before_part, [0, 0]+pad_after_part]).transpose(),
                      mode=pad_mode)
    new_lbls = None
    if croped_lbls is not None:
        new_lbls = np.pad(croped_lbls, np.array([[0, 0] + pad_before_part, [0, 0] + pad_after_part]).transpose(),
                          mode=pad_mode)
    return new_imgs, new_lbls


def augment_elastic_rotation_scale(imgs, lbls=None,
                    patch_size=(20, 256, 256), patch_center_dist_from_border=(0, 128, 128), center_crop=True,
                    p_elc = 0.2, alpha_range=(0., 900.), sigma_range=(9., 13.),
                    p_rot_axes=(0.2, 0, 0), angles_rot_axes=(np.pi/12, 0, 0),
                    p_scl_axes=(0, 0.2, 0.2), rates_scl_axes=(0, 0.25, 0.25)):
    '''
    Augmentation with elastic deformation, random rotation and random scale.
    :param data: (n, c, x, y (, z ...))
    :param mask: (n, c, x, y (, z ...))
    :param patch_size: (px, py (, pz ...))
    :return:
    '''
    n_dim = len(imgs.shape[2:])
    if patch_size is None:
        patch_size = imgs.shape[2:]
    elif not isinstance(patch_size, (tuple, list, np.ndarray)):
        patch_size = [patch_size]*n_dim
    else:
        patch_size = patch_size

    axesIndices = [np.arange(0, i) for i in patch_size]
    coords = np.meshgrid(*axesIndices, indexing='ij')
    coords = np.array([coords[i].astype(float) - (np.max(coords[i])-1)/2. for i in range(n_dim)], dtype=float)

    augmented = False
    # First step, random elastic deformation.
    if np.random.uniform(0, 1) < p_elc:
        if isinstance(sigma_range, (tuple, list, np.ndarray)):
            sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        else:
            sigma = sigma_range
        if isinstance(alpha_range, (tuple, list, np.ndarray)):
            alpha = np.random.uniform(alpha_range[0], alpha_range[1])
        else:
            alpha = alpha_range
        coords = elastic_deformation(coords, alpha=alpha, sigma=sigma)
        augmented = True

    # Second step, random rotation
    if n_dim == 2:
        angle = angles_rot_axes[0] if isinstance(angles_rot_axes, (tuple, list, np.ndarray)) else angles_rot_axes
        angle = np.random.uniform(-angle, angle)
        coords = rotation_2d(coords, angle)
        augmented = True

    elif n_dim == 3:
        if not isinstance(angles_rot_axes, (tuple, list, np.ndarray)):
            angles_rot_axes = [angles_rot_axes] * n_dim
        factors_rot_axes = [np.random.uniform(0, 1) < p_rot_axes[i] for i in range(n_dim)]
        angles_rot_axes = [np.random.uniform(-angles_rot_axes[i], angles_rot_axes[i]) for i in range(n_dim)]
        angles_rot_axes = np.array(factors_rot_axes, dtype=float)*np.array(angles_rot_axes)
        coords = rotation_3d(coords, angles_rot_axes)
        augmented = True if np.alltrue(factors_rot_axes) else False

    # Third step, random scale
    if not isinstance(rates_scl_axes, (tuple, list, np.ndarray)):
        rates_scl_axes = [rates_scl_axes] * n_dim
    factors_scl_axes = [np.random.uniform(0, 1) < p_scl_axes[i] for i in range(n_dim)]
    rates_scl_axes = [np.random.uniform(-rates_scl_axes[i], rates_scl_axes[i]) for i in range(n_dim)]
    rates_scl_axes = np.array(factors_scl_axes, dtype=float)*np.array(rates_scl_axes) + 1.
    coords = random_scale(coords, rates_scl_axes)
    augmented = True if np.alltrue(factors_scl_axes) else False

    # get images
    if augmented:
        if not center_crop:
            patch_center = [np.random.uniform(patch_center_dist_from_border[i],
                                              imgs.shape[i+2]-patch_center_dist_from_border[i])
                            for i in range(n_dim)]
        else:
            patch_center = [(imgs.shape[i+2]-1)/2. for i in range(n_dim)]
        coords = np.array([coords[i] + patch_center[i] for i in range(n_dim)])

        newimgs = np.zeros([*imgs.shape[0:2], *patch_size], dtype=float)
        for bi in range(imgs.shape[0]):
            for chi in range(imgs.shape[1]):
                newimgs[bi, chi, :] = interpolate_img(imgs[bi, chi], coords)
        newlbls = None
        if lbls is not None:
            newlbls = np.zeros(imgs.shape[0:2] + patch_size, dtype=float)
            for bi in range(lbls.shape[0]):
                for chi in range(lbls.shape[1]):
                    newlbls[bi, chi, :] = interpolate_img(lbls[bi, chi], coords, order=0, is_lbl=True)
    else:
        newimgs, newlbls = crop_and_pad(imgs, lbls, patch_size, patch_center_dist_from_border, cent_crop=center_crop)

    return newimgs, newlbls

