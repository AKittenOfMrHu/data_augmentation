import numpy as np


def crop_and_pad(imgs, lbls=None, patch_size=None,
                 cent_dist_from_border=None, cent_crop=False,
                 pad_mode='edge'):
    '''
    :param data: (n, c, x, y (z, ...))
    :patch_size: (dx, dy (,dz, ...))
    :return:
    '''
    n_dim = len(imgs.shape[2:])
    assert patch_size is not None, '\'patch_size\' must be not None!'
    if not isinstance(patch_size, (tuple, list, np.ndarray)):
        patch_size = [patch_size] * n_dim
    if not cent_crop and not isinstance(cent_dist_from_border, (tuple, list, np.ndarray)):
        assert cent_dist_from_border is not None, 'The \'cent_dist_from_border\' should be given!'
        cent_dist_from_border = [cent_dist_from_border] * n_dim
    if cent_crop:
        patch_center = [(imgs.shape[2+i]-1)//2 for i in range(n_dim)]
    else:
        patch_center = [int(np.random.uniform(cent_dist_from_border[i], imgs.shape[2+i]-cent_dist_from_border[i]))
                        for i in range(n_dim)]

    crop_frist_point = [patch_center[i]-patch_size[i]//2 for i in range(n_dim)]
    crop_end_point = [crop_frist_point[i] + patch_size[i] for i in range(n_dim)]
    pad_before_part = [abs(min(0, crop_frist_point[i])) for i in range(n_dim)]
    pad_after_part = [max(0, crop_end_point[i] - imgs.shape[2+i]) for i in range(n_dim)]
    crop_frist_point_corrected = [max(0, crop_frist_point[i]) for i in range(n_dim)]
    crop_end_point_corrected = [min(imgs.shape[2+i], crop_end_point[i]) for i in range(n_dim)]

    print(imgs.shape[2:])
    print(crop_frist_point_corrected, crop_end_point_corrected)
    print(pad_before_part, pad_after_part)
    croped_lbls = None
    if n_dim == 2:
        croped_imgs = imgs[:, :,
                      crop_frist_point_corrected[0]: crop_end_point_corrected[0],
                      crop_frist_point_corrected[1]: crop_end_point_corrected[1]]
        if lbls is not None:
            croped_lbls = lbls[:, :,
                          crop_frist_point_corrected[0]: crop_end_point_corrected[0],
                          crop_frist_point_corrected[1]: crop_end_point_corrected[1]]
    elif n_dim == 3:
        croped_imgs = imgs[:, :,
                      crop_frist_point_corrected[0]: crop_end_point_corrected[0],
                      crop_frist_point_corrected[1]: crop_end_point_corrected[1],
                      crop_frist_point_corrected[2]: crop_end_point_corrected[2]]
        if lbls is not None:
            croped_lbls = lbls[:, :,
                          crop_frist_point_corrected[0]: crop_end_point_corrected[0],
                          crop_frist_point_corrected[1]: crop_end_point_corrected[1],
                          crop_frist_point_corrected[2]: crop_end_point_corrected[2]]
    else:
        croped_imgs = None
        raise Exception(f'the \'data\' shape should be (b, c, x, y (, z)) ! ')

    new_imgs = np.pad(croped_imgs, np.array([[0, 0]+pad_before_part, [0, 0]+pad_after_part]).transpose(),
                      mode=pad_mode)
    new_lbls = None
    if croped_lbls is not None:
        new_lbls = np.pad(croped_lbls, np.array([[0, 0] + pad_before_part, [0, 0] + pad_after_part]).transpose(),
                          mode=pad_mode)
    return new_imgs, new_lbls
