import numpy as np
import matplotlib.pyplot as plt


def visualization_2d(imgs):
    plt.figure(figsize=(10, 10))
    if len(imgs.shape) == 3:
        for i, si in enumerate(np.linspace(0, imgs.shape[0]-1, 9)):
            #print(i, si, int(np.round(si)))
            si = int(np.round(si))
            if si > imgs.shape[0]-1:
                si = imgs.shape[0]-1
            slc_i = imgs[si]
            axs = plt.subplot(3, 3, i+1)
            axs.imshow(slc_i, 'gray')
    elif len(imgs.shape) == 2:
        plt.imshow(imgs, 'gray')
    plt.show()
    return