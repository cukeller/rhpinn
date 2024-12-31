# simple matplotlib-based plots that are useful for showing rhpinn results
# author: Christoph U.Keller, ckeller@nso.edu

import matplotlib.pyplot as plt
import numpy as np


def comparison(sim, pinn, title=' ', invertY=False, title2='fit', aspect='equal', bar=True, save=False):
    min = np.concatenate((sim.flatten(), pinn.flatten())).min()
    max = np.concatenate((sim.flatten(), pinn.flatten())).max()
    fig = plt.figure()
    fig_a = fig.add_subplot(1, 2, 1)
    im = fig_a.imshow(sim, vmin = min, vmax = max, aspect=aspect)
    if (invertY):
        im.axes.invert_yaxis()
    plt.title(title)
    fig_b = fig.add_subplot(1, 2, 2)
    im = fig_b.imshow(pinn, vmin = min, vmax = max, aspect=aspect)
    if (invertY):
        im.axes.invert_yaxis()
    plt.title(title2)
    if bar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    if (save):
        plt.savefig(title + '.png')
        plt.close()
    else:
        plt.show()

def image(diff, title='', invertY=False, aspect='equal', cmap='viridis', save=False):
    fig = plt.figure()
    im = plt.imshow(diff, cmap=cmap, aspect=aspect)
    if (invertY):
        im.axes.invert_yaxis()
    plt.title(title)
    fig.colorbar(im)
    if (save):
        plt.savefig(title + '.png')
        plt.close()
    else:
        plt.show()
