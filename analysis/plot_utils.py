"""Utils to plot or process heatmaps"""
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def make_grid_np(img_arr, nrow=8, padding=2, pad_value=0):
    if type(img_arr) is list:
        try:
            img_tsr = np.stack(tuple(img_arr), axis=3)
            img_arr = img_tsr
        except ValueError:
            raise ValueError("img_arr is a list and its elements do not have the same shape as each other.")
    nmaps = img_arr.shape[3]
    nchan = img_arr.shape[2] # 3 for RGB or 4 for RGBA
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(img_arr.shape[0] + padding), int(img_arr.shape[1] + padding)
    grid = np.zeros((height * ymaps + padding, width * xmaps + padding, nchan), dtype=img_arr.dtype)
    grid.fill(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height + padding: (y + 1) * height, x * width + padding: (x + 1) * width, :] = img_arr[:,:,:,k]
            k = k + 1
    return grid


def upscale_pix2square(img, scale: int):
    return np.repeat(np.repeat(img, scale, 0), scale, 1)


def summary_montage_heatmaps(heatmap_dir, loginterval=3, upscale=12, nrow=10):
    heatmap_col = []
    for i in range(loginterval, 10000, loginterval):
        if os.path.exists(join(heatmap_dir, f"{i}.png")):
            img = plt.imread(join(heatmap_dir, f"{i}.png"))
            heatmap_col.append(img)
        else:
            break
    if len(heatmap_col) == 0:
        return 
    print("Collect %d heatmaps"%len(heatmap_col))
    mtg = make_grid_np(heatmap_col, nrow=10)
    plt.imsave(join(heatmap_dir, "summary_heatmap.png"), upscale_pix2square(mtg, 12, ),)
    return mtg