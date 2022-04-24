"""Utils to plot or process heatmaps"""
import os
from os.path import join
import numpy as np
import pandas as pd
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


def get_coverage_curve(heatmap_dir, loginterval=3, total_num=None, show=True):
    visitmap_col = []
    iter_col = []
    for itr in range(loginterval, 2000, loginterval):
        if os.path.exists(join(heatmap_dir, f"{itr}.npy")):
            visitmap = np.load(join(heatmap_dir, f"{itr}.npy"))
            visitmap_col.append(visitmap)
            iter_col.append(itr)
        else:
            break
    if len(visitmap_col) == 0:
        return np.array([]), np.array([]), pd.DataFrame({"iter": [], "visit_states_num": [], "run_name": []})
    else:
        print("Collect %d heatmaps" % len(visitmap_col))
        visit_tsr = np.array(visitmap_col)
        iter_vec = np.array(iter_col)
        visit_states_num = np.count_nonzero(visit_tsr, axis=(1, 2))
        #
        # plt.plot(visit_states_num, label="visit states num")
        # runname = heatmap_dir[len(result_root)+1:-len("heatmaps")-1]
        runname = heatmap_dir[heatmap_dir.find("results\\")+len("results\\"):-len("heatmaps")-1]
        plt.plot(iter_vec, visit_states_num, )
        plt.xlabel("Iteration")
        plt.ylabel("visit states")
        plt.title(f"coverage curve of {runname}\n total{total_num}")
        plt.savefig(join(heatmap_dir, "coverage_curve.png"))
        if show:
            plt.show()
        df = pd.DataFrame({"iter": iter_vec, "visit_states_num": visit_states_num, "run_name": runname})
        df.to_csv(join(heatmap_dir, "state_coverage.csv"))
        return iter_vec, visit_states_num, df