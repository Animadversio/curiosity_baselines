import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
# from skimage.transform import rescale, resize, downscale_local_mean
from analysis.plot_utils import make_grid_np, upscale_pix2square

# Collect all heatmaps in the folder
loginterval = 3
# D:\DL_Projects\RL\curiosity_baselines
# heatmap_dir = r"results\ppo_deepmind_maze\run_0\heatmaps"
heatmap_dir = r"E:\DL_Projects\RL\curiosity_baselines\results\ppo_ICM_DM5RoomBouncing\run_0\heatmaps"
heatmap_dir = r"E:\DL_Projects\RL\curiosity_baselines\results\ppo_ICM_DM5Room\run_0\heatmaps"
heatmap_dir = r"E:\DL_Projects\RL\curiosity_baselines\results\ppo_count_DM5Room-v0\run_0\heatmaps"
heatmap_col = []
for i in range(loginterval, 2000, loginterval):
    if os.path.exists(join(heatmap_dir, f"{i}.png")):
        img = plt.imread(join(heatmap_dir, f"{i}.png"))
        heatmap_col.append(img)
    else:
        break
print("Collect %d heatmaps"%len(heatmap_col))
#%
mtg = make_grid_np(heatmap_col, nrow=10)
plt.imsave(join(heatmap_dir, "summary_heatmap.png"), upscale_pix2square(mtg, 12, ),)
#%
# plt.imshow(rescale(mtg, 5, order=0, multichannel=True, anti_aliasing=False))
plt.imshow(upscale_pix2square(mtg, 10, ))
plt.show()
#%%
