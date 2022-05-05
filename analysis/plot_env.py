"""Plot the maze environment"""
import numpy as np
import matplotlib.pyplot as plt
from analysis.plot_utils import upscale_pix2square
from rlpyt.envs.pycolab.pycolab.examples.deepmind_maze import COLOUR_FG
from rlpyt.envs.gym_schema import deepmind_make
COLOUR_FG_RGB = {k: ((np.array(v)/1000 * 255).astype(np.uint8)) for k, v in COLOUR_FG.items()}

#%% Use the same color scheme as in NIDGO paper.
COLOUR_FG_NEW = {' ': (0, 0, 0),
                 '@': (254, 169, 0),
                 '#': (62, 62, 62),
                 'P': (255, 255, 255),
                 'a': (175, 255, 13),
                 'b': ( 21,  0, 253),
                 'c': (  0, 248,  67),
                 'd': (254, 0, 131),
                 'e': (255, 0,  0),}
COLOUR_FG_RGB = {k: np.array(v).astype(np.uint8) for k, v in COLOUR_FG_NEW.items()}
#%%
envsfx = "mid"
env = deepmind_make(game=f"DeepmindMaze_goal-{envsfx}-v0", obs_type="mask", max_steps_per_episode=500,
              logdir="tmp", log_heatmaps=False, no_negative_reward=True)
board_np = env.env.env.current_game._board.board
board_masks = env.env.env.current_game._board.layers
img = np.zeros((*board_np.shape, 3), dtype=np.uint8)
for chr, msk in board_masks.items():
    img[msk, :] = COLOUR_FG_RGB[chr]

img_us = upscale_pix2square(img, 30)
plt.imshow(img_us)
plt.axis('off')
plt.title("DeepmindMaze Medium", fontsize=16)
plt.tight_layout()
plt.savefig(f"figures/DMMaze_difficulty_titrate/DMMaze_{envsfx}_env.png", dpi=300)
plt.show()
plt.imsave(f"figures/DMMaze_difficulty_titrate/DMMaze_{envsfx}_env_pure.png", img_us)
#%% legends

img_lgd = np.zeros((1, len(COLOUR_FG_RGB), 3), dtype=np.uint8)
for i, (k, color) in enumerate(COLOUR_FG_RGB.items()):
    img_lgd[0, i, :] = COLOUR_FG_RGB[k]
img_lgd_us = upscale_pix2square(img_lgd, 30)
plt.imsave(f"figures/DMMaze_difficulty_titrate/DMMaze_env_legend.png", img_lgd_us)
# plt.imshow(img_lgd_us)
# plt.axis('off')
