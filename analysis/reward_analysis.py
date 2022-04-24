from mazeworld.envs import MazeWorld, DeepmindMazeWorld_with_goal
from rlpyt.models.curiosity import random_reward, rnd, disagreement, ndigo
from rlpyt.models.curiosity.random_reward import RandomReward, RandomDistrReward
from rlpyt.models.curiosity.rnd import RND
from rlpyt.models.curiosity.disagreement import Disagreement
from rlpyt.models.curiosity.ndigo import NDIGO
import matplotlib.pyplot as plt
import torch
import numpy as np
from pycolab.things import Sprite
#%%
from gym import make as gym_make
# env = gym_make('DeepmindMaze_goal-v0')
env = DeepmindMazeWorld_with_goal()
#%%
env.pycolab_init("tmp", False)
env.reset()
#%%
#%%
curiose_mod = RandomReward(image_shape=(4, 5, 5),
             reward_scale=1.0,
             gamma=0.99,
             nonneg=True,
             device='cuda')
#%%
obs, rew, done, info = env.step(3)
obs_t = torch.tensor(obs).float()#.cuda()
with torch.no_grad():
    rew_t = curiose_mod(obs_t.unsqueeze(0).unsqueeze(0))
print(rew_t)


#%%
env.current_game._sprites_and_drapes["P"]._position = Sprite.Position(2, 2)
obs, rew, done, info = env.step(4)
env.current_game.play_one_step(4)
#%%
env.current_game._update_groups[0][1][0]._position = Sprite.Position(1, 5)
#%%
from pycolab.examples.deepmind_maze import PlayerSprite
# env.current_game._update_groups[0][1][0] = PlayerSprite(Sprite.Position(17, 17), Sprite.Position(3, 2), "P")
# env.current_game._update_groups[0][1][0]._teleport((1,2))

obs_arr = []
rew_arr = []
row, col = env.current_game._board.board.shape
for ri in range(row):
    for ci in range(col):
        if env.current_game._board.board[ri, ci] == ord(" "):
            env.current_game._sprites_and_drapes["P"]._teleport((ri, ci))
            obs, rew, done, info = env.step(4)
            obs_arr.append(obs)
obs_arr = np.array(obs_arr)
obs_tsr = torch.tensor(obs_arr).float().unsqueeze(1)
#%%

curiose_mod = RandomReward(image_shape=(4, 5, 5),
             reward_scale=1.0, gamma=0.99, nonneg=True, device='cuda')
with torch.no_grad():
  intrrew_t, B, T = curiose_mod(obs_tsr)

plt.hist(intrrew_t.numpy().squeeze(), bins=35)
plt.show()
#%%
curiose_mod2 = RandomDistrReward(image_shape=(4, 5, 5), zero_prob=0.5,
             reward_scale=1.0, gamma=0.99, nonneg=True, device='cuda')
with torch.no_grad():
  intrrew_t, B, T = curiose_mod2(obs_tsr)

plt.hist(intrrew_t.numpy().squeeze(), bins=35)
plt.show()
#%%
reward_assoc = torch.rand(1, 1, 4, 5, 5).abs()
mask = torch.rand(1, 1, 4, 5, 5)
reward_assoc[mask < 0.7] = 0
#%%
intrew_inprod = torch.sum(reward_assoc * obs_tsr, dim=(2,3,4))
plt.hist(intrew_inprod.numpy().squeeze(), bins=35)
plt.show()
#%%
with torch.no_grad():
  intrrew_t, B, T = curiose_mod2(obs_tsr)
#%%
weights = torch.einsum("BT,BTCHW,BT->CHW", intrrew_t, obs_tsr, 1/(1E-5+obs_tsr.sum(dim=(2,3,4))))
#%%
intrew_prodW = torch.einsum("CHW,BTCHW->BT", weights, obs_tsr, )
plt.hist(intrew_prodW.numpy().squeeze(), bins=35)
plt.show()

#%% 
# self._teleport(self._empty_coords[np.random.choice(len(self._empty_coords))])
