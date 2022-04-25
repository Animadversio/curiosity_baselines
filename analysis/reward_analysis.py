from mazeworld.envs import MazeWorld, DeepmindMazeWorld_with_goal
# from rlpyt.models.curiosity import random_reward, rnd, disagreement, ndigo
from rlpyt.models.curiosity.random_reward import RandomReward, RandomDistrReward
from rlpyt.models.curiosity.rnd import RND
from rlpyt.models.curiosity.disagreement import Disagreement
from rlpyt.models.curiosity.ndigo import NDIGO
import matplotlib.pyplot as plt
import torch
import numpy as np
from pycolab.things import Sprite
import seaborn as sns
from gym import make as gym_make
#%%
def collect_all_obs(env, ):
    env.reset()
    obs_arr = []
    done_arr = []
    coords = []
    row, col = env.current_game._board.board.shape
    for ri in range(row):
        for ci in range(col):
            if env.current_game._board.board[ri, ci] == ord(" "):
                env.current_game._sprites_and_drapes["P"]._teleport((ri, ci))
                obs, rew, done, info = env.step(4)
                obs_arr.append(obs)
                done_arr.append(done)
                coords.append((ri, ci))

    obs_arr = np.array(obs_arr)
    done_arr = np.array(done_arr)
    coords_arr = np.array(coords)
    done_tsr = torch.tensor(done_arr)
    obs_tsr = torch.tensor(obs_arr).float().unsqueeze(1)
    return obs_tsr, done_tsr, coords_arr


def compute_bonus_from_obs(env, curios_mod, obs_tsr, done_tsr, coords_arr):
    row, col = env.current_game._board.board.shape
    with torch.no_grad():
        intr_rew = curios_mod.compute_bonus(obs_tsr, done_tsr.int().unsqueeze(1))

    reward_map = torch.zeros(row, col)
    for i, (ri, ci) in enumerate(coords_arr):
        reward_map[ri, ci] = intr_rew[i]

    return reward_map, intr_rew, obs_tsr


def calculate_bonus_map(env, curios_mod):
    obs_tsr, done_tsr, coords_arr = collect_all_obs(env, )
    with torch.no_grad():
        intr_rew = curios_mod.compute_bonus(obs_tsr, done_tsr.int().unsqueeze(1))

    reward_map = torch.zeros(row, col)
    for i, (ri, ci) in enumerate(coords_arr):
        reward_map[ri, ci] = intr_rew[i]

    return reward_map, intr_rew, obs_tsr


def visualize_bonus(reward_map, intr_rew, modelname=""):
    plt.hist(intr_rew.numpy().squeeze(), bins=35)
    plt.title(modelname)
    plt.show()
    sns.heatmap(reward_map)
    plt.axis("equal")
    plt.title(modelname)
    plt.show()
#%%
# env = DeepmindMazeWorld_with_goal()
#
env = gym_make('DeepmindMaze_goal-v0')
env.pycolab_init("tmp", False)
#%%
obs_tsr, done_tsr, coords_arr = collect_all_obs(env, )
#%%
curios_mod = RND(image_shape=(7, 5, 5), device="cuda")
reward_map, intr_rew, obs_tsr = compute_bonus_from_obs(env, curios_mod,
                                   obs_tsr, done_tsr, coords_arr)
# reward_map, intr_rew, obs_tsr = calculate_bonus_map(env, curios_mod)
visualize_bonus(reward_map, intr_rew, "RND initial state")

#%%
curios_mod = RandomReward(image_shape=(7, 5, 5),
             reward_scale=1.0, gamma=0.99, nonneg=True, device='cuda')
reward_map, intr_rew, obs_tsr = compute_bonus_from_obs(env, curios_mod,
                                   obs_tsr, done_tsr, coords_arr)
# reward_map, intr_rew, obs_tsr = calculate_bonus_map(env, curios_mod)
visualize_bonus(reward_map, intr_rew, "Random Reward Network")

#%%
curios_mod = RandomDistrReward(image_shape=(4, 5, 5), zero_prob=0.5,
             reward_scale=0.002, gamma=0.99, nonneg=True, device='cuda')
reward_map, intr_rew, obs_tsr = compute_bonus_from_obs(env, curios_mod,
                                   obs_tsr, done_tsr, coords_arr)
# reward_map, intr_rew, obs_tsr = calculate_bonus_map(env, curios_mod)
visualize_bonus(reward_map, intr_rew, "Random distribution")


#%%
from easydict import EasyDict
from os.path import join
import json
import pickle as pkl
expdir = r"results/ppo_RND_DM5roomRandGoal_dp01/run_0"
# pkl.load(open(join(expdir, "params.pkl"), "r"))
net_params = torch.load(join(expdir, "params.pkl"))
args = EasyDict(json.load(open(join(expdir, "arguments.json"))))
list(net_params["agent_state_dict"])
#%%
from rlpyt.agents.pg.atari import AtariLstmAgent
model_args = dict(curiosity_kwargs=dict(curiosity_alg="RND", ))
model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
model_args['curiosity_kwargs']['drop_probability'] = args.drop_probability
model_args['curiosity_kwargs']['gamma'] = args.discount
model_args['curiosity_kwargs']['device'] = args.sample_mode
model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
agent = AtariLstmAgent(initial_model_state_dict=net_params["agent_state_dict"],
                       model_kwargs=model_args, no_extrinsic=False)
#%%
obs_arr = []
done_arr = []
coords = []
row, col = env.current_game._board.board.shape
for ri in range(row):
    for ci in range(col):
        if env.current_game._board.board[ri, ci] == ord(" "):
            env.current_game._sprites_and_drapes["P"]._teleport((ri, ci))
            obs, rew, done, info = env.step(4)
            obs_arr.append(obs)
            done_arr.append(done)
            coords.append((ri, ci))
obs_arr = np.array(obs_arr)
done_arr = np.array(done_arr)
coords_arr = np.array(coords)
done_tsr = torch.tensor(done_arr)
obs_tsr = torch.tensor(obs_arr).float().unsqueeze(1)
#%%
RND_mod = RND(image_shape=(4, 5, 5), device="cuda")
#%%
with torch.no_grad():
    intr_rew = RND_mod.compute_bonus(obs_tsr, done_tsr.int().unsqueeze(1))

#%%
plt.hist(intr_rew.numpy().squeeze(), bins=35)
plt.show()
#%%
reward_map = torch.zeros(row, col)
for i, (ri, ci) in enumerate(coords_arr):
    reward_map[ri, ci] = intr_rew[i]
#%%
import seaborn as sns
sns.heatmap(reward_map)
plt.axis("equal")
plt.show()
#%%
curiose_mod = RandomReward(image_shape=(4, 5, 5),
             reward_scale=1.0,
             gamma=0.99,
             nonneg=True,
             device='cuda')
#%%
obs, rew, done, info = env.step(3)
obs_t = torch.tensor(obs).float()  # .cuda()
with torch.no_grad():
    rew_t = curiose_mod(obs_t.unsqueeze(0).unsqueeze(0))
print(rew_t)
#%%
from rlpyt.envs.base import EnvSpaces
env_spaces = EnvSpaces(observation=env.observation_space,
        action=env.action_space, )
agent.initialize(env_spaces, share_memory=True, )

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
