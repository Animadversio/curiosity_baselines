import os
from PIL import Image
import numpy as np
import torch
from torch import nn
torch.set_printoptions(edgeitems=3)

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.models.utils import Flatten
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead


class CountBasedReward(nn.Module):
	def __init__(
            self, 
            image_shape, 
            alpha, 
            gamma=0.99,
            device='cpu'
            ):
		super(CountBasedReward, self).__init__()
        self.image_shape = image_shape
        self.alpha = alpha
        self.gamma = gamma
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')
        self.count = {}
        
       
    def compute_bonus(self, next_observation, done):
    	#raise NotImplementedError
        if not next_observation.shape[2:] == torch.Size([4,5,5]):
            next_observation = next_observation[:,:,-1,:,:]
            next_observation = next_observation.unsqueeze(2)
        key = torch.flatten(next_observation)
        if key not in self.count:
            obs_count = 1
        else:
            obs_count = self.count[key]+1
        rewards = torch.from_numpy(self.alpha / np.sqrt(ob_counts))
        rewards *= done
        return rewards

    def compute_loss(self, observations, valid):
    	#raise NotImplementedError
        #phi, predicted_phi, T, B = self.forward(observations, done=None)
        if not observations.shape[2:] == torch.Size([4,5,5]):
            observations = next_observation[:,:,-1,:,:]
            observations = next_observation.unsqueeze(2)
        key = torch.flatten(observations)
        if key not in self.count:
            self.count[key] = 1
        else:
            self.count[key] += 1
        return 0