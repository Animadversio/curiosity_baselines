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

        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')

    def compute_bonus(self, next_observation, done):
    	raise NotImplementedError
    	return alpha * bonus

    def compute_loss(self, observations, valid):
    	raise NotImplementedError
        phi, predicted_phi, T, B = self.forward(observations, done=None)
        return forward_loss