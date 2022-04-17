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

import torch


class SimHash(object) :
    """https://github.com/clementbernardd/Count-Based-Exploration/blob/main/python/simhash.py"""
    def __init__(self, state_emb, k, device, EPS=1E-4) :
        ''' Hashing between continuous state space and discrete state space '''
        self.hash = {}
        self.A = np.random.normal(0,1, (k , state_emb))
        self.device = device
        self.EPS = EPS

    def count(self, states):
        ''' Increase the count for the states and retourn the counts '''
        counts = []
        for state in states:
            key = str(np.sign(self.A @ state.detach().cpu().numpy()).tolist())
            if key in self.hash:
                self.hash[key] = self.hash[key] + 1
            else:
                self.hash[key] = 1
            counts.append(self.hash[key])

        return torch.from_numpy(np.array(counts)).to(self.device)

    def retrieve(self, states):
        ''' Increase the count for the states and retourn the counts '''
        counts = []
        for state in states:
            key = str(np.sign(self.A @ state.detach().cpu().numpy()).tolist())
            if key in self.hash:
                counts.append(self.hash[key])
            else:
                counts.append(self.EPS)

        return torch.from_numpy(np.array(counts)).to(self.device)

# counter = SimHash(75, 256, "cuda")
class CountBasedReward(nn.Module):
    def __init__(
            self,
            image_shape,
            alpha,
            gamma=0.99,
            EPS=1E-2,
            device='cpu',
            hashfun="SimHash"
            ):
        super(CountBasedReward, self).__init__()
        self.image_shape = image_shape
        self.alpha = alpha
        self.gamma = gamma
        self.feature_size = 256
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')
        self.count = {}
        self.EPS = EPS
        self.hashfun = hashfun
        if self.hashfun == "SimHash":
            self.counter = SimHash(np.prod(self.image_shape), 256, "cuda")

    def compute_bonus(self, next_observation, done):
        # raise NotImplementedError
        # if not next_observation.shape[2:] == torch.Size([4,5,5]):
        #     next_observation = next_observation[:,:,-1,:,:]
        #     next_observation = next_observation.unsqueeze(2)
        features = next_observation.flatten(start_dim=1)
        counts = self.counter.retrieve(features)
        # key = torch.flatten(next_observation)
        # if key not in self.count:
        #     obs_count = self.EPS
        # else:
        #     obs_count = self.count[key] + self.EPS
        rewards = 1 / torch.sqrt(counts).unsqueeze(1)
        rewards *= done
        return self.alpha * rewards

    def compute_loss(self, observations, valid):
        # raise NotImplementedError
        # phi, predicted_phi, T, B = self.forward(observations, done=None)
        # if not observations.shape[2:] == torch.Size([4, 5, 5]):
        #     observations = observations[:, :, -1, :, :]
        #     observations = observations.unsqueeze(2)
        features = observations.flatten(start_dim=1)
        counts = self.counter.count(features)
        # key = torch.flatten(observations)
        # if key not in self.count:
        #     self.count[key] = 1
        # else:
        #     self.count[key] += 1
        return 0

