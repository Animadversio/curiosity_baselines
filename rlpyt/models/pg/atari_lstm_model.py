
import numpy as np
import torch
import torch.nn.functional as F

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel

from rlpyt.models.curiosity.encoders import UniverseHead, BurdaHead, MazeHead
from rlpyt.models.curiosity.disagreement import Disagreement
from rlpyt.models.curiosity.icm import ICM
from rlpyt.models.curiosity.ndigo import NDIGO
from rlpyt.models.curiosity.rnd import RND, RND_noerr
from rlpyt.models.curiosity.random_reward import RandomReward, RandomDistrReward, RandomReward_movthresh
from rlpyt.models.curiosity.count import CountBasedReward

RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work


class AtariLstmModel(torch.nn.Module):
    """Recurrent model for Atari agents: a convolutional network into an FC layer
    into an LSTM which outputs action probabilities and state-value estimate.
    """

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,  # Between conv and lstm.
            lstm_size=512,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            curiosity_kwargs=dict(curiosity_alg='none'),
            obs_stats=None
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()

        self.obs_stats = obs_stats
        if self.obs_stats is not None:
            self.obs_mean, self.obs_std = self.obs_stats

        if curiosity_kwargs['curiosity_alg'] != 'none':
            if curiosity_kwargs['curiosity_alg'] == 'icm':
                self.curiosity_model = ICM(image_shape=image_shape,
                                           action_size=output_size,
                                           feature_encoding=curiosity_kwargs['feature_encoding'],
                                           batch_norm=curiosity_kwargs['batch_norm'],
                                           prediction_beta=curiosity_kwargs['prediction_beta'],
                                           obs_stats=self.obs_stats,
                                           forward_loss_wt=curiosity_kwargs['forward_loss_wt'])
            elif curiosity_kwargs['curiosity_alg'] == 'disagreement':
                self.curiosity_model = Disagreement(image_shape=image_shape,
                                                    action_size=output_size,
                                                    feature_encoding=curiosity_kwargs['feature_encoding'],
                                                    batch_norm=curiosity_kwargs['batch_norm'],
                                                    prediction_beta=curiosity_kwargs['prediction_beta'],
                                                    obs_stats=self.obs_stats,
                                                    device=curiosity_kwargs['device'],
                                                    forward_loss_wt=curiosity_kwargs['forward_loss_wt'])
            elif curiosity_kwargs['curiosity_alg'] == 'ndigo':
                self.curiosity_model = NDIGO(image_shape=image_shape,
                                             action_size=output_size,
                                             obs_stats=self.obs_stats,
                                             horizon=curiosity_kwargs['pred_horizon'],
                                             feature_encoding=curiosity_kwargs['feature_encoding'],
                                             batch_norm=curiosity_kwargs['batch_norm'],
                                             num_predictors=curiosity_kwargs['num_predictors'],
                                             device=curiosity_kwargs['device'],
                                             )
            elif curiosity_kwargs['curiosity_alg'] == 'rnd':
                if curiosity_kwargs['no_error']:
                    self.curiosity_model = RND_noerr(image_shape=image_shape,
                                           prediction_beta=curiosity_kwargs['prediction_beta'],
                                           drop_probability=curiosity_kwargs['drop_probability'],
                                           gamma=curiosity_kwargs['gamma'],
                                           device=curiosity_kwargs['device'])
                else:
                    self.curiosity_model = RND(image_shape=image_shape,
                                           prediction_beta=curiosity_kwargs['prediction_beta'],
                                           drop_probability=curiosity_kwargs['drop_probability'],
                                           gamma=curiosity_kwargs['gamma'],
                                           device=curiosity_kwargs['device'],
                                           shuffle=curiosity_kwargs['shuffle'])
            # TODO: add our curiosity type, initialization of our curiosity algorithm
            elif curiosity_kwargs['curiosity_alg'] == 'random_reward':
                if curiosity_kwargs['use_distr']:
                    self.curiosity_model = RandomDistrReward(image_shape=image_shape,
                                           reward_scale=curiosity_kwargs['reward_scale'],
                                           zero_prob=curiosity_kwargs['zero_prob'], 
                                           nonneg=curiosity_kwargs['nonneg'],
                                           gamma=curiosity_kwargs['gamma'],
                                           device=curiosity_kwargs['device'])
                else:
                    self.curiosity_model = RandomReward(image_shape=image_shape,
                                           reward_scale=curiosity_kwargs['reward_scale'],
                                           # drop_probability=curiosity_kwargs['drop_probability'],
                                           nonneg=curiosity_kwargs['nonneg'],
                                           gamma=curiosity_kwargs['gamma'],
                                           device=curiosity_kwargs['device'])
            elif curiosity_kwargs['curiosity_alg'] == 'random_reward_mov':
                self.curiosity_model = RandomReward_movthresh(image_shape=image_shape,
                                                    reward_scale=curiosity_kwargs['reward_scale'],
                                                    update_freq=curiosity_kwargs['update_freq'],
                                                    decay_timescale=curiosity_kwargs['decay_timescale'],
                                                    gamma=curiosity_kwargs['gamma'],
                                                    nonneg=curiosity_kwargs['nonneg'],
                                                    device=curiosity_kwargs['device'],
                                                    )
            elif curiosity_kwargs['curiosity_alg'] == 'count':
                self.curiosity_model = CountBasedReward(image_shape=image_shape,
                                           alpha=curiosity_kwargs['reward_scale'],
                                           hashfun=curiosity_kwargs['hashfun'],
                                           device=curiosity_kwargs['device'])
            
            if curiosity_kwargs['feature_encoding'] == 'idf':
                self.conv = UniverseHead(image_shape=image_shape,
                                         batch_norm=curiosity_kwargs['batch_norm'])
                self.conv.output_size = self.curiosity_model.feature_size
            elif curiosity_kwargs['feature_encoding'] == 'idf_burda':
                self.conv = BurdaHead(image_shape=image_shape,
                                      output_size=self.curiosity_model.feature_size,
                                      batch_norm=curiosity_kwargs['batch_norm'])
                self.conv.output_size = self.curiosity_model.feature_size
            elif curiosity_kwargs['feature_encoding'] == 'idf_maze':
                self.conv = MazeHead(image_shape=image_shape,
                                     output_size=self.curiosity_model.feature_size,
                                     batch_norm=curiosity_kwargs['batch_norm'])
                self.conv.output_size = self.curiosity_model.feature_size
            elif curiosity_kwargs['feature_encoding'] == 'none':
                if image_shape[1:] == (5, 5):
                    self.conv = MazeHead(image_shape=image_shape,
                                     output_size=fc_sizes,
                                     batch_norm=False)
                else:
                    self.conv = Conv2dHeadModel(image_shape=image_shape,
                                            channels=channels or [16, 32],
                                            kernel_sizes=kernel_sizes or [8, 4],
                                            strides=strides or [4, 2],
                                            paddings=paddings or [0, 1],
                                            use_maxpool=use_maxpool,
                                            hidden_sizes=fc_sizes) # Applies nonlinearity at end.

        else:
            if image_shape[1:] == (5, 5):
                self.conv = MazeHead(image_shape=image_shape,
                                 output_size=fc_sizes,
                                 batch_norm=False)
            else:
                self.conv = Conv2dHeadModel(
                image_shape=image_shape,
                channels=channels or [16, 32],
                kernel_sizes=kernel_sizes or [8, 4],
                strides=strides or [4, 2],
                paddings=paddings or [0, 1],
                use_maxpool=use_maxpool,
                hidden_sizes=fc_sizes, # Applies nonlinearity at end.
            )

        self.lstm = torch.nn.LSTM(self.conv.output_size + output_size, lstm_size)
        self.pi = torch.nn.Linear(lstm_size, output_size)
        self.value = torch.nn.Linear(lstm_size, 1)


    def forward(self, image, prev_action, prev_reward, init_rnn_state):
        """
        Compute action probabilities and value estimate from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        *image_shape], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Recurrent layers processed as [T,B,H]. Used in
        both sampler and in algorithm (both via the agent).  Also returns the
        next RNN state.
        """       
        if self.obs_stats is not None: # don't normalize observation
            image = (image - self.obs_mean) / self.obs_std

        img = image.type(torch.float)  # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3) 

        fc_out = self.conv(img.view(T * B, *img_shape))
        lstm_input = torch.cat([
            fc_out.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)

        pi = F.softmax(self.pi(lstm_out.view(T * B, -1)), dim=-1)
        v = self.value(lstm_out.view(T * B, -1)).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return pi, v, next_rnn_state
