
set common_param=-alg ppo -iterations 2500000 -log_heatmaps -lstm -num_envs 48 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 48 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no

rem Small scale 
FOR /L %i IN (0,1,1) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT40UF100/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 20 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT500UF100/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 500 -update_freq 100 %common_param% 
)
FOR /L %i IN (0,1,1) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF100/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 100 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF5/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 100 -update_freq 5 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF40/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 100 -update_freq 40 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF500/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 100 -update_freq 500 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT40UF40/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 40 -update_freq 40 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF100_neg/run_%i -feature_encoding idf_maze -nonneg 0 -reward_scale 0.05 -decay_timescale 100 -update_freq 100 %common_param% 
)

rem Larger scale matching the scale of RND. 

FOR /L %i IN (2,1,4) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT40UF100RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 20 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT500UF100RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 500 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF100RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 100 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF5RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 100 -update_freq 5 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF40RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 100 -update_freq 40 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF500RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 100 -update_freq 500 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT40UF40RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 40 -update_freq 40 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results/ppo_randrewmov_DMMaze-dif-DT100UF100RS1_neg/run_%i -feature_encoding idf_maze -nonneg 0 -reward_scale 1 -decay_timescale 100 -update_freq 100 %common_param% 
)



rem Shuffle version RND for the DeepmindMaze 

FOR /L %i IN (0,1,2) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-dif_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1 -shuffle
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-dif_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5 -shuffle
)

FOR /L %i IN (3,1,5) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-dif_dp00/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.0 -shuffle
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-dif_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1 -shuffle
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-dif_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5 -shuffle
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-dif_dp095/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.95 -shuffle
)

rem Control for the shuffled version RND 
FOR /L %i IN (3,1,6) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg none -log_dir results_DMMaze/ppo_none_DMMaze-dif/run_%i -feature_encoding none %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results_DMMaze/ppo_RND_DMMaze-dif_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results_DMMaze/ppo_RND_DMMaze-dif_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.0
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results_DMMaze/ppo_RND_DMMaze-dif_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results_DMMaze/ppo_RND_DMMaze-dif_dp095/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.95


python launch.py -env DeepmindMaze_goal-mid-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-mid_dp00/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.0 -shuffle
python launch.py -env DeepmindMaze_goal-mid-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-mid_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1 -shuffle
python launch.py -env DeepmindMaze_goal-mid-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-mid_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5 -shuffle
python launch.py -env DeepmindMaze_goal-mid-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-mid_dp095/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.95 -shuffle
)


python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results/ppo_RNDShfl_DMMaze-dif_dp09/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9 -shuffle
