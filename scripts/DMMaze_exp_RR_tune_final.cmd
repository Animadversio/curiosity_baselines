

set common_param=-alg ppo -iterations 2500000 -log_heatmaps -lstm -num_envs 48 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 48 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no

FOR /L %i IN (2,1,5) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT40UF100/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 20 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT40UF40/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 40 -update_freq 40 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT100UF100/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 100 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT100UF40/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 100 -update_freq 40 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT100UF100/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 0.05 -decay_timescale 100 -update_freq 100 %common_param% 
)


FOR /L %i IN (3,1,6) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -use_distr  -log_dir results_DMMaze/ppo_randdstr_DMMaze-dif_sprs01/run_%i -feature_encoding idf_maze -zero_prob 0.1 -nonneg 1 -reward_scale 0.002 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -use_distr  -log_dir results_DMMaze/ppo_randdstr_DMMaze-dif_sprs05/run_%i -feature_encoding idf_maze -zero_prob 0.5 -nonneg 1 -reward_scale 0.002 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -use_distr  -log_dir results_DMMaze/ppo_randdstr_DMMaze-dif_sprs08/run_%i -feature_encoding idf_maze -zero_prob 0.8 -nonneg 1 -reward_scale 0.002 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -use_distr  -log_dir results_DMMaze/ppo_randdstr_DMMaze-dif_sprs095/run_%i -feature_encoding idf_maze -zero_prob 0.95 -nonneg 1 -reward_scale 0.002 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -use_distr  -log_dir results_DMMaze/ppo_randdstr_DMMaze-dif_sprs01_neg/run_%i -feature_encoding idf_maze -zero_prob 0.1 -reward_scale 0.002 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -use_distr  -log_dir results_DMMaze/ppo_randdstr_DMMaze-dif_sprs05_neg/run_%i -feature_encoding idf_maze -zero_prob 0.5 -reward_scale 0.002 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -use_distr  -log_dir results_DMMaze/ppo_randdstr_DMMaze-dif_sprs08_neg/run_%i -feature_encoding idf_maze -zero_prob 0.8 -reward_scale 0.002 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -use_distr  -log_dir results_DMMaze/ppo_randdstr_DMMaze-dif_sprs095_neg/run_%i -feature_encoding idf_maze -zero_prob 0.95 -reward_scale 0.002 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd -no_error -log_dir results_DMMaze/ppo_randDrift_DMMaze-dif_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd -no_error -log_dir results_DMMaze/ppo_randDrift_DMMaze-dif_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd -no_error -log_dir results_DMMaze/ppo_randDrift_DMMaze-dif_dp09/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9
python launch.py -env Maze-Lv1-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_MazeLv1_r10/run_0 -feature_encoding idf_maze %common_param% 
)


FOR /L %i IN (3,1,6) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results_DMMaze/ppo_RND_DMMaze-dif_dp10/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -log_dir results_DMMaze/ppo_randrew_DMMaze-dif/run_%i -feature_encoding idf_maze %common_param% -reward_scale 1.0
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -log_dir results_DMMaze/ppo_randrew_DMMaze-dif_nonneg/run_%i -feature_encoding idf_maze %common_param% -nonneg 1 -reward_scale 1.0
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward -log_dir results_DMMaze/ppo_randrew_DMMaze-dif_nonneg_r5/run_%i -feature_encoding idf_maze %common_param% -nonneg 1 -reward_scale 5.0
)


FOR /L %i IN (4,1,7) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT100UF100RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 100 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT100UF40RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 100 -update_freq 40 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT40UF100RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 20 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT40UF40RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 40 -update_freq 40 %common_param% 
)

FOR /L %i IN (6,1,8) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results_DMMaze/ppo_RND_DMMaze-dif_dp00/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.0
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd -shuffle -log_dir results_DMMaze/ppo_RNDShfl_DMMaze-dif_dp095/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.95
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd -shuffle -log_dir results_DMMaze/ppo_RNDShfl_DMMaze-dif_dp10/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0
)
FOR /L %i IN (9,1,11) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd  -log_dir results_DMMaze/ppo_RND_DMMaze-dif_dp00/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.0
)
FOR /L %i IN (6,1,9) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg rnd -shuffle -log_dir results_DMMaze/ppo_RNDShfl_DMMaze-dif_dp00/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.0
)

FOR /L %i IN (6,1,9) DO (
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT100UF5RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 100 -update_freq 5 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT100UF500RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 100 -update_freq 500 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT100UF500/run_%i -feature_encoding idf_maze -nonneg 1 -decay_timescale 100 -update_freq 500 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT100UF100_negRS1/run_%i -feature_encoding idf_maze -reward_scale 1 -decay_timescale 100 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT500UF100/run_%i -feature_encoding idf_maze -nonneg 1 -decay_timescale 500 -update_freq 100 %common_param% 
python launch.py -env DeepmindMaze_goal-dif-v0 -curiosity_alg random_reward_mov -log_dir results_DMMaze/ppo_randrewmov_DMMaze-dif-DT500UF100RS1/run_%i -feature_encoding idf_maze -nonneg 1 -reward_scale 1 -decay_timescale 500 -update_freq 100 %common_param% 
)
