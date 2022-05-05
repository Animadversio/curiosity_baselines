
set common_param=-alg ppo -iterations 2500000 -log_heatmaps -lstm -num_envs 48 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 48 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no

python launch.py -env Maze-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_MazeLv0/run_3 -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Maze-v0 -curiosity_alg rnd -no_error -log_dir results/ppo_randDrift_MazeLv0_dp01/run_3 -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1 
python launch.py -env Maze-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp01/run_3 -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1



set offpolicy_param=-alg r2d1 -iterations 2500000 -log_heatmaps -lstm -num_envs 48 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 48 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no


python launch.py -env Maze-v0 -curiosity_alg none  -log_dir results/r2d1_none_MazeLv0_dp01/run_0 -feature_encoding none %offpolicy_param% -prediction_beta 1.0 -drop_probability 0.1