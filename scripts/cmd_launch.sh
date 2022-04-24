# Deepmind Maze
python launch.py -alg ppo -curiosity_alg icm -env DeepmindMaze-v0 -serial 0 -log_heatmaps -iterations 100000000 -lstm -num_envs 8 -sample_mode cpu -num_gpus 0 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_deepmind_maze/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg icm -env Deepmind5Room-v0 -serial 0 -log_heatmaps -iterations 100000000 -lstm -num_envs 8 -sample_mode cpu -num_gpus 0 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_deepmind_maze/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 -launch_tmux no


# Mario
python launch.py -alg ppo -curiosity_alg icm -env SuperMarioBros-v0 -iterations 100000000 -lstm -num_envs 4 -sample_mode cpu -serial 0 -num_gpus 1 -num_cpus 4 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_mario_icm/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -mario_level Level1-1 -feature_encoding idf_burda -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 -launch_tmux no


# Atari games
python launch.py -alg ppo -curiosity_alg icm -env breakout -lstm  -iterations 100000000 -num_envs 2 -sample_mode cpu -serial 1 -num_gpus 1 -num_cpus 1 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_breakout/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -max_episode_steps 27000 -feature_encoding idf_burda -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg icm -env breakout -lstm -iterations 100000000 -num_envs 4 -sample_mode cpu -serial 0 -num_gpus 1 -num_cpus 4 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_breakout/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -max_episode_steps 27000 -feature_encoding idf_burda -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 -launch_tmux no

# New curiosity algorithms
python launch.py -alg ppo -curiosity_alg random_reward -log_heatmaps -env Deepmind5RoomBouncing-v0 -iterations 40000 -lstm -num_envs 4 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 4 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_randrew_DeepmindMaze5Room-v0/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze -launch_tmux no

python launch.py -alg ppo -curiosity_alg count -log_heatmaps -env Deepmind5Room-v0 -iterations 2000000 -lstm -num_envs 1 -sample_mode gpu -serial 1 -num_gpus 1 -num_cpus 1 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_count_DM5Room-v0/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze -launch_tmux no

python launch.py -alg ppo -curiosity_alg count -log_heatmaps -env Deepmind5Room-v0 -iterations 2000000 -lstm -num_envs 1 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 1 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_count_DM5Room-v0/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze -launch_tmux no -reward_scale 5.0

# baselines
python launch.py -alg ppo -curiosity_alg icm -env Deepmind5RoomBouncing-v0 -serial 0 -log_heatmaps -iterations 2000000 -lstm -num_envs 4 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 4 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_ICM_DM5RoomBounc/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg icm -env Deepmind5Room-v0 -serial 0 -log_heatmaps -iterations 2000000 -lstm -num_envs 4 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 4 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_ICM_DM5Room/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env Deepmind5Room-v0 -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 6 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DM5Room/run_drop2_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze  -prediction_beta 1.0 -drop_probability 0.2 -launch_tmux no


python launch.py -alg ppo -curiosity_alg rnd -env Deepmind5Room-v0 -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 6 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/pppo_RND_DM5RoomBounc/run_drop2_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze  -prediction_beta 1.0 -drop_probability 0.2 -launch_tmux no

python launch.py -alg ppo -curiosity_alg none -env Deepmind5Room-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 6 -sample_mode gpu -serial 1 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_none_DM5Room/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -launch_tmux no



# ----------------- DeepmindMaze --------------------
# python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 6 -sample_mode gpu -serial 1 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze  -prediction_beta 1.0 -drop_probability 0.2 -launch_tmux no

python launch.py -alg ppo -curiosity_alg icm -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 6 -sample_mode gpu -serial 1 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_ICM_DMMaze/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding idf_maze  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0  -launch_tmux no

python launch.py -alg ppo -curiosity_alg none -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 6 -sample_mode gpu -serial 1 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_none_DMMaze/run_nonefeat_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 6 -sample_mode gpu -serial 1 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze/run_nonefeat_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.2 -launch_tmux no

python launch.py -alg ppo -curiosity_alg count -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 6 -sample_mode gpu -serial 1 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_count_DMMaze/run_nonefeat_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -reward_scale 5.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp00/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp02/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.2 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp05/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.5 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp08/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.8 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp09/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.9 -launch_tmux no
# ----
python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp00/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp02/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.2 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp05/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.5 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp08/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.8 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp09/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.9 -launch_tmux no
# -----
python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp00/run_2 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp02/run_2 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.2 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp05/run_2 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.5 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp08/run_2 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.8 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp09/run_2 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.9 -launch_tmux no

# -----
python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp00/run_3 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp02/run_3 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.2 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp05/run_3 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.5 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp08/run_3 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.8 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp09/run_3 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 0.9 -launch_tmux no

# -----------
python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp10/run_0 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 1.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp10/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 1.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp10/run_2 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 1.0 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0  -serial 0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp10/run_3 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -feature_encoding none  -prediction_beta 1.0 -drop_probability 1.0 -launch_tmux no



DeepmindMaze-v0

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp09_RGB/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type rgb -max_episode_steps 500 -feature_encoding none -prediction_beta 0.9 -launch_tmux no

python launch.py -alg ppo -curiosity_alg rnd -env DeepmindMaze-v0 -log_heatmaps -iterations 1000000 -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -log_dir results/ppo_RND_DMMaze_dp10_RGB/run_1 -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type rgb -max_episode_steps 500 -feature_encoding none -prediction_beta 1.0 -launch_tmux no



# --------- Complex rewarded maze 

set common_param=-alg ppo -iterations 2000000 -log_heatmaps -lstm -num_envs 8 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 8 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no

# Looping through iterations
FOR /L %i IN (0,1,3) DO (

python launch.py -env Maze-Lv0-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_MazeLv0/run_%i -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Maze-Lv0-v0 -curiosity_alg count  -log_dir results/ppo_count_MazeLv0_r10/run_%i -feature_encoding idf_maze %common_param%  -reward_scale 10.0
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_MazeLv0_r10/run_%i -feature_encoding idf_maze %common_param%  
python launch.py -env Maze-Lv0-v0 -curiosity_alg none -log_dir results/ppo_none_MazeLv0/run_%i -feature_encoding none %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp09/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp10/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0

)

python launch.py -env Maze-Lv0-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_MazeLv0/run_0 -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Maze-Lv0-v0 -curiosity_alg count  -log_dir results/ppo_count_MazeLv0_r10/run_0 -feature_encoding idf_maze %common_param%  -reward_scale 10.0
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_MazeLv0_r10/run_0 -feature_encoding idf_maze %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg none -log_dir results/ppo_none_MazeLv0/run_0 -feature_encoding none %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp05/run_0 -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp09/run_0 -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp10/run_0 -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0


python launch.py -env Maze-Lv1-v0 -curiosity_alg none -log_dir results/ppo_none_MazeLv1/run_0 -feature_encoding none %common_param% 
python launch.py -env Maze-Lv1-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv1_dp05/run_0 -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
python launch.py -env Maze-Lv1-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv1_dp09/run_0 -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9
python launch.py -env Maze-Lv1-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv1_dp10/run_0 -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0
python launch.py -env Maze-Lv1-v0 -curiosity_alg count  -log_dir results/ppo_count_MazeLv1_r10/run_0 -feature_encoding idf_maze %common_param%  -reward_scale 10.0
python launch.py -env Maze-Lv1-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_MazeLv1/run_0 -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Maze-Lv1-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_MazeLv1_r10/run_0 -feature_encoding idf_maze %common_param% 

set runId=0
set envId=3
python launch.py -env Maze-Lv%envId%-v0 -curiosity_alg none -log_dir results/ppo_none_MazeLv%envId%/run_%runId% -feature_encoding none %common_param% 
python launch.py -env Maze-Lv%envId%-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv%envId%_dp05/run_%runId% -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
python launch.py -env Maze-Lv%envId%-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv%envId%_dp09/run_%runId% -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9
python launch.py -env Maze-Lv%envId%-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv%envId%_dp10/run_%runId% -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0
python launch.py -env Maze-Lv%envId%-v0 -curiosity_alg count  -log_dir results/ppo_count_MazeLv%envId%_r10/run_%runId% -feature_encoding idf_maze %common_param%  -reward_scale 10.0
python launch.py -env Maze-Lv%envId%-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_MazeLv%envId%/run_%runId% -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Maze-Lv%envId%-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_MazeLv%envId%_r10/run_%runId% -feature_encoding idf_maze %common_param% 


python launch.py -env Maze-Lv3-v0 -curiosity_alg none -log_dir results/ppo_none_MazeLv3/run_0 -feature_encoding none -alg ppo -iterations 2000000 -log_heatmaps -lstm -num_envs 1 -sample_mode gpu -serial 1 -num_gpus 1 -num_cpus 1 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no

# ---- Simple Maze with goal ---- 
set common_param=-alg ppo -iterations 1000000 -log_heatmaps -lstm -num_envs 32 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 32 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no


set envId=0
set runId=0
python launch.py -env Deepmind5Room_goal-Lv%envId%-v0 -curiosity_alg none -log_dir results/ppo_none_DM5roomGoal_Lv%envId%/run_%runId% -feature_encoding none %common_param% 
python launch.py -env Deepmind5Room_goal-Lv%envId%-v0 -curiosity_alg count  -log_dir results/ppo_count_DM5roomGoal_Lv%envId%_r10/run_%runId% -feature_encoding idf_maze %common_param%  -reward_scale 10.0
python launch.py -env Deepmind5Room_goal-Lv%envId%-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_DM5roomGoal_Lv%envId%/run_%runId% -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Deepmind5Room_goal-Lv%envId%-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_DM5roomGoal_Lv%envId%_dp01/run_%runId% -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1
python launch.py -env Deepmind5Room_goal-Lv%envId%-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_DM5roomGoal_Lv%envId%_dp05/run_%runId% -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
python launch.py -env Deepmind5Room_goal-Lv%envId%-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_DM5roomGoal_Lv%envId%_dp09/run_%runId% -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9
python launch.py -env Deepmind5Room_goal-Lv%envId%-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_DM5roomGoal_Lv%envId%_dp10/run_%runId% -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0
python launch.py -env Deepmind5Room_goal-Lv%envId%-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_DM5roomGoal_Lv%envId%_r10/run_%runId% -feature_encoding idf_maze %common_param% 


# ---- Simple Maze with randomized goal ---- 
FOR /L %i IN (0,1,3) DO (
set runId=%i
echo %i
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg none -log_dir results/ppo_none_DM5roomRandGoal/run_%i -feature_encoding none %common_param% 
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg count  -log_dir results/ppo_count_DM5roomRandGoal_r10/run_%i -feature_encoding idf_maze %common_param%  -reward_scale 10.0
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_DM5roomRandGoal/run_%i -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_DM5roomRandGoal_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_DM5roomRandGoal_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_DM5roomRandGoal_dp09/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_DM5roomRandGoal_dp10/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_DM5roomRandGoal_r10/run_%i -feature_encoding idf_maze %common_param% 
)

FOR /L %i IN (0,1,10) DO (
python -c "print(%i)"
)
# set runId=%i
echo %i
set common_param=-alg ppo -iterations 1000000 -log_heatmaps -lstm -num_envs 6 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no



# python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_DM5roomRandGoal_tmp/run_%i -feature_encoding idf_maze -use_distr -zero_prob 0.5 -nonneg 1 %common_param% 
# python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_DM5roomRandGoal_tmp/run_%i -feature_encoding idf_maze -use_distr -zero_prob 0.5 -nonneg 1 %common_param% 

FOR /L %i IN (0,1,1) DO (
python launch.py -env Deepmind5Room_goal-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randdstr_DM5roomGoal_sprs02/run_%i -feature_encoding idf_maze -use_distr -zero_prob 0.2 -nonneg 1 %common_param% 
python launch.py -env Deepmind5Room_goal-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randdstr_DM5roomGoal_sprs05/run_%i -feature_encoding idf_maze -use_distr -zero_prob 0.5 -nonneg 1 %common_param% 
python launch.py -env Deepmind5Room_goal-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randdstr_DM5roomGoal_sprs08/run_%i -feature_encoding idf_maze -use_distr -zero_prob 0.8 -nonneg 1 %common_param% 
python launch.py -env Deepmind5Room_goal-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randdstr_DM5roomGoal_sprs10/run_%i -feature_encoding idf_maze -use_distr -zero_prob 1.0 -nonneg 1 %common_param% 
python launch.py -env Deepmind5Room_goal-Lv0-v0 -curiosity_alg none -log_dir results/ppo_none_DM5roomGoal/run_%i -feature_encoding none %common_param% 
python launch.py -env Deepmind5Room_goal-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_DM5roomGoal_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1
)

python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg random_reward  -log_dir results/ppo_randdstr_DM5roomRandGoal_sprs02/run_%i -feature_encoding idf_maze -use_distr -zero_prob 0.2 -nonneg 1 %common_param% 
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg random_reward  -log_dir results/ppo_randdstr_DM5roomRandGoal_sprs05/run_%i -feature_encoding idf_maze -use_distr -zero_prob 0.5 -nonneg 1 %common_param% 
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg random_reward  -log_dir results/ppo_randdstr_DM5roomRandGoal_sprs08/run_%i -feature_encoding idf_maze -use_distr -zero_prob 0.8 -nonneg 1 %common_param% 
python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg random_reward  -log_dir results/ppo_randdstr_DM5roomRandGoal_sprs10/run_%i -feature_encoding idf_maze -use_distr -zero_prob 1.0 -nonneg 1 %common_param% 



# ---------------------- Off policy learning ------------------------
set common_param=-alg dqn -iterations 1000000 -log_heatmaps -lstm -num_envs 6 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 6 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no

python launch.py -env Deepmind5Room_randomgoal-v0 -curiosity_alg random_reward  -log_dir results/dqn_randrew_DM5roomRandGoal/run_%i -feature_encoding idf_maze %common_param% 


