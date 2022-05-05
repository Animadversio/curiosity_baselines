

set common_param=-alg ppo -iterations 2500000 -log_heatmaps -lstm -num_envs 48 -sample_mode gpu -serial 0 -num_gpus 1 -num_cpus 48 -eval_envs 0 -eval_max_steps 51000 -eval_max_traj 50 -timestep_limit 20 -log_interval 10000 -record_freq 0 -pretrain None -discount 0.99 -lr 0.0001 -v_loss_coeff 1.0 -entropy_loss_coeff 0.001 -grad_norm_bound 1.0 -gae_lambda 0.95 -minibatches 1 -epochs 3 -ratio_clip 0.1 -kernel_mu 0.0 -kernel_sigma 0.001 -obs_type mask -max_episode_steps 500 -launch_tmux no

FOR /L %i IN (0,1,2) DO (
set runId=%i
echo %i
python launch.py -env Maze-Lv0-v0 -curiosity_alg none -log_dir results/ppo_none_MazeLv1/run_%i -feature_encoding none %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward -use_distr  -log_dir results/ppo_randdstr_MazeLv1_sprs02/run_%i -feature_encoding idf_maze -zero_prob 0.2 -nonneg 1 -reward_scale 0.002 %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward -use_distr  -log_dir results/ppo_randdstr_MazeLv1_sprs05/run_%i -feature_encoding idf_maze -zero_prob 0.5 -nonneg 1 -reward_scale 0.002 %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward -use_distr  -log_dir results/ppo_randdstr_MazeLv1_sprs08/run_%i -feature_encoding idf_maze -zero_prob 0.8 -nonneg 1 -reward_scale 0.002 %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward -use_distr  -log_dir results/ppo_randdstr_MazeLv1_sprs10/run_%i -feature_encoding idf_maze -zero_prob 1.0 -nonneg 1 -reward_scale 0.002 %common_param% 

python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd -no_error -log_dir results/ppo_randDrift_MazeLv1_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1 
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd -no_error -log_dir results/ppo_randDrift_MazeLv1_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5 
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd -no_error -log_dir results/ppo_randDrift_MazeLv1_dp10/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0 

python launch.py -env Maze-Lv0-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_MazeLv1/run_%i -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Maze-Lv0-v0 -curiosity_alg ndigo -log_dir results/ppo_ndigo_MazeLv1/run_%i -feature_encoding idf_maze %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv1_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv1_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
)

FOR /L %i IN (0,1,3) DO (

python launch.py -env Maze-Lv0-v0 -curiosity_alg none -log_dir results/ppo_none_MazeLv0/run_%i -feature_encoding none %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp09/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp10/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0

python launch.py -env Maze-Lv0-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_MazeLv0/run_%i -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Maze-Lv0-v0 -curiosity_alg count  -log_dir results/ppo_count_MazeLv0_r10/run_%i -feature_encoding idf_maze %common_param%  -reward_scale 10.0
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_MazeLv0_r10/run_%i -feature_encoding idf_maze %common_param%  

)

python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd -no_error -log_dir results/ppo_randDrift_MazeLv0_dp09/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9 

FOR /L %i IN (0,1,3) DO (
set runId=%i
echo %i
python launch.py -env Maze-Lv0-v0 -curiosity_alg none -log_dir results/ppo_none_MazeLv0/run_%i -feature_encoding none %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg count  -log_dir results/ppo_count_MazeLv0_r10/run_%i -feature_encoding idf_maze %common_param%  -reward_scale 10.0
python launch.py -env Maze-Lv0-v0 -curiosity_alg icm  -log_dir results/ppo_ICM_MazeLv0/run_%i -feature_encoding idf_maze %common_param%  -forward_loss_wt 0.2 -prediction_beta 1.0 -prediction_lr_scale 10.0 
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp01/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.1
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp05/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.5
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp09/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 0.9
python launch.py -env Maze-Lv0-v0 -curiosity_alg rnd  -log_dir results/ppo_RND_MazeLv0_dp10/run_%i -feature_encoding none %common_param% -prediction_beta 1.0 -drop_probability 1.0
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_MazeLv0/run_%i -feature_encoding idf_maze %common_param% -reward_scale 1.0
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_MazeLv0_nonneg/run_%i -feature_encoding idf_maze %common_param% -nonneg 1 -reward_scale 1.0
python launch.py -env Maze-Lv0-v0 -curiosity_alg random_reward  -log_dir results/ppo_randrew_MazeLv0_nonneg_r5/run_%i -feature_encoding idf_maze %common_param% -nonneg 1 -reward_scale 5.0
python launch.py -env Maze-Lv0-v0 -curiosity_alg ndigo -log_dir results/ppo_ndigo_MazeLv0/run_%i -feature_encoding idf_maze %common_param% 
python launch.py -env Maze-Lv0-v0 -curiosity_alg disagreement -log_dir results/ppo_disagr_MazeLv0/run_%i -feature_encoding idf_maze %common_param% 
)

