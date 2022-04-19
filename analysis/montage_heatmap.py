import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
# from skimage.transform import rescale, resize, downscale_local_mean
from analysis.plot_utils import make_grid_np, upscale_pix2square, summary_montage_heatmaps

def summary_montage_heatmaps(heatmap_dir, loginterval=3, upscale=12, nrow=10):
    heatmap_col = []
    for i in range(loginterval, 2000, loginterval):
        if os.path.exists(join(heatmap_dir, f"{i}.png")):
            img = plt.imread(join(heatmap_dir, f"{i}.png"))
            heatmap_col.append(img)
        else:
            break
    print("Collect %d heatmaps"%len(heatmap_col))
    mtg = make_grid_np(heatmap_col, nrow=10)
    plt.imsave(join(heatmap_dir, "summary_heatmap.png"), upscale_pix2square(mtg, 12, ),)
    return mtg

result_root = r"E:\DL_Projects\RL\curiosity_baselines\results"
loginterval = 3
#%%
# Collect all heatmaps in the folder
# D:\DL_Projects\RL\curiosity_baselines
# heatmap_dir = r"results\ppo_deepmind_maze\run_0\heatmaps"
heatmap_dir = join(result_root, r"ppo_ICM_DM5RoomBouncing\run_0\heatmaps")
heatmap_dir = join(result_root, r"ppo_ICM_DM5Room\run_0\heatmaps")
heatmap_dir = join(result_root, r"ppo_count_DM5Room-v0\run_0\heatmaps")
mtg = summary_montage_heatmaps(heatmap_dir, loginterval=3, upscale=12, nrow=10)
plt.imshow(mtg)
plt.show()
#%%
from rlpyt.envs.mazeworld.mazeworld import DeepmindMazeWorld_maze
from rlpyt.envs.pycolab.pycolab.examples.deepmind_maze import MAZES_ART
charmap = np.array([[*s] for s in MAZES_ART[0]])
visitable_pos_num = np.count_nonzero(charmap == " ") + \
                np.count_nonzero(charmap == "P")
#%%
itr = 270
heatmap_dir = join(result_root, r"ppo_RND_DMMaze\run_nonefeat_0\heatmaps")
visitmap_col = []
iter_col = []
for itr in range(loginterval, 2000, loginterval):
    if os.path.exists(join(heatmap_dir, f"{itr}.npy")):
        visitmap = np.load(join(heatmap_dir, f"{itr}.npy"))
        visitmap_col.append(visitmap)
        iter_col.append(itr)
    else:
        break
print("Collect %d heatmaps" % len(visitmap_col))
visit_tsr = np.array(visitmap_col)
iter_vec = np.array(iter_col)
visit_states_num = np.count_nonzero(visit_tsr, axis=(1, 2))
#%%
# plt.plot(visit_states_num, label="visit states num")
plt.plot(iter_vec, visit_states_num / visitable_pos_num, )
plt.xlabel("Iteration")
plt.ylabel("visit states fraction")
plt.show()
#%%
import pandas as pd
import seaborn as sns
from glob import glob
def get_coverage_curve(heatmap_dir, loginterval=3, total_num=visitable_pos_num):
    visitmap_col = []
    iter_col = []
    for itr in range(loginterval, 2000, loginterval):
        if os.path.exists(join(heatmap_dir, f"{itr}.npy")):
            visitmap = np.load(join(heatmap_dir, f"{itr}.npy"))
            visitmap_col.append(visitmap)
            iter_col.append(itr)
        else:
            break
    if len(visitmap_col) == 0:
        return np.array([]), np.array([]), pd.DataFrame({"iter": [], "visit_states_num": [], "run_name": []})
    else:
        print("Collect %d heatmaps" % len(visitmap_col))
        visit_tsr = np.array(visitmap_col)
        iter_vec = np.array(iter_col)
        visit_states_num = np.count_nonzero(visit_tsr, axis=(1, 2))
        #
        # plt.plot(visit_states_num, label="visit states num")
        runname = heatmap_dir[len(result_root)+1:-len("heatmaps")-1]
        plt.plot(iter_vec, visit_states_num, )
        plt.xlabel("Iteration")
        plt.ylabel("visit states")
        plt.title(f"coverage curve of {runname}\n total{total_num}")
        plt.savefig(join(heatmap_dir, "coverage_curve.png"))
        plt.show()
        df = pd.DataFrame({"iter": iter_vec, "visit_states_num": visit_states_num, "run_name": runname})
        return iter_vec, visit_states_num, df


def sweep_folders(subdirs=None, result_root=result_root, loginterval=3):
    if subdirs is None:
        heatmap_dirs = glob(join(result_root, "*\\*run*\\heatmaps"))
    elif isinstance(subdirs, str):
        heatmap_dirs = glob(join(result_root, subdirs+"\\heatmaps"))
    else:
        heatmap_dirs = [join(result_root, subdir) for subdir in subdirs]
    df_col = []
    for heatmap_dir in heatmap_dirs:
        _, _, df = get_coverage_curve(heatmap_dir, loginterval=loginterval)
        df_col.append(df)

    df_all = pd.concat(df_col)
    df_all.reset_index(inplace=True)
    return df_all
#%%
heatmap_dirs = [join(result_root, r"ppo_ICM_DM5RoomBouncing\run_0\heatmaps"),
                    join(result_root, r"ppo_ICM_DM5Room\run_0\heatmaps"),
                    join(result_root, r"ppo_count_DM5Room-v0\run_0\heatmaps"),
                    join(result_root, r"ppo_RND_DMMaze\run_nonefeat_0\heatmaps"),
                    ]
df_col = []
for heatmap_dir in heatmap_dirs:
    _, _, df = get_coverage_curve(heatmap_dir, loginterval=loginterval)
    df_col.append(df)
df_all = pd.concat(df_col)
df_all.reset_index(inplace=True)
#%%
# df_all.groupby("run_name").plot(x="iter", y="visit_states_num", kind="line", legend=True)
sns.lineplot(x="iter", y="visit_states_num", hue="run_name",
             data=df_all, alpha=0.7, legend=True)
plt.show()
#%%
df_all = sweep_folders(subdirs=None, result_root=result_root, loginterval=3)
#%%
import re
subdirs = [r"ppo_count_DMMaze\run_nonefeat_0",
           r"ppo_RND_DMMaze\run_nonefeat_0",
           r"ppo_RND_DMMaze\run_0",
           r"ppo_none_DMMaze\run_nonefeat_0",
           r"ppo_ICM_DMMaze\run_0",]

# mask = np.logical_or.reduce([df_all.run_name.str.contains(i, regex=False, case=False) for i in subdirs])
mask = df_all.run_name.str.contains("|".join(re.escape(s) for s in subdirs))
plt.figure(figsize=(10, 6))
sns.lineplot(x="iter", y="visit_states_num", hue="run_name",
             data=df_all[mask], alpha=0.7, legend=True)
plt.show()
#%%
subdirs = ["ppo_RND_DM5Room\\run_drop2_0",
           "ppo_ICM_DM5Room\\run_0",
           "ppo_count_DM5Room-v0\\run_0",
           "ppo_randrew_DeepmindMaze5Room-v0\\run_0",]

mask = df_all.run_name.str.contains("|".join(re.escape(s) for s in subdirs))
plt.figure(figsize=(10, 6))
sns.lineplot(x="iter", y="visit_states_num", hue="run_name",
             data=df_all[mask], alpha=0.7, legend=True)
plt.xlim([-50, 450])
plt.show()
#%%
subdirs = ["ppo_RND_DM5Room\\run_0",
           "ppo_RND_DM5RoomBounc\\run_0",
           "ppo_RND_DM5Room\\run_drop2_0",]

mask = df_all.run_name.str.contains("|".join(re.escape(s) for s in subdirs))

plt.figure(figsize=(10, 6))
sns.lineplot(x="iter", y="visit_states_num", hue="run_name",
             data=df_all[mask], alpha=0.5, lw=2, legend=True)
plt.xlim([-25, 425])
plt.show()
#%%
subdirs = ["ppo_ICM_DM5Room\\run_0",
           "ppo_ICM_DM5RoomBouncing\\run_0",
           "ppo_none_DM5Room\\run_0",]

mask = df_all.run_name.str.contains("|".join(re.escape(s) for s in subdirs))
plt.figure(figsize=(10, 6))
sns.lineplot(x="iter", y="visit_states_num", hue="run_name",
             data=df_all[mask], alpha=0.5, lw=2, legend=True)
plt.xlim([-25, 425])
plt.show()
#%%
drop_rate_df_all = sweep_folders(subdirs="ppo_RND_DMMaze_dp*\\run_*", result_root=result_root, loginterval=3)
drop_rate_df_all["expname"] = drop_rate_df_all.run_name.str.split("\\").apply(lambda l: l[0])
drop_rate_df_all["run"] = drop_rate_df_all.run_name.str.split("\\").apply(lambda l: l[1])
#%%
# mask = df_all.run_name.str.contains("|".join(re.escape(s) for s in subdirs))
plt.figure(figsize=(10, 6))
sns.lineplot(x="iter", y="visit_states_num", hue="expname",
             data=drop_rate_df_all, alpha=0.3, lw=2, legend=True)
plt.xlim([-25, 275])
plt.savefig(join(outdir, "visit_states_num_cmp.png"))
plt.show()
#%%



#%%

def sweep_folders_csv(subdirs=None, result_root=result_root, loginterval=3):
    if subdirs is None:
        expdirs = glob(join(result_root, "*\\*run*"))
    elif isinstance(subdirs, str):
        expdirs = glob(join(result_root, subdirs))
    else:
        expdirs = [join(result_root, subdir) for subdir in subdirs]

    df_col = []
    for expdir in expdirs:
        df = pd.read_csv(join(expdir, "progress.csv"))
        df["run_name"] = "\\".join(expdir.split("\\")[-2:])
        df["expname"] = expdir.split("\\")[-2]
        df["run"] = expdir.split("\\")[-1]
        # _, _, df = get_coverage_curve(heatmap_dir, loginterval=loginterval)
        df_col.append(df)

    df_all = pd.concat(df_col)
    df_all.reset_index(inplace=True)
    return df_all

progress_df_all = sweep_folders_csv(subdirs="ppo_RND_DMMaze_dp*\\run_*", result_root=result_root, )
#%%
progress_df_all[['first_visit_%s/Average'%(c) for c in ["a", "b", "c", "d", "e"]]]
#%%
plt.figure(figsize=(10, 6))
sns.lineplot(x='Diagnostics/Iteration',
             y='first_visit_a/Average', hue="expname",
             data=progress_df_all, alpha=0.3, lw=2, legend=True)
plt.xlim([-25, 275])
plt.show()
#%%
plt.figure(figsize=(10, 6))
sns.lineplot(x='Diagnostics/Iteration',
             y='first_visit_a/Average', hue="run_name",
             data=progress_df_all, alpha=0.3, lw=2, legend=True)
plt.xlim([-25, 275])
plt.show()
#%%
outdir = r"E:\DL_Projects\RL\curiosity_baselines\summary"
for c in ["a", "b", "c", "d", "e"]:
    figh, axs = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Diagnostics/Iteration',
                 y='visit_freq_%s/Average'%c, hue="expname",
                data=progress_df_all, alpha=0.8, lw=2, legend=True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    axs.relim()
    plt.tight_layout()
    figh.savefig(join(outdir, "visit_freq_%s_cmp.png"%c))
    plt.show()
    #%
    figh, axs = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Diagnostics/Iteration',
                 y='visit_freq_%s/Average'%c, hue="run_name",
                data=progress_df_all, alpha=0.8, lw=2, legend=True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    axs.relim()
    plt.tight_layout()
    figh.savefig(join(outdir, "visit_freq_%s_allrun_cmp.png"%c))
    plt.show()

    figh, axs = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Diagnostics/Iteration',
                 y='first_visit_%s/Average'%c, hue="expname",
                data=progress_df_all, alpha=0.8, lw=2, legend=True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    axs.relim()
    plt.tight_layout()
    figh.savefig(join(outdir, "first_visit_%s_cmp.png"%c))
    plt.show()

