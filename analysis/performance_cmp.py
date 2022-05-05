#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from os.path import join
from glob import glob
#%%
rootdir = r"D:\DL_Projects\RL\curiosity_baselines\results_DMMaze"
rootdir0 = r"D:\DL_Projects\RL\curiosity_baselines\results"
#%%
def find_matching_runs(rootdir, expname_pattern, ):
    """
    Find all runs matching the given pattern.
        find_matching_runs(rootdir0, "*DMMaze-mid*")
    """
    expnames = glob(join(rootdir, expname_pattern))
    expnames = [os.path.basename(x) for x in expnames]
    return expnames


def find_matching_runs_re(rootdir, expname_pattern, ):
    """
    Example:
        find_matching_runs_re(rootdir0, ".*DMMaze-mid.*")
    expname_pattern:
    """
    expnames = glob(join(rootdir, "*"))
    expnames = [os.path.basename(x) for x in expnames if os.path.isdir(x)]
    expnames = [x for x in expnames if re.match(expname_pattern, x)]
    return expnames


def plot_all_runs(rootdir, expname, y='EpExtrinsicReward/Average', x='Diagnostics/CumSteps', ax=None, ):
    if ax is None:
        figh, ax = plt.subplots(1, 1, figsize=(6, 5))
    expset = join(rootdir, expname)
    csvs = glob(join(expset, "*", "progress.csv"))
    for csvfn in csvs:
        try:
            df = pd.read_csv(csvfn)
        except pd.errors.EmptyDataError:
            print(f"EmptyDataError: {csvfn}")
            continue
        df.plot(x=x, y=y, # 'EpExtrinsicReward/Average'
                title=f"ExtReward {expname}", ax=ax, label=csvfn.split('\\')[-2])
    return ax


def summarize_traj_all_runs(rootdir, expname,
                            y='EpExtrinsicReward/Average', x='Diagnostics/CumSteps', ax=None, ):
    expset = join(rootdir, expname)
    csvs = glob(join(expset, "*", "progress.csv"))
    vec = []
    for csvfn in csvs:
        try:
            df = pd.read_csv(csvfn)
        except pd.errors.EmptyDataError:
            print(f"EmptyDataError: {csvfn}")
            continue
        xtraj = df[x]
        ytraj = df[y]
        validmsk = ~np.isnan(ytraj)
        area_u_curv = np.trapz(ytraj[validmsk], xtraj[validmsk])
        if np.isnan(area_u_curv):
            return ytraj, xtraj
        vec.append(area_u_curv)
        # df.plot(x=x, y=y,  # 'EpExtrinsicReward/Average'
        #         title=f"ExtReward {expname}", ax=ax, label=csvfn.split('\\')[-2])
    print(f"{expname}: {np.mean(vec):.2e}+-{np.std(vec):.2e} N={len(vec)}")
    return vec
#%%
for dppr in ["dp00", "dp01", "dp05", "dp095"]:
    expname = 'ppo_RNDShfl_DMMaze-dif_' + dppr
    plot_all_runs(rootdir, expname, )
    plt.show()

#%%
for dppr in ["dp00", "dp01", "dp05", ]:
    expname = 'ppo_RND_DMMaze-dif_' + dppr
    plot_all_runs(rootdir, expname, )
    plt.show()
#%%
expname = 'ppo_none_DMMaze-dif'
plot_all_runs(rootdir, expname, )
plt.show()
#%%
sorted(list(df))
#%%
expname = 'ppo_RNDShfl_DMMaze-dif_dp01'
summarize_traj_all_runs(rootdir, expname, )
plot_all_runs(rootdir, expname, )
plt.show()
#%%
expname = 'ppo_none_DMMaze-dif'
vec = summarize_traj_all_runs(rootdir, expname, )
plot_all_runs(rootdir, expname, )
plt.show()
#%%
expname = 'ppo_ICM_DMMaze-dif'
vec = summarize_traj_all_runs(rootdir, expname, )
plot_all_runs(rootdir, expname, )
plt.show()
#%%
expname = 'ppo_RND_DMMaze-dif_dp01'
summarize_traj_all_runs(rootdir, expname, )
plot_all_runs(rootdir, expname, )
plt.show()
#%%
expname = 'ppo_RNDShfl_DMMaze-dif_dp00'
summarize_traj_all_runs(rootdir, expname, )

#%%
for expname in ['ppo_RND_DMMaze-dif_dp01',
                'ppo_RND_DMMaze-dif_dp05',
                'ppo_RND_DMMaze-dif_dp095']:
    summarize_traj_all_runs(rootdir, expname, )
    plot_all_runs(rootdir, expname, )
    plt.show()
#%%
expname = 'ppo_none_DMMaze-dif'
summarize_traj_all_runs(rootdir, expname, )
expname = 'ppo_ICM_DMMaze-dif'
summarize_traj_all_runs(rootdir, expname, )
for expname in ["ppo_RNDShfl_DMMaze-dif_dp00",
                'ppo_RNDShfl_DMMaze-dif_dp01',
                'ppo_RNDShfl_DMMaze-dif_dp05',
                'ppo_RNDShfl_DMMaze-dif_dp095']:
    summarize_traj_all_runs(rootdir, expname, )
for expname in ['ppo_RND_DMMaze-dif_dp01',
                'ppo_RND_DMMaze-dif_dp05',
                'ppo_RND_DMMaze-dif_dp095']:
    summarize_traj_all_runs(rootdir, expname, )
    plot_all_runs(rootdir, expname, )
    plt.show()

#%%
for expname in ['ppo_RND_DMMaze-dif_dp095', 'ppo_RNDShfl_DMMaze-dif_dp095']:
    summarize_traj_all_runs(rootdir, expname, )
    plot_all_runs(rootdir, expname, )
    plt.show()
#%%
expname = 'ppo_RNDShfl_DMMaze-dif_dp095'
summarize_traj_all_runs(rootdir, expname, )
plot_all_runs(rootdir, expname, )
plt.show()
#%%
for expname in ["ppo_randrewmov_DMMaze-dif-DT40UF40",
                "ppo_randrewmov_DMMaze-dif-DT40UF100",
                "ppo_randrewmov_DMMaze-dif-DT100UF100",
                "ppo_randrewmov_DMMaze-dif-DT500UF100",
                "ppo_randrewmov_DMMaze-dif-DT100UF100_neg",]:
    summarize_traj_all_runs(rootdir, expname, y='EpDiscountedExtrinsicReward/Average')
    # plot_all_runs(rootdir, expname, )
    # plt.show()
#%
for expname in ["ppo_randrewmov_DMMaze-dif-DT100UF5",
                "ppo_randrewmov_DMMaze-dif-DT100UF40",
                "ppo_randrewmov_DMMaze-dif-DT100UF100",
                "ppo_randrewmov_DMMaze-dif-DT100UF500",]:
    summarize_traj_all_runs(rootdir, expname, y='EpDiscountedExtrinsicReward/Average')
    # plot_all_runs(rootdir, expname, )
    # plt.show()
#%%
for expname in ["ppo_randrewmov_DMMaze-dif-DT100UF5RS1",
                #"ppo_randrewmov_DMMaze-dif-DT100UF40RS1",
                "ppo_randrewmov_DMMaze-dif-DT40UF100RS1",
                "ppo_randrewmov_DMMaze-dif-DT100UF100RS1",
                "ppo_randrewmov_DMMaze-dif-DT500UF100RS1"]:
    summarize_traj_all_runs(rootdir, expname, )
    plot_all_runs(rootdir, expname, )
    plt.show()
#
#%%
for expname in ['ppo_randDrift_DMMaze-dif_dp01',
                 'ppo_randDrift_DMMaze-dif_dp05',
                 'ppo_randDrift_DMMaze-dif_dp09',
                 'ppo_randdstr_DMMaze-dif_sprs01',
                 'ppo_randdstr_DMMaze-dif_sprs05',
                 'ppo_randdstr_DMMaze-dif_sprs08', ]:
    summarize_traj_all_runs(rootdir, expname, y='EpDiscountedExtrinsicReward/Average')
#%%
for expname in ['ppo_randDrift_DMMaze-dif_dp01',
                 'ppo_randDrift_DMMaze-dif_dp05',
                 'ppo_randDrift_DMMaze-dif_dp09',
                 'ppo_randdstr_DMMaze-dif_sprs01',
                 'ppo_randdstr_DMMaze-dif_sprs05',
                 'ppo_randdstr_DMMaze-dif_sprs08', ]:
    summarize_traj_all_runs(rootdir, expname, y='EpDiscountedExtrinsicReward/Average')
#%%
for expname in ["ppo_randdstr_DMMaze-dif_sprs01",
                "ppo_randdstr_DMMaze-dif_sprs05",
                "ppo_randdstr_DMMaze-dif_sprs08",
                "ppo_randdstr_DMMaze-dif_sprs095",
                "ppo_randdstr_DMMaze-dif_sprs01_neg",
                "ppo_randdstr_DMMaze-dif_sprs05_neg",
                "ppo_randdstr_DMMaze-dif_sprs08_neg",
                "ppo_randdstr_DMMaze-dif_sprs095_neg",]:
    summarize_traj_all_runs(rootdir, expname, y='EpDiscountedExtrinsicReward/Average')
#%%
rootdir0 = 'D:\\DL_Projects\\RL\\curiosity_baselines\\results'
for expname in ["ppo_randdstr_DMMaze-dif_sprs01",
                "ppo_randdstr_DMMaze-dif_sprs05",
                "ppo_randdstr_DMMaze-dif_sprs08",
                "ppo_randdstr_DMMaze-dif_sprs095",
                "ppo_randdstr_DMMaze-dif_sprs01_neg",
                "ppo_randdstr_DMMaze-dif_sprs05_neg",
                "ppo_randdstr_DMMaze-dif_sprs08_neg",
                "ppo_randdstr_DMMaze-dif_sprs095_neg",]:
    summarize_traj_all_runs(rootdir0, expname, y='EpDiscountedExtrinsicReward/Average')
#%%


expnames = find_matching_runs(rootdir0, "*DMMaze-mid*")
for expname in expnames:
    summarize_traj_all_runs(rootdir0, expname, y='EpDiscountedExtrinsicReward/Average')
    #%%
expnames = find_matching_runs(rootdir0, "*DMMaze-ddif*")
for expname in expnames:
    summarize_traj_all_runs(rootdir0, expname, y='EpDiscountedExtrinsicReward/Average')

#%%
from tensorboard.backend.event_processing import event_accumulator
# def plot_all_runs_tb(rootdir, expname, y='EpExtrinsicReward/Average', x='Diagnostics/CumSteps', ax=None, ):
#     if ax is None:
#         figh, ax = plt.subplots(1, 1, figsize=(6, 5))
#     expset = join(rootdir, expname)
#     evtfiles = glob(join(expset, "*", "events.out.tfevents.*"))
#     evtfiles.sort()
#     for evtfile in evtfiles:
#         try:
#             df = pd.read_csv(csvfn)
#         except pd.errors.EmptyDataError:
#             print(f"EmptyDataError: {csvfn}")
#             continue
#         df.plot(x=x, y=y, # 'EpExtrinsicReward/Average'
#                 title=f"ExtReward {expname}", ax=ax, label=csvfn.split('\\')[-2])
#     return ax


def summarize_traj_all_runs_tb(rootdir, expname,
                            y='EpExtrinsicReward/Average', x='Diagnostics/CumSteps', plot=False, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    expset = join(rootdir, expname)
    evtfiles = glob(join(expset, "*", "events.out.tfevents.*"))
    vec = []
    for evtfile in evtfiles:
        ea = event_accumulator.EventAccumulator(evtfile)
        ea.Reload()
        try:
            xtab = pd.DataFrame(ea.Scalars(x))
            ytab = pd.DataFrame(ea.Scalars(y))
            mergetab = pd.merge(xtab, ytab, left_on=["step"], right_on=["step"], how="inner")\
                [["step", "value_x", "value_y"]]
            xtraj = mergetab.value_x.values
            ytraj = mergetab.value_y.values
        except KeyError:
            print(f"KeyError: {evtfile}, {x}, {y} not found")
            continue
        # if len(xtraj) == 0 or len(ytraj) == 0:
        #     print(f"{evtfile}: len(xtraj) == 0 || len(ytraj) == 0")
        #     continue
        # if len(xtraj) > len(ytraj):
        #     xtraj = xtraj[len(xtraj) - len(ytraj):]
        # if len(xtraj) != len(ytraj):
        #     print(f"{evtfile}: len(xtraj) != len(ytraj)")
        #     return xtraj, ytraj, ea.Scalars(x), ea.Scalars(y)
        validmsk = ~np.isnan(ytraj)
        area_u_curv = np.trapz(ytraj[validmsk], xtraj[validmsk])
        vec.append(area_u_curv)
        if plot:
            ax.plot(xtraj, ytraj, label=evtfile.split('\\')[-2], **kwargs)
    print(f"{expname}: {np.mean(vec):.2e}+-{np.std(vec):.2e} N={len(vec)}")
    return vec


def merge_traj_all_runs_tb(rootdir, expname,
                            y='EpExtrinsicReward/Average', x='Diagnostics/CumSteps', plot=False, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    expset = join(rootdir, expname)
    evtfiles = glob(join(expset, "*", "events.out.tfevents.*"))
    mergetab_all = []
    for evtfile in evtfiles:
        ea = event_accumulator.EventAccumulator(evtfile)
        ea.Reload()
        try:
            xtab = pd.DataFrame(ea.Scalars(x))
            ytab = pd.DataFrame(ea.Scalars(y))
            mergetab = pd.merge(xtab, ytab, left_on=["step"], right_on=["step"], how="inner")\
                [["step", "value_x", "value_y"]]
        except KeyError:
            print(f"KeyError: {evtfile}, {x}, {y} not found")
            continue
        mergetab_all.append(mergetab)
    mergetab_all = pd.concat(mergetab_all, axis=0)
    mergetab_all.reset_index(level=0, inplace=True)  # renumber duplicated index
    return mergetab_all
#%%
figh, ax = plt.subplots(1, 1, figsize=(6, 5))
vec0 = summarize_traj_all_runs_tb(rootdir0, "ppo_none_DMMaze-ddif", plot=True,
                         ax=ax, color="k", linestyle="-", linewidth=1, alpha=0.5)
vec1 = summarize_traj_all_runs_tb(rootdir0, "ppo_RND_DMMaze-ddif_dp05", plot=True,
                         ax=ax, color="r", linestyle="-", linewidth=1, alpha=0.5)
vec2 = summarize_traj_all_runs_tb(rootdir0, "ppo_RND_DMMaze-ddif_dp01", plot=True,
                         ax=ax, color="magenta", linestyle="-", linewidth=1, alpha=0.5)
plt.legend(loc='best')
figh.show()
#%%
vec0 = summarize_traj_all_runs_tb(rootdir0, "ppo_none_DMMaze-ddif", plot=False,)
vec1 = summarize_traj_all_runs_tb(rootdir0, "ppo_RND_DMMaze-ddif_dp05", plot=False,)
vec2 = summarize_traj_all_runs_tb(rootdir0, "ppo_RND_DMMaze-ddif_dp01", plot=False,)
#%%
mergetab = merge_traj_all_runs_tb(rootdir0, "ppo_none_DMMaze-ddif",)
mergetab2 = merge_traj_all_runs_tb(rootdir0, "ppo_RND_DMMaze-ddif_dp05",)
mergetab3 = merge_traj_all_runs_tb(rootdir0, "ppo_RND_DMMaze-ddif_dp01",)
#%%
figh, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(x="step", y="value_y", data=mergetab, color="k", estimator=np.mean, ci=50, n_boot=500, label="none")
# sns.lineplot(x="step", y="value_y", data=mergetab2, color="r", estimator=np.mean, ci=50, n_boot=500, label="RND drop0.5")
sns.lineplot(x="step", y="value_y", data=mergetab3, color="r", estimator=np.mean, ci=50, n_boot=500, label="RND drop0.1")
plt.legend(loc='best')
plt.ylabel("Extrinsic Reward")
plt.xlabel("Steps")
plt.title("DMMaze-difficult+")
plt.show()
#%% Difficulty comparison
figdir = r"figures/DMMaze_difficulty_titrate/"
os.makedirs(figdir, exist_ok=True)
envnames = ["DMMaze-mid", "DMMaze-dif", "DMMaze-ddif", "DMMaze-dddif",]
envlabels = ["DMMaze-medium", "DMMaze-difficult", "DMMaze-difficult+", "DMMaze-difficult++", ]
for envnm, envlabel in zip(envnames, envlabels):
    mergetab = merge_traj_all_runs_tb(rootdir0, f"ppo_none_{envnm}", )
    mergetab3 = merge_traj_all_runs_tb(rootdir0, f"ppo_RND_{envnm}_dp01", )
    figh, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.lineplot(x="value_x", y="value_y", data=mergetab, estimator=np.mean, ci=50, n_boot=500,
                 color="k", label="none")
    sns.lineplot(x="value_x", y="value_y", data=mergetab3, estimator=np.mean, ci=50, n_boot=500,
                 color="r", label="RND drop0.1")
    plt.legend(loc='best')
    plt.ylabel("Extrinsic Reward", fontsize=12)
    plt.xlabel("Steps", fontsize=12)
    plt.title(envlabel, fontsize=14)
    plt.tight_layout()
    plt.savefig(join(figdir, f"{envnm}_none-RND.png"))
    plt.savefig(join(figdir, f"{envnm}_none-RND.pdf"))
    plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(x="value_x", y="value_y", data=mergetab, estimator=np.mean, ci=50, n_boot=500,
             color="k", label="none")
sns.lineplot(x="value_x", y="value_y", data=mergetab3, estimator=np.mean, ci=50, n_boot=500,
             color="r", label="RND drop0.1")
plt.legend(loc='best')
plt.ylabel("Extrinsic Reward", fontsize=12)
plt.xlabel("Frames", fontsize=12)
plt.title(envlabel, fontsize=14)
plt.tight_layout()
plt.show()

#%%
for expname in ["ppo_RNDShfl_DMMaze-ddif_dp01",
                "ppo_RNDShfl_DMMaze-ddif_dp05",
                "ppo_RNDShfl_DMMaze-ddif_dp095",]:
    summarize_traj_all_runs_tb(rootdir0, expname, plot=True,)
#%%
for expname in ["ppo_none_DMMaze-ddif",
                "ppo_RND_DMMaze-ddif_dp01",
                "ppo_RND_DMMaze-ddif_dp05",]:
    summarize_traj_all_runs_tb(rootdir0, expname, plot=True,)
#%% dev zone

# subdirs = os.listdir(rootdir)
# expname = subdirs[1]
# expname = 'ppo_RNDShfl_DMMaze-dif_dp095'
# expset = join(rootdir, expname)
# csvs = glob(join(expset, "*", "progress.csv"))
# figh, ax = plt.subplots(1, 1, figsize=(6, 5))
# for csvfn in csvs:
#     df = pd.read_csv(csvfn)
#     df.plot(x='Diagnostics/CumSteps', y='EpNonzeroExtrinsicRewards/Average', # 'EpExtrinsicReward/Average'
#             title=f"ExtReward {expname}", ax=ax, label=csvfn.split('\\')[-2])
# figh.show()


#%%
# expdir = r"D:\DL_Projects\RL\curiosity_baselines\results\ppo_none_DMMaze-ddif\run_0"
# evtfiles = glob(join(expdir, "events.out.tfevents.*"))
# evtfiles.sort()
# for evtfile in evtfiles:
#     ea = event_accumulator.EventAccumulator(evtfile)
#     ea.Reload()
#     ytraj = [s.value for s in ea.Scalars('EpExtrinsicReward/Average')]
#     plt.plot()
#
# plt.show()