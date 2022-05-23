#%%
import os
import re
from glob import glob
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
#%%
rootdir = r"D:\DL_Projects\RL\curiosity_baselines\results_DMMaze"
rootdir0 = r"D:\DL_Projects\RL\curiosity_baselines\results"
figroot = r"D:\DL_Projects\RL\curiosity_baselines\figures"
figroot = r"C:\Users\ponce\OneDrive - Harvard University\MIT6.484SensorimotorLearning\ProjectReport\figures"
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


def plot_all_runs(rootdir, expname, y='EpExtrinsicReward/Average', x='Diagnostics/CumSteps',
                  ax=None, **kwargs):
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
                title=f"ExtReward {expname}", ax=ax, label=csvfn.split('\\')[-2], **kwargs)
    return ax


def summarize_traj_all_runs(rootdir, expname,
                            y='EpExtrinsicReward/Average', x='Diagnostics/CumSteps', ax=None):
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


def merge_traj_all_runs(rootdir, expname, y='EpExtrinsicReward/Average', x='Diagnostics/CumSteps',
                            plot=False, ax=None, **kwargs):
    expset = join(rootdir, expname)
    csvs = glob(join(expset, "*", "progress.csv"))
    mergetab_all = []
    for csvfn in csvs:
        try:
            df = pd.read_csv(csvfn)
        except pd.errors.EmptyDataError:
            print(f"EmptyDataError: {csvfn}")
            continue
        # xtraj = df[x]
        # ytraj = df[y]
        # steps = df['Diagnostics/Iteration']
        # mergetab = pd.DataFrame({"step": steps, "value_x": xtraj, "value_y": ytraj})
        mergetab = df[['Diagnostics/Iteration', x, y]]
        mergetab = mergetab.rename(columns={'Diagnostics/Iteration': "step", x: "value_x", y: "value_y"})
        # validmsk = ~np.isnan(ytraj)
        # return df
        # ea = event_accumulator.EventAccumulator(evtfile)
        # ea.Reload()
        # try:
        #     xtab = pd.DataFrame(ea.Scalars(x))
        #     ytab = pd.DataFrame(ea.Scalars(y))
        #     mergetab = pd.merge(xtab, ytab, left_on=["step"], right_on=["step"], how="inner") \
        #         [["step", "value_x", "value_y"]]
        # except KeyError:
        #     print(f"KeyError: {evtfile}, {x}, {y} not found")
        #     continue
        mergetab_all.append(mergetab)
    mergetab_all = pd.concat(mergetab_all, axis=0)
    mergetab_all.reset_index(level=0, inplace=True)  # renumber duplicated index
    return mergetab_all


# statistical comparison of AUC
def compare_conditions(rootdir, expnames, read_tensorboard=True, stat_name="AUC"):
    """
    Compare conditions of different runs, generate a merged table useful for bar plot and stats
    """
    df = pd.DataFrame()
    for expname in expnames:
        if read_tensorboard:
            vec = summarize_traj_all_runs_tb(rootdir, expname,)
        else:
            vec = summarize_traj_all_runs(rootdir, expname,)
        df_part = pd.DataFrame({stat_name: vec, "condition": expname})
        df = pd.concat([df, df_part], axis=0)
    return df
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

#%% Figure: Difficulty comparison
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
#%%
summarize_traj_all_runs(rootdir, "ppo_RND_DMMaze-dif_dp095",)
plot_all_runs(rootdir, "ppo_RND_DMMaze-dif_dp095",)
plt.show()
#%%
find_matching_runs(rootdir, "*randdstr*",)
#%% Figure: Experiment 2 - random reward from fixed distribution.
for expname in ['ppo_randdstr_DMMaze-dif_sprs01',
                'ppo_randdstr_DMMaze-dif_sprs01_neg',
                'ppo_randdstr_DMMaze-dif_sprs05',
                'ppo_randdstr_DMMaze-dif_sprs05_neg',
                'ppo_randdstr_DMMaze-dif_sprs08',
                'ppo_randdstr_DMMaze-dif_sprs08_neg',
                'ppo_randdstr_DMMaze-dif_sprs095',
                'ppo_randdstr_DMMaze-dif_sprs095_neg']:
    summarize_traj_all_runs_tb(rootdir, expname,)
#%%
#%% Experiment 1: naive fixed random reward.
expnames = ['ppo_none_DMMaze-dif',
            "ppo_RND_DMMaze-dif_dp01",
            "ppo_RND_DMMaze-dif_dp10",
            "ppo_randrew_DMMaze-dif",
            "ppo_randrew_DMMaze-dif_nonneg",
            "ppo_randrew_DMMaze-dif_nonneg_r5",]
stat_df = compare_conditions(rootdir, expnames, read_tensorboard=False, )
stat_df["shortname"] = stat_df.condition.apply(lambda x: x.replace("_DMMaze-dif","").removeprefix("ppo_"))
# stat_df["dynam"] = stat_df.shortname.apply(lambda x: x.rstrip("RS1"))
# stat_df["rewardscale"] = stat_df.shortname.apply(lambda x: 1 if x.endswith("RS1") else 0.05)
bsl_stat_df = compare_conditions(rootdir, find_matching_runs(rootdir, "*none*dif",), read_tensorboard=False, )
#%%
figh, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.barplot(x="shortname", y="AUC", data=stat_df, ci=50, estimator=np.mean, )
sns.stripplot(x="shortname", y="AUC", data=stat_df, dodge=True,
              color="r", jitter=0.2, edgecolor="black", )
plt.axhline(bsl_stat_df.AUC.mean(), color="k", linestyle="-", label="PPO baseline")
plt.axhline(bsl_stat_df.AUC.mean() - bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.axhline(bsl_stat_df.AUC.mean() + bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.xticks(rotation=30)
plt.ylabel("AUC", fontsize=12)
plt.xlabel("Condition: Fixed reward function", fontsize=12)
plt.title("Fixed reward function", fontsize=14)
plt.legend(loc='best')  # [None,None,"PPO baseline", "RND shuffled", "RND"],
plt.tight_layout()
plt.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_bar2.pdf"), dpi=300)
plt.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_bar2.png"), dpi=300)
plt.show()
#%% Plot trajectories for each condition.

def load_merge_plot_traj(rootdir, expname_label_dict, read_tensorboard=False,
                         y='EpExtrinsicReward/Average', x='Diagnostics/CumSteps'):
    """Load and merge trajectories for each condition."""
    for runname, label in expname_label_dict.items():
        tab = merge_traj_all_runs_tb(rootdir, runname, y=y, x=x) if read_tensorboard else \
            merge_traj_all_runs(rootdir, runname, y=y, x=x)
        sns.lineplot(x="value_x", y="value_y", data=tab, label=label, estimator=np.mean, ci=50, n_boot=500)

    plt.legend(loc='best')

runlabeldict = {"ppo_none_DMMaze-dif": 'none',
                "ppo_RND_DMMaze-dif_dp01": 'RND dp0.1',
                "ppo_RND_DMMaze-dif_dp10": 'RND dp1.0',
                "ppo_randrew_DMMaze-dif": 'rand reward',
                "ppo_randrew_DMMaze-dif_nonneg": 'rand reward nonneg',
                "ppo_randrew_DMMaze-dif_nonneg_r5": 'rand reward nonneg x5',
                }
figh, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
load_merge_plot_traj(rootdir, runlabeldict, read_tensorboard=False, y='EpExtrinsicReward/Average')
plt.ylabel("Extrinsic Reward", fontsize=14)
plt.xlabel("Steps", fontsize=14)
plt.title("DMMaze-difficult: Fixed reward function", fontsize=16)
plt.tight_layout()
figh.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_traj.pdf"), dpi=300)
figh.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_traj.png"), dpi=300)
plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
load_merge_plot_traj(rootdir, runlabeldict, read_tensorboard=False, y='intrinsic_rewards/Average')
plt.ylabel("Intrinsic Reward", fontsize=14)
plt.xlabel("Steps", fontsize=14)
plt.title("DMMaze-difficult: Fixed reward function", fontsize=16)
ax.set_ylim([-0.04,0.07])
plt.tight_layout()
figh.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_intrreward_traj.pdf"), dpi=300)
figh.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_intrreward_traj.png"), dpi=300)
plt.show()
#%% RND baseline
find_matching_runs(rootdir, "ppo_RND_DMMaze-dif*")
RNDlabeldict = {
                "ppo_RND_DMMaze-dif_dp00": 'RND dp0.0',
                "ppo_RND_DMMaze-dif_dp01": 'RND dp0.1',
                "ppo_RND_DMMaze-dif_dp05": 'RND dp0.5',
                'ppo_RND_DMMaze-dif_dp095': 'RND dp0.95',
                "ppo_RND_DMMaze-dif_dp10": 'RND dp1.0',
                }
figh, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
load_merge_plot_traj(rootdir, RNDlabeldict, read_tensorboard=False, y='intrinsic_rewards/Average')
plt.ylabel("Intrinsic Reward", fontsize=14)
plt.xlabel("Steps", fontsize=14)
plt.title("DMMaze-difficult: RND baseline", fontsize=16)
# ax.set_ylim([-0.04,0.07])
plt.tight_layout()
figh.savefig(join(figroot, "fix_distribution_reward", "RND_baseline_cmp_intrreward_traj.pdf"), dpi=300)
figh.savefig(join(figroot, "fix_distribution_reward", "RND_baseline_cmp_intrreward_traj.png"), dpi=300)
plt.show()
#%% Experiment 2: Reward function from fixed distribution.
stat_df = compare_conditions(rootdir, find_matching_runs(rootdir, "*randdstr*",), read_tensorboard=False, )
stat_df["nonneg"] = ~stat_df.condition.str.contains("_neg")
stat_df["shortname"] = stat_df.condition.apply(lambda x: x[len("ppo_randdstr_DMMaze-dif_"):])
bsl_stat_df = compare_conditions(rootdir, find_matching_runs(rootdir, "*none*dif",), read_tensorboard=False, )
#%%
figh, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.barplot(x="shortname", y="AUC", data=stat_df, hue="nonneg", ci=50, estimator=np.mean, )
sns.stripplot(x="shortname", y="AUC", hue="nonneg", data=stat_df, dodge=True,
              color="r", jitter=0.2, edgecolor="black", )
plt.axhline(bsl_stat_df.AUC.mean(), color="k", linestyle="-", label="PPO baseline")
plt.axhline(bsl_stat_df.AUC.mean() - bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.axhline(bsl_stat_df.AUC.mean() + bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.legend(loc='best')
plt.xticks(rotation=30)
plt.ylabel("AUC", fontsize=12)
plt.xlabel("Condition: Sparseness ratio (sprs); nonnegativity", fontsize=12)
plt.title("Stationary random distribution", fontsize=14)
plt.tight_layout()
plt.savefig(join(figroot, "fix_distribution_reward", "fix_distribution_reward_cmp_bar.pdf"), dpi=300)
plt.savefig(join(figroot, "fix_distribution_reward", "fix_distribution_reward_cmp_bar.png"), dpi=300)
plt.show()


#%% Experiment 3: Random moving reward
stat_df = compare_conditions(rootdir, find_matching_runs(rootdir, "*randrewmov_DMMaze-dif*",), read_tensorboard=False, )
stat_df["shortname"] = stat_df.condition.apply(lambda x: x[len("ppo_randrewmov_DMMaze-dif-"):])
stat_df["dynam"] = stat_df.shortname.apply(lambda x: x.rstrip("RS1"))
stat_df["rewardscale"] = stat_df.shortname.apply(lambda x: 1 if x.endswith("RS1") else 0.05)
bsl_stat_df = compare_conditions(rootdir, find_matching_runs(rootdir, "*none*dif",), read_tensorboard=False, )
# RND_stat_df = compare_conditions(rootdir, find_matching_runs(rootdir, "*RND*dif*",), read_tensorboard=False, )
#%%
xorder = ['DT100UF5', 'DT100UF40', 'DT100UF100', 'DT100UF100_neg', 'DT100UF500']
figh, ax = plt.subplots(1, 1, figsize=(6, 4.5))
sns.barplot(x="dynam", y="AUC", hue="rewardscale", data=stat_df, ci=50, estimator=np.mean, order=xorder)
sns.stripplot(x="dynam", y="AUC", hue="rewardscale", data=stat_df, dodge=True,
              color="r", jitter=0.2, edgecolor="black", order=xorder)
plt.axhline(bsl_stat_df.AUC.mean(), color="k", linestyle="-", label="PPO baseline")
plt.axhline(bsl_stat_df.AUC.mean() - bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.axhline(bsl_stat_df.AUC.mean() + bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.xticks(rotation=30)
plt.ylabel("AUC", fontsize=12)
plt.xlabel("Condition: Dynamics (DT, decay timescale; UF, update frequency)", fontsize=12)
plt.title("Drifting Reward Function", fontsize=14)
plt.legend(loc='best')  # [None,None,"PPO baseline", "RND shuffled", "RND"],
plt.tight_layout()
plt.savefig(join(figroot, "fix_distribution_reward", "drifting_reward_cmp_bar_UF.pdf"), dpi=300)
plt.savefig(join(figroot, "fix_distribution_reward", "drifting_reward_cmp_bar_UF.png"), dpi=300)
plt.show()
#%%
xorder = ['DT40UF40', 'DT40UF100', 'DT100UF100', 'DT500UF100']
figh, ax = plt.subplots(1, 1, figsize=(4.8, 4.5))
sns.barplot(x="dynam", y="AUC", hue="rewardscale", data=stat_df, ci=50, estimator=np.mean, order=xorder)
sns.stripplot(x="dynam", y="AUC", hue="rewardscale", data=stat_df, dodge=True,
              color="r", jitter=0.2, edgecolor="black", order=xorder)
plt.axhline(bsl_stat_df.AUC.mean(), color="k", linestyle="-", label="PPO baseline")
plt.axhline(bsl_stat_df.AUC.mean() - bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.axhline(bsl_stat_df.AUC.mean() + bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.xticks(rotation=30)
plt.ylabel("AUC", fontsize=12)
plt.xlabel("Condition: Dynamics (DT, decay timescale)", fontsize=12)
plt.title("Drifting Reward Function", fontsize=14)
plt.legend(loc='best')  # [None,None,"PPO baseline", "RND shuffled", "RND"],
plt.tight_layout()
plt.savefig(join(figroot, "fix_distribution_reward", "drifting_reward_cmp_bar_DT.pdf"), dpi=300)
plt.savefig(join(figroot, "fix_distribution_reward", "drifting_reward_cmp_bar_DT.png"), dpi=300)
plt.show()
#%% Experiment 4: Shuffled RND
stat_df = compare_conditions(rootdir, find_matching_runs(rootdir, "*RND*dif*",), read_tensorboard=False, )
stat_df["shuffled"] = stat_df.condition.str.contains("RNDShfl")
stat_df["shortname"] = stat_df.condition.apply(lambda x: x.split("_")[-1])
bsl_stat_df = compare_conditions(rootdir, find_matching_runs(rootdir, "*none*dif",), read_tensorboard=False, )
# RND_stat_df = compare_conditions(rootdir, find_matching_runs(rootdir, "*RND*dif*",), read_tensorboard=False, )
#%%
figh, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.barplot(x="shortname", y="AUC", hue="shuffled", data=stat_df, ci=50, estimator=np.mean, )
sns.stripplot(x="shortname", y="AUC", hue="shuffled", data=stat_df, dodge=True,
              color="r", jitter=0.2, edgecolor="black", )
plt.axhline(bsl_stat_df.AUC.mean(), color="k", linestyle="-", label="PPO baseline")
plt.axhline(bsl_stat_df.AUC.mean() - bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.axhline(bsl_stat_df.AUC.mean() + bsl_stat_df.AUC.sem(), color="k", linestyle=":", )
plt.xticks(rotation=30)
plt.ylabel("AUC", fontsize=12)
plt.xlabel("Condition: Drop probability", fontsize=12)
plt.title("Shuffled RND", fontsize=14)
plt.legend(loc='best') # [None,None,"PPO baseline", "RND shuffled", "RND"],
plt.tight_layout()
plt.savefig(join(figroot, "fix_distribution_reward", "RNDshuffle_reward_cmp_bar2.pdf"), dpi=300)
plt.savefig(join(figroot, "fix_distribution_reward", "RNDshuffle_reward_cmp_bar2.png"), dpi=300)
plt.show()

#%% Intrinsic reward curve for different RND drop probabilities


# mergetab0 = merge_traj_all_runs(rootdir, "ppo_none_DMMaze-dif",)

#%%
mergetab0 = merge_traj_all_runs_tb(rootdir, "ppo_none_DMMaze-dif",)
mergetab1 = merge_traj_all_runs_tb(rootdir, "ppo_RND_DMMaze-dif_dp01",)
mergetab2 = merge_traj_all_runs_tb(rootdir, "ppo_RND_DMMaze-dif_dp10",)
mergetab3 = merge_traj_all_runs_tb(rootdir, "ppo_randrew_DMMaze-dif",)
mergetab4 = merge_traj_all_runs_tb(rootdir, "ppo_randrew_DMMaze-dif_nonneg_r5",)
#%%
figh, ax = plt.subplots(1, 1, figsize=(6, 4.5))
labels = ['none', 'RND dp0.1', 'RND dp10', 'rand reward', 'rand reward nonneg r5']
for tab, label in zip([mergetab0, mergetab1, mergetab2, mergetab3, mergetab4], labels):
    sns.lineplot(x="step", y="value_y", data=tab, estimator=np.mean, ci=50, n_boot=500, label=label)

plt.legend(loc='best')
plt.ylabel("Extrinsic Reward", fontsize=14)
plt.xlabel("Steps", fontsize=14)
plt.title("DMMaze-difficult: Fixed reward function", fontsize=16)
plt.tight_layout()
plt.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_traj.pdf"), dpi=300)
plt.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_traj.png"), dpi=300)
plt.show()
#%%
#%%
mergetab_intr_0 = merge_traj_all_runs_tb(rootdir, "ppo_none_DMMaze-dif", y="intrinsic_rewards/Average")
mergetab_intr_1 = merge_traj_all_runs_tb(rootdir, "ppo_RND_DMMaze-dif_dp01", y="intrinsic_rewards/Average")
mergetab_intr_2 = merge_traj_all_runs_tb(rootdir, "ppo_RND_DMMaze-dif_dp10", y="intrinsic_rewards/Average")
mergetab_intr_3 = merge_traj_all_runs_tb(rootdir, "ppo_randrew_DMMaze-dif", y="intrinsic_rewards/Average")
mergetab_intr_42 = merge_traj_all_runs_tb(rootdir, "ppo_randrew_DMMaze-dif_nonneg", y="intrinsic_rewards/Average")
mergetab_intr_4 = merge_traj_all_runs_tb(rootdir, "ppo_randrew_DMMaze-dif_nonneg_r5", y="intrinsic_rewards/Average")
#%%
figh, ax = plt.subplots(1, 1, figsize=(6, 4.5))
labels = ['none', 'RND dp0.1', 'RND dp10', 'rand reward', 'rand reward nonneg', 'rand reward nonneg r5']
for tab, label in zip([mergetab_intr_0, mergetab_intr_1, mergetab_intr_2, mergetab_intr_3, mergetab_intr_42, mergetab_intr_4], labels):
    sns.lineplot(x="step", y="value_y", data=tab, estimator=np.mean, ci=50, n_boot=500, label=label)

plt.legend(loc='best')
plt.ylabel("Intrinsic Reward", fontsize=14)
plt.xlabel("Steps", fontsize=14)
plt.title("DMMaze-difficult: Fixed reward function", fontsize=16)
plt.tight_layout()
plt.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_intr_rew_traj.pdf"), dpi=300)
plt.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_intr_rew_traj.png"), dpi=300)
plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=(6, 4.5))
labels = ['none', 'RND dp0.1', 'RND dp10', 'rand reward', 'rand reward nonneg']
for tab, label in zip([mergetab_intr_0, mergetab_intr_1, mergetab_intr_2, mergetab_intr_3, mergetab_intr_42], labels):
    sns.lineplot(x="step", y="value_y", data=tab, estimator=np.mean, ci=50, n_boot=500, label=label)

plt.legend(loc='best')
plt.ylabel("Intrinsic Reward", fontsize=14)
plt.xlabel("Steps", fontsize=14)
plt.title("DMMaze-difficult: Fixed reward function", fontsize=16)
plt.tight_layout()
figh.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_intr_rew_traj_zoom.pdf"), dpi=300)
figh.savefig(join(figroot, "fix_distribution_reward", "fixed_reward_cmp_intr_rew_traj_zoom.png"), dpi=300)
plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=(7, 5))
vec0 = summarize_traj_all_runs_tb(rootdir, "ppo_none_DMMaze-dif", plot=True,
                         ax=ax, color="k", linestyle="-", linewidth=1, alpha=0.5)
vec1 = summarize_traj_all_runs_tb(rootdir, "ppo_RND_DMMaze-dif_dp10", plot=True,
                         ax=ax, color="r", linestyle="-", linewidth=1, alpha=0.5)
vec2 = summarize_traj_all_runs_tb(rootdir, "ppo_randrew_DMMaze-dif", plot=True,
                         ax=ax, color="magenta", linestyle="-", linewidth=1, alpha=0.5)
vec3 = summarize_traj_all_runs_tb(rootdir, "ppo_randrew_DMMaze-dif_nonneg", plot=True,
                         ax=ax, color="green", linestyle="-", linewidth=1, alpha=0.5)
vec4 = summarize_traj_all_runs_tb(rootdir, "ppo_randrew_DMMaze-dif_nonneg_r5", plot=True,
                         ax=ax, color="blue", linestyle="-", linewidth=1, alpha=0.5)
plt.show()
#%%


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