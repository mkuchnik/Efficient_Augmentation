#!/usr/bin/env python
# coding: utf-8


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.metrics
import scipy
import string


# In[3]:


# Set the default plot style
#default_plt_width = 15
#default_plt_height = 10
#plt.rcParams['figure.figsize'] = [default_plt_width, default_plt_height]


# In[4]:


sns.set_style("whitegrid")
sns.set_context("paper")
sns.despine(left=True)
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set(font_scale=1)
#cmap = sns.color_palette("dark")
cmap = sns.color_palette("dark", 10)
sns.palplot(cmap)
sns.set_palette(cmap)


# In[5]:


filename_prefix = "aug_results_CIFAR10_0_vs_1_crop_10"


# In[6]:


# Parameters
#filename_prefix = "aug_results_NORB_0_vs_1_crop_10_loss"


# In[7]:


runs_data = np.load("{}.npz".format(filename_prefix))


# In[8]:


runs_data.keys()


# In[9]:


for k in runs_data:
    try:
        v = runs_data[k]
        print("{}: {}".format(k, v))
    except AttributeError as ex:
        print(ex)
        raise ex


# In[10]:


hist_bins=None
hist_color=cmap.as_hex()[0]
#hist_color="r"
hist_kws=dict(alpha=1)


# In[11]:


hist_color


# In[12]:


initial_aug_scores = pd.Series(runs_data["initial_aug_scores"],
                               name="scores").abs()
hist = sns.distplot(initial_aug_scores,
                    kde=False,
                    bins=hist_bins,
                    color=hist_color,
                    hist_kws=hist_kws)
hist.set_xlabel('')
hist.set_yscale('log')
hist.get_figure().savefig(filename_prefix + "_abs_initial_aug_scores_histogram.pdf")


# In[ ]:


fig, ax = plt.subplots()
after_abs_aug_scores = pd.Series(runs_data["after_aug_scores"],
                                 name="scores").abs()
hist = sns.distplot(after_abs_aug_scores,
                    kde=False,
                    bins=hist_bins,
                    color=hist_color,
                    hist_kws=hist_kws)
hist.set_xlabel('')
hist.set_yscale('log')
hist.get_figure().savefig(filename_prefix + "_after_abs_aug_scores_histogram.pdf")


# In[ ]:


after_abs_aug_scores_cond = after_abs_aug_scores[:len(initial_aug_scores)]
hist = sns.distplot(after_abs_aug_scores_cond,
                    kde=False,
                    bins=hist_bins,
                    color=hist_color,
                    hist_kws=hist_kws)
hist.set_xlabel('')
hist.set_yscale('log')
hist.get_figure().savefig(filename_prefix + "_after_abs_aug_scores_cond_histogram.pdf")


# In[ ]:


joint = sns.jointplot(x=initial_aug_scores,
                      y=after_abs_aug_scores_cond,
                      marginal_kws=dict(bins=10,
                                        rug=False,
                                        hist=False,
                                        kde=True,
                                        kde_kws=dict(bw=.001,
                                                     alpha=1.0,
                                                    ))
             );
joint.ax_joint.set_xlabel('')
joint.ax_joint.set_ylabel('')
joint.ax_joint.get_figure().savefig(filename_prefix + "_init_after_joint_histogram.pdf")


# ## Correlation in Scores

# Let's see how correlated the influences are before and after augmentation

# In[ ]:


scipy.stats.spearmanr(initial_aug_scores, after_abs_aug_scores_cond)


# In[ ]:


small_scores_idxs = initial_aug_scores < initial_aug_scores.mean()


# In[ ]:


scipy.stats.spearmanr(initial_aug_scores[small_scores_idxs],
                      after_abs_aug_scores_cond[small_scores_idxs])


# In[ ]:


initial_aug_scores.describe()


# In[ ]:


after_abs_aug_scores.describe()


# In[ ]:


initial_aug_scores[small_scores_idxs].describe()


# In[ ]:


after_abs_aug_scores_cond[small_scores_idxs].describe()


# In[ ]:


sns.jointplot(x=initial_aug_scores[small_scores_idxs],
              y=after_abs_aug_scores_cond[small_scores_idxs],
              marginal_kws=dict(bins=10,
                                rug=False,
                                hist=False,
                                kde=True,
                                kde_kws=dict(bw=.0005))
             );


# ## Support Vectors

# In[ ]:


is_SV = runs_data["is_SV"].astype(np.int)
print("There are {} support vectors".format(np.sum(is_SV)))
hist = sns.distplot(is_SV, kde=False, bins=hist_bins)
hist.set_yscale('log')
hist.get_figure().savefig("is_SV_histogram.pdf")


# In[ ]:


VSV_acc = runs_data["VSV_acc"]


# In[ ]:


VSV_acc


# ## Parameters

# In[ ]:


runs_data["run_parameters"]


# In[ ]:


baseline_acc = runs_data["no_aug_no_poison_acc"]
poisoned_acc = runs_data["poisoned_acc"]
all_aug_train_poisoned_acc = runs_data["all_aug_train_poisoned_acc"]
n_aug_sample_points = runs_data["n_aug_sample_points"]
n_train = runs_data["n_train"]


# In[ ]:


baseline_acc


# In[ ]:


poisoned_acc


# In[ ]:


all_aug_train_poisoned_acc


# In[ ]:


n_train


# In[ ]:


runs_data["experiment_results"].item()


# In[ ]:


labels = list(runs_data["experiment_results"].item().keys())


# In[ ]:


run_matrix = np.array([
    np.array(runs_data["experiment_results"].item()[k]) for k in labels  
])


# In[ ]:


run_matrix


# In[ ]:


run_df_rows = []
for i, label in enumerate(labels):
    for test in range(run_matrix[i].shape[0]):
        run_df_row = pd.Series()
        run_df_row["test_i"] = test
        run_df_row["n_auged"] = 0
        run_df_row["test_type"] = label
        run_df_row["test_accuracy"] = float(poisoned_acc)
        run_df_rows.append(run_df_row)
        for step in range(run_matrix[i].shape[1]):
            run_df_row = pd.Series()
            run_df_row["test_i"] = test
            run_df_row["n_auged"] = n_aug_sample_points[step]
            run_df_row["test_type"] = label
            run_df_row["test_accuracy"] = run_matrix[i][test, step]
            run_df_rows.append(run_df_row)
run_df = pd.DataFrame(run_df_rows)


# In[ ]:


run_df


# In[ ]:


#old_filename_prefix = filename_prefix + "_old"


# In[ ]:


#old_run_df = pd.read_pickle("{}.pkl".format(old_filename_prefix))


# In[ ]:


#old_run_df


# In[ ]:


#to_remove = [i for i, x in enumerate(old_run_df["test_type"]) if "update" in x]


# In[ ]:


#to_remove


# In[ ]:


#old_run_df.drop(index=to_remove,
#                inplace=True)


# In[ ]:


#run_df


# In[ ]:


#to_not_remove = [i for i, x in enumerate(run_df["test_type"]) if "update" not in x]


# In[ ]:


#run_df.drop(index=to_not_remove,
#            inplace=True)


# In[ ]:


#run_df = pd.concat([old_run_df, run_df])


# In[ ]:


run_df


# In[ ]:


run_df.to_pickle(filename_prefix + ".pkl")
run_df.to_csv(filename_prefix + ".csv", encoding='utf-8')


# In[ ]:


n_aug_sample_points = run_df["n_auged"].unique()


# In[ ]:


n_aug_sample_points


# In[ ]:


all_samples_points = np.unique(np.concatenate([[0], n_aug_sample_points]))


# In[ ]:


all_samples_points


# In[ ]:


aucs = (run_df
        .sort_values("n_auged", ascending=True)
        .groupby(["test_type", "test_i"])["test_accuracy"]
        .apply(
            lambda x: sklearn.metrics.auc(all_samples_points, x)
        )
       )


# In[ ]:


aucs


# In[ ]:


aucs.groupby("test_type").mean()


# In[ ]:


aucs.groupby("test_type").var()


# In[ ]:


auc_means = aucs.groupby("test_type").mean().sort_values(ascending=False).rename("AUC Mean")


# In[ ]:


auc_std = aucs.groupby("test_type").std().sort_values(ascending=False).rename("AUC Std.")


# In[ ]:


auc_mean_std = pd.concat([auc_means, auc_std], axis=1).sort_values(ascending=False, by="AUC Mean")


# In[ ]:


#auc_mean_std.index.name = "Test Type"


# In[ ]:


auc_mean_std


# In[ ]:


replace_dict = {x: string.capwords(x.replace("_", " ")) for x in auc_mean_std.index.unique()}


# In[ ]:


replace_dict


# In[ ]:


formatted_auc_mean_std = auc_mean_std.copy(deep=True)
formatted_auc_mean_std = formatted_auc_mean_std.replace(np.nan, "{\textemdash}")


# In[ ]:


formatted_auc_mean_std.index.name = "Policy"
formatted_auc_mean_std = formatted_auc_mean_std.reset_index()


# In[ ]:


allowed_columns = {"baseline",
                   "random_proportional",
                   "random_proportional_update",
                   "random_proportional_downweight",
                   "random_proportional_update_downweight",
                   "random_inverse_proportional",
                   "deterministic_proportional",
                   "deterministic_proportional_update",
                   "deterministic_proportional_downweight",
                   "deterministic_proportional_update_downweight",
                   "deterministic_inverse_proportional"
                  }
removed_idxs = [i for i, x in
                enumerate(formatted_auc_mean_std["Policy"])
                if x not in allowed_columns]


# In[ ]:


removed_idxs


# In[ ]:


formatted_auc_mean_std.drop(index=removed_idxs, inplace=True)


# In[ ]:


formatted_auc_mean_std["Policy"].unique()


# In[ ]:


#formatted_auc_mean_std["Policy"] = formatted_auc_mean_std["Policy"].str.replace(r"_", r"\_")
formatted_auc_mean_std["Policy"] = formatted_auc_mean_std["Policy"].replace(replace_dict, regex=False)


# In[ ]:


formatted_auc_mean_std


# In[ ]:


header_names=["{{{}}}".format(c) for c in formatted_auc_mean_std.columns]
with open(filename_prefix + '_auc_mean_std.tex', 'w') as f:
    f.write(formatted_auc_mean_std.to_latex(
        na_rep="---",
        escape=False,
        header=header_names,
        index=False,
        column_format="l*{{{right_cols}}}{{S}}".format(right_cols=len(formatted_auc_mean_std.columns)-1)
    ))


# In[ ]:


best_at_n = run_df.groupby("n_auged", as_index=False).max()


# In[ ]:


best_at_n


# In[ ]:


baseline_perf = run_df.query("test_type == 'baseline'").groupby("n_auged", as_index=False).mean()


# In[ ]:


baseline_perf


# In[ ]:


run_df.dtypes


# In[ ]:


run_plot = sns.lineplot(x="n_auged",
                        y="test_accuracy",
                        ci=95,
                        data=run_df.query("test_type == 'baseline'"))
run_plot.axhline(y=baseline_acc,
                 color="b",
                 linestyle="--",
                 label="baseline_acc")
run_plot.axhline(y=poisoned_acc,
                 color="r",
                 linestyle="--",
                 label="poisoned_acc")
run_plot.axhline(y=all_aug_train_poisoned_acc,
                 color="g",
                 linestyle="--",
                 label="all_aug_train_poisoned_acc")
run_plot = sns.lineplot(x="n_auged",
                        y="test_accuracy",
                        ci=95,
                        data=best_at_n,
                        ax=run_plot)
"""
run_plot = sns.relplot(x="n_auged",
                       y="test_accuracy",
                       hue="test_i",
                       col="test_type",
                       col_wrap=4,
                       ci=0,
                       markers=True,
                       kind="line",
                       data=run_df,
                       ax=run_plot,
                       alpha=0.1)
                       """


# In[ ]:


run_plot = sns.lineplot(x="n_auged",
                        y="test_accuracy",
                        hue="test_type",
                        ci=95,
                        data=run_df)
run_plot.axhline(y=baseline_acc,
                 color="b",
                 linestyle="--",
                 label="baseline_acc")
run_plot.axhline(y=poisoned_acc,
                 color="r",
                 linestyle="--",
                 label="poisoned_acc")
run_plot.axhline(y=all_aug_train_poisoned_acc,
                 color="g",
                 linestyle="--",
                 label="all_aug_train_poisoned_acc")
#run_plot.get_figure().savefig("test_accuracy_summary.pdf")


# In[ ]:


run_plot = sns.relplot(x="n_auged",
                       y="test_accuracy",
                       hue="test_i",
                       col="test_type",
                       col_wrap=4,
                       ci=95,
                       markers=True,
                       kind="line",
                       data=run_df)


# In[ ]:


run_plot = sns.relplot(x="n_auged",
                       y="test_accuracy",
                       hue="test_type",
                       col="test_type",
                       col_wrap=4,
                       ci=95,
                       markers=True,
                       kind="line",
                       palette=sns.color_palette("Set2", n_colors=len(run_df["test_type"].unique())),
                       data=run_df.query("n_auged <= 10"))


# In[ ]:


sns.set_style("whitegrid")
sns.set_context("paper")
sns.despine(left=True)
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
cmap = sns.color_palette("Set1")
sns.palplot(cmap)
sns.set_palette(cmap)


# In[ ]:


update_downweight_run_df = (run_df.query("test_type == 'random_proportional'"
                                          "| test_type == 'random_proportional_update'"
                                          "| test_type == 'random_proportional_downweight'"
                                          "| test_type == 'random_proportional_update_downweight'"
                                          "| test_type == 'baseline'")
                            .query("n_auged < 1100"))
update_downweight_run_df = update_downweight_run_df.rename(
    index=str,
    columns={
            "test_accuracy": "Test Accuracy",
            "n_auged": "Number of Augmented Points",

            },
)
update_downweight_run_df["test_type"] = update_downweight_run_df["test_type"].replace(replace_dict, regex=False)
fig, ax = plt.subplots()
run_plot = sns.lineplot(x="Number of Augmented Points",
                        y="Test Accuracy",
                        hue="test_type",
                        style="test_type",
                        ci=95,
                        data=update_downweight_run_df,
                        markers=True,
                        dashes=True,
                        ax=ax)
l = ax.legend()
#l.texts[0].set_text("")
#l.set_title('Whatever you want')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])
#run_plot.axhline(y=baseline_acc,
#                 color="b",
#                 linestyle="--",
#                 label="baseline_acc")
run_plot.axhline(y=poisoned_acc,
                 color="r",
                 linestyle="--",
                 label="poisoned_acc")
run_plot.axhline(y=all_aug_train_poisoned_acc,
                 color="g",
                 linestyle="--",
                 label="all_aug_train_poisoned_acc")
run_plot.get_figure().savefig(filename_prefix + "_modifications_accuracy.pdf")


# In[ ]:


cmap = sns.color_palette("Set1")
yellow = cmap[5]
cmap[5] = cmap[6]
cmap[6] = yellow
sns.palplot(cmap)
sns.set_palette(cmap)
sns.set(font_scale=1.00)


# In[ ]:


update_downweight_run_df = (run_df.query("test_type == 'deterministic_proportional'"
                                          "| test_type == 'deterministic_proportional_update'"
                                          "| test_type == 'deterministic_proportional_downweight'"
                                          "| test_type == 'deterministic_proportional_update_downweight'"
                                          "| test_type == 'deterministic_inverse_proportional'"
                                          "| test_type == 'baseline'")
                            .query("n_auged < 1100"))
update_downweight_run_df = update_downweight_run_df.rename(
    index=str,
    columns={"test_accuracy": "Test Accuracy",
             "n_auged": "Number of Augmented Points",
            },
)
update_downweight_run_df["test_type"] = update_downweight_run_df["test_type"].replace(replace_dict, regex=False)
fig, ax = plt.subplots()
run_plot = sns.lineplot(x="Number of Augmented Points",
                        y="Test Accuracy",
                        hue="test_type",
                        style="test_type",
                        ci=95,
                        data=update_downweight_run_df,
                        markers=True,
                        dashes=True,
                        ax=ax)
l = ax.legend()
#l.texts[0].set_text("")
#l.set_title('Whatever you want')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])
#run_plot.axhline(y=baseline_acc,
#                 color="b",
#                 linestyle="--",
#                 label="baseline_acc")
run_plot.axhline(y=poisoned_acc,
                 color="r",
                 linestyle="--",
                 label="poisoned_acc")
run_plot.axhline(y=all_aug_train_poisoned_acc,
                 color="g",
                 linestyle="--",
                 label="all_aug_train_poisoned_acc")
run_plot.get_figure().savefig(filename_prefix + "_deterministic_accuracy.pdf")


# In[ ]:




