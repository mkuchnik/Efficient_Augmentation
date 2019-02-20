import itertools
import collections

import papermill as pm

datasets = ["MNIST_3_vs_8", "CIFAR10_0_vs_1", "NORB_0_vs_1"]
datasets = ["CIFAR10_0_vs_1"]
#datasets = ["NORB_0_vs_1"]
augs = ["translate", "rotate", "crop"]
cluster_ks = [1, 10, 50, 100, 250, 500, 750, 1000]
#is_losses = [False, True]
is_losses = [False]

configs = list(itertools.product(datasets, augs, is_losses, cluster_ks))

print(configs)

dataset_dfs = collections.defaultdict(list)

for config in configs:
    print(config)
    dataset, aug, is_loss, cluster_k = config
    if is_loss:
        loss_str = "_loss"
    else:
        loss_str = ""
    parameters = {
        "filename_prefix": "aug_results_{}_{}_10_{}{}".format(
            dataset, aug, cluster_k, loss_str
        ),
    }
    try:
        pm.execute_notebook(
            "Visualize_LOO_Experiments.ipynb",
            "Visualize_LOO_Experiments.ipynb",
            parameters=parameters
        )
    except Exception as ex:
        print("Error: {}".format(ex))
