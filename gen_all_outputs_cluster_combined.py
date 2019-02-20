import itertools

import papermill as pm

datasets = ["MNIST_3_vs_8", "CIFAR10_0_vs_1", "NORB_0_vs_1"]
augs = ["translate", "rotate", "crop"]
is_losses = [False, True]

configs = list(itertools.product(datasets, augs, is_losses))

print(configs)

for config in configs:
    print(config)
    dataset, aug, is_loss = config
    if is_loss:
        loss_str = "_loss"
    else:
        loss_str = ""
    parameters = {
        "filename_prefix": "aug_results_{}_{}_10{}".format(
            dataset, aug, loss_str
        ),
    }
    try:
        pm.execute_notebook(
            "Visualize_LOO_Experiments-Combine_Cluster.ipynb",
            "Visualize_LOO_Experiments-Combine_Cluster.ipynb",
            parameters=parameters
        )
    except Exception as ex:
        print("Error: {}".format(ex))
