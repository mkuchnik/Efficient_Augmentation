import itertools

import papermill as pm

datasets = ["CIFAR10_0_vs_1"]
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
        "filename_prefix": "aug_results_dpp_norm_{}_{}_10{}".format(
            dataset, aug, loss_str
        ),
    }
    try:
        pm.execute_notebook(
            "Visualize_LOO_Experiments.ipynb",
            "Visualize_LOO_Experiments.ipynb",
            parameters=parameters
        )
    except Exception as ex:
        print("Except: {}".format(ex))
