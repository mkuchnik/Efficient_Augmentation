import itertools

import papermill as pm

datasets = ["MNIST_3_vs_8", "CIFAR10_0_vs_1", "NORB_0_vs_1"]
augs = ["translate", "rotate", "crop"]

configs = list(itertools.product(datasets, augs))

print(configs)

for config in configs:
    print(config)
    dataset, aug = config
    parameters = {
        "filename_prefix": "aug_results_{}_{}_10".format(
            dataset, aug
        ),
    }
    pm.execute_notebook(
        "Visualize_LOO_Experiments-Combine_Inf_Loss_Random.ipynb",
        "Visualize_LOO_Experiments-Combine_Inf_Loss_Random.ipynb",
        parameters=parameters
    )
