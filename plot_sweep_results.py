import pandas as pd
import pathlib

results_filenames = [
    "aug_results_CIFAR10_0_vs_1_translate_10_sweep.csv",
    "aug_results_CIFAR10_0_vs_1_rotate_10_sweep.csv",
    "aug_results_CIFAR10_0_vs_1_crop_10_sweep.csv",
    "aug_results_NORB_0_vs_1_translate_10_sweep.csv",
    "aug_results_NORB_0_vs_1_rotate_10_sweep.csv",
    "aug_results_NORB_0_vs_1_crop_10_sweep.csv",
    "aug_results_MNIST_3_vs_8_translate_10_sweep.csv",
    "aug_results_MNIST_3_vs_8_rotate_10_sweep.csv",
    "aug_results_MNIST_3_vs_8_crop_10_sweep.csv",
]
for filename in results_filenames:
    df = pd.read_csv(filename).set_index("n_clusters")
    print(df)
    df[
        #["test_silhoette_score", "train_silhoette_score"]
        ["test_silhouette_score", "train_silhouette_score"]
    ].plot().get_figure().savefig(
        pathlib.Path("sil_" + filename).with_suffix(".pdf")
    )
    df[["train_intertia"]].plot().get_figure().savefig(
        pathlib.Path("intertia_" + filename).with_suffix(".pdf")
    )
