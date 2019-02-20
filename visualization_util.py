import matplotlib.pyplot as plt
import logging


def show_aug_images(x_, x_aug_, ):
    f, ax = plt.subplots(len(x_aug_), 2)
    if len(x_) > 1:
        logging.warning("x_ has shape '{}' which is greater than"
                        " length 1".format(x_.shape))
    x_show = x_[0]
    if x_show.shape[2] < 3:
        x_show = x_show.reshape(x_show.shape[:2])
        ax[0, 0].imshow(x_show,
                        cmap="gray",
                        aspect="auto")
    else:
        ax[0, 0].imshow(x_show,
                        aspect="auto")
    for i in range(len(x_aug_)):
        ax[i, 0].axis("off")
    for i, x_i in enumerate(x_aug_):
        if x_aug_[i].shape[2] < 3:
            ax[i, 1].imshow(x_aug_[i].reshape(x_aug_[i].shape[:2]),
                            cmap="gray",
                            aspect="auto")
        else:
            ax[i, 1].imshow(x_aug_[i],
                            aspect="auto")
        ax[i, 1].axis("off")
    plt.show()


def plot_figures(figures, nrows=1, ncols=1, title_on=False):
    """Plot a dictionary of figures.

    From:
    https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols,
                                 nrows=nrows,
                                 gridspec_kw={"wspace": 0, "hspace": 0},
                                 squeeze=True)
    for ind, title in zip(range(len(figures)), figures):
        img = figures[title]
        if img.shape[2] < 3:
            img = img.reshape(img.shape[:2])
        #axeslist.ravel()[ind].imshow(img, cmap="gray", aspect="auto")
        axeslist.ravel()[ind].imshow(img, cmap="gray")
        if title_on:
            axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional
