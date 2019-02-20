import numpy as np
import os
from skimage import img_as_float
import pylearn2.filetensor as filetensor

def to_one_hot(y, n_classes=10):
    Y = np.zeros([y.shape[0], n_classes])
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
    return Y

def load_norb_data(train_data_dir, test_data_dir, dims, n_classes, one_hot=True, as_float=True,
    validation_set=False, jittered=True, data_slice=1, feat_slice=1):
    """Loads training and test data. If validation_set=False, 
    returns only training, otherwise splits into training and validation.
    """
    # load data
    train = {}
    test = {}
    train["dat"] = os.path.join(train_data_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
    train["cat"] = os.path.join(train_data_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat')
    train["info"] = os.path.join(train_data_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat')
    if jittered:
        test["dat"]  = os.path.join(test_data_dir, 'norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat')
        test["cat"]  = os.path.join(test_data_dir, 'norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat')
        test["info"]  = os.path.join(test_data_dir, 'norb-5x01235x9x18x6x2x108x108-testing-01-info.mat')
    else:
        test["dat"]  = os.path.join(test_data_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
        test["cat"]  = os.path.join(test_data_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat')
        test["info"]  = os.path.join(test_data_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat')

    # load testing data
    te_data = filetensor.read(open(test["dat"], "rb"))
    te_labels = filetensor.read(open(test["cat"], "rb"))
    if jittered:
        # force image to be size 96 x 96
        te_data = te_data[::data_slice, 0, 7:102:feat_slice, 7:102:feat_slice]
        # get rid of 6th class
        te_data = te_data[np.where(te_labels!=5),:,:]
        te_labels = te_labels[np.where(te_labels!=5)]
    else:
        te_data = te_data[::data_slice, 0, ::feat_slice, ::feat_slice] # just use one image, not stereo
        te_labels = te_labels[::data_slice]
    te_data = img_as_float(te_data)
    X_test = te_data.reshape(te_data.shape[0],-1)
    X_test = X_test.reshape(-1, dims[0], dims[1], dims[2])
    if one_hot:
        Y_test = to_one_hot(te_labels, n_classes)
    else:
        Y_test = te_labels

    # load training data
    tr_data = filetensor.read(open(train["dat"], "rb"))
    tr_data = tr_data[::data_slice, 0, ::feat_slice, ::feat_slice] # just use one image, not stereo
    tr_data = img_as_float(tr_data)
    tr_labels = filetensor.read(open(train["cat"], "rb"))
    tr_labels = tr_labels[::data_slice]
    if one_hot:
        tr_labels  = to_one_hot(tr_labels, n_classes)

    # split into training and validation
    X_train, Y_train, X_valid, Y_valid = [], [], [], []
    nsamples = tr_data.shape[0]
    if validation_set:
        # validation is random set of training
        ntrain = (nsamples * 4) / 5
        nvalid = nsamples - ntrain
        rng = np.random.RandomState(0)
        indices = rng.permutation(nsamples)
        itr  = indices[0:ntrain]
        ival = indices[ntrain:ntrain+nvalid]
        X_train = tr_data[itr,...].reshape(ntrain, -1)
        Y_train = tr_labels[itr]
        X_valid = tr_data[ival,...].reshape(nvalid, -1)
        Y_valid = tr_labels[ival]
    else:
        X_train = tr_data.reshape(nsamples, -1)
        Y_train = tr_labels
        X_valid = X_test
        Y_valid = Y_test

    # reshape
    X_train = X_train.reshape(-1, dims[0], dims[1], dims[2])
    X_valid = X_valid.reshape(-1, dims[0], dims[1], dims[2])

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
