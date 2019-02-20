import numpy as np
import logging
import PSD_util.PSD_util as PSD_util

def oct_sample_k_dpp(L, k, one_hot=False, enable_logging=False):
    """
    This function expects 

    Arguments: 
    L: lambda matrix from L-ensembles (a determinantal point process).
        The matrix is real and symmetric (and positive semidefinite).
        The rows and columns of L using an index set form a marginal probability
        of inclusion of those subsets.
        L only needs to indicate proportionality of drawing the subset as the
        normalization is available in closed form.
    k: The number of samples to draw from the dpp
    one_hot: If true, return sample indexes in a one_hot form.
    enable_logging: If True, turns on debug logger

    Returns:
        The indexes of the samples
    """
    import oct2py
    print("mean L", np.mean(L))
    print("std L", np.std(L))
    assert isinstance(L, np.ndarray), "L must be a numpy array"
    assert isinstance(k, int), "K must be integer"
    assert len(L.shape) == 2, "L must be a matrix"
    assert L.shape[0] == L.shape[1], "L must be square"
    assert np.allclose(L, L.T, atol=1e-8), "L must be symmetric"
    assert k >= 0, "K must be non-negative"
    assert k <= len(L), "K can't be greater than the number of points"
    if not PSD_util.isPD(L):
        PD_L = PSD_util.nearestPD(L)
        assert PSD_util.isPD(PD_L), "L must be PSD"
        PD_correction = np.linalg.norm(L - PD_L)
        print("PD_correction: {}".format(PD_correction))
        L = PD_L
    if enable_logging:
        logging.basicConfig(level=logging.DEBUG)
        oc = oct2py.Oct2Py(logger=logging.getLogger())
    else:
        oc = oct2py.Oct2Py()
    oc.addpath(".")
    if k == 0:
        return np.array([])
    else:
        sample = oc.wrapped_sample_dpp(L, k)
        print("DPP sample: {}".format(type(sample)))
        if k == 1:
            # For some reason, this is not an array
            sample = np.array([sample])
        else:
            sample = np.array(sample).flatten()

    assert len(sample) == k

    # TODO this may be unsafe
    sample = np.around(sample).astype(int)

    # Convert from matlab index to python
    sample -= 1
    assert np.all(sample) >= 0, "Samples index must be non-negative"
    assert np.all(sample) < len(L), \
        "Samples index must be less than number of elements"

    if one_hot:
        sample_one_hot = np.zeros(len(L))
        sample_one_hot[sample] = np.ones(k)
        return sample_one_hot
    else:
        return sample
