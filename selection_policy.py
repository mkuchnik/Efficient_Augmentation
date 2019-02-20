import numpy as np


def softmax(x):
    """Stable softmax"""
    x -= np.max(x, axis=0)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=0)


def get_idx_aug_baseline(LOO_influences):
    """Returns points randomly"""
    idxs = np.random.choice(
            len(LOO_influences),
            len(LOO_influences),
            p=None,
            replace=False,
    )
    for idx in idxs:
        yield [idx]


def get_idx_aug_influence(LOO_influences):
    """Returns points with probability proportional to magnitude of LOO"""
    p = np.abs(LOO_influences, dtype=float)
    p[p == 0] = min(np.min(p[p > 0]), 1e-20)
    p /= np.sum(p)
    idxs = np.random.choice(
            len(LOO_influences),
            len(LOO_influences),
            p=p,
            replace=False,
    )
    for idx in idxs:
        yield [idx]


def get_idx_aug_k_dpp(LOO_influences, k):
    """Returns points with probability proportional to L matrix using DPP"""
    import sample_dpp
    L = LOO_influences.T.dot(LOO_influences)
    assert len(L) == len(LOO_influences)
    idxs = sample_dpp.oct_sample_k_dpp(
        L,
        k=k,
        one_hot=False)
    for idx in idxs:
        yield [idx]


def get_idx_aug_influence_reverse(LOO_influences):
    """Returns points with probability proportional to magnitude of LOO"""
    p = np.abs(LOO_influences)
    p[p == 0] = min(np.min(p[p > 0]), 1e-20)
    p = 1 / p
    p /= np.sum(p)
    p[p == 0] = 1e-20
    p /= np.sum(p)
    idxs = np.random.choice(
            len(LOO_influences),
            len(LOO_influences),
            p=p,
            replace=False,
    )
    for idx in idxs:
        yield [idx]


def get_idx_aug_softmax_influence(LOO_influences):
    """Returns points with probability proportional to softmax of magnitude
       of LOO"""
    p = np.abs(LOO_influences)
    p[p == 0] = min(np.min(p[p > 0]), 1e-20)
    p = math_util.softmax(p)
    idxs = np.random.choice(
            len(LOO_influences),
            len(LOO_influences),
            p=p,
            replace=False,
    )
    for idx in idxs:
        yield [idx]


def get_idx_aug_softmax_influence_reverse(LOO_influences):
    """Returns points with probability proportional to softmax of magnitude
       of LOO"""
    p = np.abs(LOO_influences)
    p[p == 0] = min(np.min(p[p > 0]), 1e-20)
    p = 1 / p
    p = math_util.softmax(p)
    p[p == 0] = 1e-20
    p /= np.sum(p)
    idxs = np.random.choice(
            len(LOO_influences),
            len(LOO_influences),
            p=p,
            replace=False,
    )
    for idx in idxs:
        yield [idx]


def get_idx_aug_deterministic_influence(LOO_influences):
    """Returns points in deterministic order ranked by LOO magnitude"""
    idxs = np.argsort(-np.abs(LOO_influences))
    for idx in idxs:
        yield [idx]


def get_idx_aug_deterministic_influence_reverse(LOO_influences):
    """Returns points in deterministic order ranked by LOO magnitude"""
    idxs = np.argsort(np.abs(LOO_influences))
    for idx in idxs:
        yield [idx]

name_to_policy = {
    "baseline": get_idx_aug_baseline,
    "random_proportional": get_idx_aug_influence,
    "random_inverse_proportional": get_idx_aug_influence_reverse,
    "random_softmax_proportional": get_idx_aug_softmax_influence,
    "random_inverse_softmax_proportional":
        get_idx_aug_softmax_influence_reverse,
    "deterministic_proportional": get_idx_aug_deterministic_influence,
    "deterministic_inverse_proportional":
        get_idx_aug_deterministic_influence_reverse,
}

def get_policy_by_name(name):
    return name_to_policy[name]

