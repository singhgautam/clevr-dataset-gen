import math
import random

import itertools as it


def bind_generator(counts, train_alphas, test_alpha):

    assert max(train_alphas) <= 1. - test_alpha, "Improper Train/Test split ratio"

    num_props = len(counts)
    max_count = max(counts)

    core_binds = set([tuple([i % counts[j] for j in range(num_props)]) for i in range(max_count)])

    all_binds = set(it.product(*[list(range(count)) for count in counts]))

    noncore_binds = all_binds.difference(core_binds)
    num_noncore_binds = len(noncore_binds)

    num_test_binds = math.floor(num_noncore_binds * test_alpha)
    test_binds = set(random.sample(noncore_binds, num_test_binds))

    noncore_nontest_binds = noncore_binds.difference(test_binds)

    train_binds = {}
    for train_alpha in train_alphas:
        num_noncore_train_binds = math.ceil(num_noncore_binds * train_alpha)
        train_binds[train_alpha] = set(random.sample(noncore_nontest_binds, num_noncore_train_binds))
        train_binds[train_alpha] = train_binds[train_alpha].union(core_binds)

    return test_binds, train_binds


if __name__ == '__main__':
    bind_generator([2, 3, 4, 3], train_alphas=[0.2, 0.4], test_alpha=0.2)
