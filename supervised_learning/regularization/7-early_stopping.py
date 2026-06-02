#!/usr/bin/env python3
"""Early Stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether training should stop early

    Args:
        cost: current validation cost
        opt_cost: lowest recorded validation cost
        threshold: minimum improvement threshold
        patience: patience count
        count: current count

    Returns:
        tuple: (stop, count)
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return (count >= patience, count)
