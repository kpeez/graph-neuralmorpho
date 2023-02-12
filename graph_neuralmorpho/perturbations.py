"""Perturbations of neuron structures for contrastive learning."""
import numpy as np
import pandas as pd


def perturbation_point(neuron: pd.DataFrame, proportion: float = 0.5) -> pd.DataFrame:

    # select proportion of points to perturb
    print(proportion)
    np.random.randint(0, 11)  # random distance between 0 and 10 µm

    # shift all points by distance

    return neuron


def perturbation_branch(neuron: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def perturbation_rotate(neuron: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
