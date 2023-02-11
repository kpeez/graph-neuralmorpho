"""Computing traditional morphology features."""
import warnings
from pathlib import Path

import pandas as pd
from morphopy.computation.feature_presentation import compute_morphometric_statistics
from morphopy.neurontree import NeuronTree as nt


def compute_morphopy_features(neuron_swc: pd.DataFrame) -> pd.DataFrame:
    """Compute morphometric features using MorphoPy.

    Args:
        neuron_swc (pd.DataFrame): swc data .

    Returns:
        morphology_features: DataFrame of morphology features.
    """
    if "id" in neuron_swc.columns:
        neuron_swc.rename(columns={"id": "n"}, inplace=True)

    neuron_tree = nt.NeuronTree(neuron_swc)
    morphopy_features = compute_morphometric_statistics(neuron_tree)

    return morphopy_features


def batch_process_morphopy_features(swc_dict: dict) -> pd.DataFrame:
    """Batch process morphometric features using MorphoPy.

    Args:
        swc_dict (dict): dictionary of swc data.


    Returns:
        morphology_features: DataFrame of morphology features.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    morpho_data_list = []
    for neuron, swc_data in swc_dict.items():
        try:
            morpho_data = compute_morphopy_features(swc_data)
            morpho_data["neuron_name"] = neuron
            morpho_data_list.append(morpho_data)
        except ValueError:
            print(f"Error processing {neuron}! Skipping...")

    return pd.concat(morpho_data_list, axis=0)


def export_features_from_pkl(pkl_path: Path | str, export_file: str) -> None:
    """Export morphometric features from pkl file to csv in the same directory.

    Args:
        pkl_path (Path): path to pkl file of swc data.
        export_file (str): name of exported file.
    """
    pkl_path = Path(pkl_path)
    print(f"Starting features extraction from {Path(pkl_path.name)}...")
    swc_dict = pd.read_pickle(pkl_path)
    morpho_features = batch_process_morphopy_features(swc_dict)
    morpho_features["filename"] = pkl_path.name
    export_file = export_file if Path(export_file).suffix == ".csv" else f"{export_file}.csv"
    morpho_features.to_csv(f"{pkl_path.parent}/{export_file}", index=False)
    print(f"Finished! Exported features to {pkl_path.parent}/{export_file}.")
