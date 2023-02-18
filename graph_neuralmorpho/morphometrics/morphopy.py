"""Computing traditional morphology features."""
import argparse
import warnings
from argparse import RawTextHelpFormatter
from dataclasses import dataclass, field
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


def export_features_from_pkl(pkl_path: Path | str, export_file: str | None) -> None:
    """Export morphometric features from pkl file to csv in the same directory.

    Args:
        pkl_path (Path): path to pkl file of swc data.
        export_file (str): name of exported file.
    """
    pkl_path = Path(pkl_path)
    print(f"Starting features extraction from {pkl_path.name}...")
    swc_dict = pd.read_pickle(pkl_path)
    morpho_features = batch_process_morphopy_features(swc_dict)
    morpho_features["filename"] = pkl_path.name
    if export_file is None:
        export_file = f"{pkl_path.stem}_morphopy_features.csv"
    else:
        export_file = export_file if Path(export_file).suffix == ".csv" else f"{export_file}.csv"
    morpho_features.to_csv(f"{pkl_path.parent}/{export_file}", index=False)
    print(f"Finished! Exported features data to {pkl_path.parent}/{export_file}.")


def load_morphopy_features(features_dir: Path | str) -> tuple[pd.DataFrame, dict[int, str]]:
    """Load morphometric features from a directory.

    Args:
        features_dir (Path): path to directory containing csv files of morphology features.

    Returns:
        tuple[pd.DataFrame, dict[int, str]]: tuple of morphology features and label dictionary.
    """
    df_list = []
    label_dict = {}

    for cls, file in enumerate(Path(features_dir).glob("*morphopy_features.csv")):
        df = pd.read_csv(file)
        df["target"] = cls
        df_list.append(df)
        label_dict[cls] = file.stem.split("_morphopy")[0]

    morphopy_features = pd.concat(df_list).drop(columns=["filename"]).reset_index(drop=True)
    morphopy_features = morphopy_features.astype({"target": "category"})

    return morphopy_features, label_dict


@dataclass(frozen=True)
class MorphopyFeatures:
    """Features data from MorphoPy.

    Attributes:
        features_dir (Path): path to directory containing csv files of morphology features.
        neurons (pd.Series): series of neuron names.
        data (pd.DataFrame): DataFrame of morphology features.
        cls_labels (dict[int, str]): dictionary of class labels.

    """

    features_dir: Path | str
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    target: pd.Series = field(default=pd.Series(dtype=int))
    neurons: pd.Series = field(default=pd.Series(dtype=str))
    label_dict: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set up data from features directory."""
        data, label_dict = load_morphopy_features(self.features_dir)
        target = data.pop("target")
        neurons = data.pop("neuron_name")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "neurons", neurons)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "label_dict", label_dict)


###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Extract morphology features from a dict of swc data.""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to file containing swc data.")
    parser.add_argument(
        "-f", "--feature_filename", help="Name of morphology features file.", default=None
    )

    args = parser.parse_args()
    export_features_from_pkl(args.input_file, args.feature_filename)
