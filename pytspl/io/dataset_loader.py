"""Module for preprocessing and loading datasets for analysis.
"""

from .data_loaders.forex_loader import load_forex_data
from .data_loaders.lastfm_loader import load_lastfm_1k_artist
from .data_loaders.paper_loader import load_paper_data
from .data_loaders.transportation_loader import (
    list_transportation_datasets,
    load_transportation_dataset,
)
from .data_loaders.wsn_loader import load_wsn_data
from .data_loaders.webkb_loader import load_webkb_data, _load_webkb_subset
from .data_loaders.matpower_loader import load_matpower_data
from .data_loaders.wntr_loader import load_wntr_data

DATASETS = {
    "paper": load_paper_data,
    "forex": load_forex_data,
    "lastfm-1k-artist": load_lastfm_1k_artist,
    "webkb-cornell": lambda only_sc=True: _load_webkb_subset("cornell", only_sc=only_sc),
    "webkb-texas":   lambda only_sc=True: _load_webkb_subset("texas", only_sc=only_sc),
    "webkb-wisconsin": lambda only_sc=True: _load_webkb_subset("wisconsin", only_sc=only_sc),
    "wsn": load_wsn_data,
    "matpower-case57": lambda only_sc=True: load_matpower_data("case57", only_sc=only_sc),
    "matpower-case118": lambda only_sc=True: load_matpower_data("case118", only_sc=only_sc),
    "matpower-case145": lambda only_sc=True: load_matpower_data("case145", only_sc=only_sc),
    "matpower-case300": lambda only_sc=True: load_matpower_data("case300", only_sc=only_sc),
    "matpower-ACTIVSg200": lambda only_sc=True: load_matpower_data("case_ACTIVSg200", only_sc=only_sc),
    "matpower-ACTIVSg500": lambda only_sc=True: load_matpower_data("case_ACTIVSg500", only_sc=only_sc),
    "wntr-bellingham": lambda only_sc=True: load_wntr_data("bellingham", only_sc=only_sc),
    "wntr-fairfield": lambda only_sc=True: load_wntr_data("fairfield", only_sc=only_sc),
    "wntr-oberlin": lambda only_sc=True: load_wntr_data("oberlin", only_sc=only_sc),
}


def list_datasets() -> list:
    """
    List the available datasets.

    Returns:
        list: The list of available datasets.
    """
    other_datasets = list(DATASETS.keys())
    transportation_datasets = list_transportation_datasets()
    return transportation_datasets + other_datasets


def load_dataset(dataset: str, only_sc: bool = True) -> tuple:
    """
    Load the dataset and return the simplicial complex
    and coordinates.

    Args:
        dataset (str): The name of the dataset.
        only_sc (bool, optional): if true load the dataset as a simplicial complex, else as a cell complex

    ValueError:
        If the dataset is not found.

    Returns:
        SimplicialComplex/CellComplex: The simplicial/cell complex of the dataset.
        dict: The coordinates of the nodes. If the coordinates do not
        exist, the coordinates are generated using spring layout.
        dict: The flow data of the dataset. If the flow data does not
        exist, an empty dictionary is returned.
    """
    datasets = list_datasets()
    if dataset not in datasets:
        raise ValueError(
            f"Dataset {dataset} not found. Available datasets: {datasets}"
        )

    if dataset in DATASETS:
        complex, coordinates, flow = DATASETS[dataset](only_sc=only_sc)
    else:
        complex, coordinates, flow = load_transportation_dataset(dataset=dataset, only_sc=only_sc)

    assert complex is not None
    assert coordinates is not None
    assert flow is not None



    # each node should have a coordinate
    assert len(complex.nodes) == len(coordinates)

    # print summary of the dataset
    complex.print_summary()
    print(f"Coordinates: {len(coordinates)}")
    print(f"Flow: {len(flow)}")

    return complex, coordinates, flow
