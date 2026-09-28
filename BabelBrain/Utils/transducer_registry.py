import os

import yaml

from Utils.paths import resource_path

def _load_transducer_list() -> list[dict]:
    """Load the built-in transducer registry from transducer_list.yaml."""
    transducer_list_yaml = os.path.join(resource_path(__file__).parent, 'SelFiles', 'transducer_list.yaml')
    with open(transducer_list_yaml, 'r') as f:
        return yaml.safe_load(f)

def _get_transducer_names(transducers: list[dict]) -> list[str]:
    tx_names = []

    for tx in transducers:
        tx_names.append(tx['name'])

    return tx_names


DEFAULT_TRANSDUCERS = _load_transducer_list()
DEFAULT_TRANSDUCER_NAMES = _get_transducer_names(DEFAULT_TRANSDUCERS)