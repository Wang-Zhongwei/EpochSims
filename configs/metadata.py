import json
from typing import Union


def load_metadata(file_path: str) -> dict:
    with open(file_path, "r") as file:
        metadata = json.load(file)
    return metadata


def get_simulation_metadata(metadata: dict, simulation_id: str) -> Union[None, dict]:
    for entry in metadata:
        if entry["id"] == simulation_id:
            return entry
    return None
