import torch
import datasets

from typing import Union, Dict, Any


class DatasetProcessor:
    def __init__(self, entity_marker: str = True):
        self.entity_marker = entity_marker

    def __call__(
        self, dataset: Union[datasets.Dataset, torch.utils.data.Dataset]
    ) -> Union[datasets.Dataset, torch.utils.data.Dataset]:
        pass

    def insert_entity_marker(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """For variants d, e, f, add entity markers."""
