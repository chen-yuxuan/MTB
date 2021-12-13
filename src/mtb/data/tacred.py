import torch
import datasets

from logging import getLogger
from typing import List

from .base import REDataset


logger = getLogger(__name__)


_COLUMNS_TO_REMOVE = [
    "id",
    "docid",
    "stanford_pos",
    "stanford_ner",
    "stanford_head",
    "stanford_deprel",
]


class TACREDDataset(REDataset):
    def __init__(self, data_file: str = None, label_column_name: str = "relation"):
        super().__init__(data_file, label_column_name)

        self.dataset = self.dataset.remove_columns(_COLUMNS_TO_REMOVE)

        self.label_to_id = self.get_label_to_id(self.label_column_name)
        self.add_column_for_label_id(
            column_name="relation", new_column_name="relation_id"
        )
