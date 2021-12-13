from transformers import AutoTokenizer
import datasets

from typing import Dict, List, Union

from .base import DatasetProcessor


class BERTProcessor(DatasetProcessor):
    """Provide processing methods that are executed between fetching batch from
    a `torch.utils.data.Dataset` and feeding this batch to a language model.
    """
    def __init__(
        self,
        tokenizer_name_or_path: str = "bert-base-cased",
        entity_marker: str = "True",
        text_column_name: str = "token",
        label_column_name: str = "relation_id",
        max_length: int = 256,
        padding: str = "max_length",
    ) -> None:
        super().__init__(entity_marker)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.max_length = max_length
        self.padding = padding

    def __call__(
        self, dataset: Union[datasets.Dataset, datasets.DatasetDict]
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        if self.entity_marker:
            dataset = dataset.map(self.insert_entity_marker)
        return dataset

