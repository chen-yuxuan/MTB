from typing import Union, Dict, Any

import torch
import datasets
from transformers import AutoTokenizer


class Processor:
    def __init__(
        self,
        entity_marker: bool = True,
        text_column_name: str = "token",
    ):
        self.entity_marker = entity_marker
        self.text_column_name = text_column_name

    def __call__(
        self, dataset: Union[datasets.Dataset, torch.utils.data.Dataset]
    ) -> Union[datasets.Dataset, torch.utils.data.Dataset]:
        pass


class BERTProcessor(Processor):
    """Provide processing methods that are executed between fetching batch from
    a `torch.utils.data.Dataset` and feeding this batch to a language model.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str = "bert-base-cased",
        entity_marker: bool = True,
        text_column_name: str = "token",
        max_length: int = 256,
        padding: str = "max_length",
    ) -> None:
        super().__init__(entity_marker, text_column_name)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if entity_marker:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]}
            )

        self.max_length = max_length
        self.padding = padding

    def __call__(
        self, dataset: Union[datasets.Dataset, datasets.DatasetDict]
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        return dataset
