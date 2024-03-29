from logging import getLogger
from typing import List, Dict, Any
import random

from .base import REDataset


logger = getLogger(__name__)


class SmilerDataset(REDataset):
    """The SemEval dataset from the raw data file(s)."""

    def __init__(
        self,
        data_file: str,
        entity_marker: bool = True,
        text_column_name: str = "token",
        label_column_name: str = "relation",
    ):
        """
        Args:
            data_file: Path to the .json file for the split of data.
            entity_marker: A boolean to indicate whether or not to insert entity markers ("<e1>",
            "</e1>", "<e2>", "</e2>") to the original text. The `True` case corresponds to variants
            "a", "b" and "c".
            text_colomn_name: The name of the column for the text. "token" for TACRED here.
            label_column_name: The name of the column for the label. "relation" for TACRED here.
        """
        super().__init__(data_file, entity_marker, text_column_name, label_column_name)

        # add entity marker accordingly
        if self.entity_marker:
            self.dataset = self.dataset.map(
                self.insert_entity_markers,
                fn_kwargs={"text_column_name": self.text_column_name},
            )


class SmilerFewShotDataset(SmilerDataset):
    """Few-shot version of the SemEval dataset.

    The size of this dataset is `N` * `K` if the sampled classes have >= K examples.
    """

    def __init__(
        self,
        data_file: str,
        kshot: int = 5,
        entity_marker: bool = True,
        include_no_relation: bool = True,
    ):
        super().__init__(data_file, entity_marker)
        self.kshot = kshot
        self.include_no_relation = include_no_relation

        self.class_indices = self._get_indices_per_class()
        self.num_examples = {k: len(v) for k, v in self.class_indices.items()}
        # print the relation names in descending order of number of examples
        logger.info("Number of examples per class:", self.num_examples)

        self.sampled_indices = self._sample_indices()
        self.dataset = self.dataset.select(self.sampled_indices)

    def _get_indices_per_class(self) -> Dict[Any, List[int]]:
        """For each class, maintain a list of indices from this class."""
        class_indices: Dict[str, List[int]] = {}
        for idx, example in enumerate(self.dataset):
            label = example[self.label_column_name]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def _sample_indices(self) -> List[int]:
        """Sample the indices in the dataset: For each of the `N` classes, sample `K` indices."""
        # get a list of valid classes
        if not self.include_no_relation:
            self.class_indices.pop("no_relation", None)        

        # sample K-shots for each sampled class; take all if the data is not sufficient for this class
        sampled_indices: List[int] = []
        for sampled_class in list(self.class_indices.keys()):
            sampled_indices.extend(
                random.sample(
                    self.class_indices[sampled_class],
                    k=min(len(self.class_indices[sampled_class]), self.kshot),
                )
            )
        return sampled_indices

    def __len__(self):
        """Length of the few-shot dataset, should equal `N` * `K`."""
        return len(self.dataset)
