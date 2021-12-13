import torch
import datasets

from typing import Union, List, Dict


class REDataset(torch.utils.data.Dataset):
    """A general relation extraction (RE) dataset from the raw dataset."""

    def __init__(self, data_file: str, label_column_name: str = "relation"):
        super().__init__()
        self.dataset = datasets.load_dataset(
            "json", data_files=data_file, split="train"
        )
        self.label_column_name = label_column_name

    def __getitem__(self, index: Union[int, List[int], torch.Tensor]):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def get_label_to_id(self, label_column_name: str = "relation") -> Dict[str, int]:
        """Get a dict of the class-id mapping."""
        label_list = list(set(self.dataset[label_column_name]))
        label_list.sort()
        return {label: i for i, label in enumerate(label_list)}

    def add_column_for_label_id(
        self, column_name: str = "relation", new_column_name: str = "relation_id"
    ) -> None:
        """Add a new column to store the (relation) class ids."""
        new_column_features = [
            self.label_to_id[label] for label in self.dataset[column_name]
        ]
        self.dataset = self.dataset.add_column(new_column_name, new_column_features)
