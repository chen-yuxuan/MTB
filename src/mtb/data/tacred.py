from logging import getLogger

from .base import REDataset


logger = getLogger(__name__)


_COLUMNS_TO_REMOVE = [
    "id",
    "docid",
    "subj_type",
    "obj_type",
    "stanford_pos",
    "stanford_ner",
    "stanford_head",
    "stanford_deprel",
]

_SPECIAL_TOKENS_DICT = {
    "-lrb-": "(",
    "-rrb-": ")",
    "-lsb-": "[",
    "-rsb-": "]",
    "-lcb-": "{",
    "-rcb-": "}",
}


class TACREDDataset(REDataset):
    """The TACRED dataset from the raw data file(s)."""

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
        self.dataset = self.dataset.remove_columns(_COLUMNS_TO_REMOVE)
        self.add_column_for_label_id(new_column_name="relation_id")

        # convert special tokens
        self.dataset = self.dataset.map(
            self.convert_special_tokens,
            fn_kwargs={
                "text_column_name": self.text_column_name,
                "special_tokens_dict": _SPECIAL_TOKENS_DICT,
            },
        )

        # add entity marker accordingly
        if self.entity_marker:
            self.dataset = self.dataset.map(
                self.insert_entity_markers,
                fn_kwargs={"text_column_name": self.text_column_name},
            )


class TACREDFewShotDataset(TACREDDataset):
    """Few-shot version of the TACRED dataset.

    The size of this dataset is `N` * `K` if the sampled classes have >= K examples.
    """

    def __init__(
        self, data_file: str, nway: int = 42, kshot: int = 5, entity_marker: bool = True
    ):
        super().__init__(data_file, entity_marker)
        self.nway = nway
        self.kshot = kshot
