from typing import Dict, Any

from transformers import AutoTokenizer


def aggregate_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate all the values of each column into a list of values.

    This step should be done during data-loading.
    """
    return {
        column_name: [example[column_name] for example in batch]
        for column_name in batch[0]
    }


class BatchTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path: str = "bert-base-cased",
        entity_marker: bool = True,
        text_column_name: str = "token",
        max_length: int = 128,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.entity_marker = entity_marker
        self.text_column_name = text_column_name
        self.max_length = max_length

        if self.entity_marker:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]}
            )

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Call the tokenizer to tokenize the text and align the cue."""
        tokenized = self.tokenizer(
            batch[self.text_column_name],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # update starts and ends for each example within a batch
        subj_starts, subj_ends, obj_starts, obj_ends = [], [], [], []
        for subj_start, subj_end, obj_start, obj_end, offset_mapping in zip(
            batch["subj_start"],
            batch["subj_end"],
            batch["obj_start"],
            batch["obj_end"],
            # shape: (batch_size, max_seq_len_per_batch, 2)
            tokenized["offset_mapping"],
        ):
            # count valid tokens, i.e. not `[CLS]` or `[SEP]` or `[PAD]` or starting with `#`
            count = 0
            for idx, offset in enumerate(offset_mapping):
                # `offset[0] != 0` refers to "starting with `#`";
                # `offset[1] == 0 refers to `[0, 0]`, means `[CLS]` or `[SEP]` or `[PAD]`
                if offset[0] != 0 or offset[1] == 0:
                    continue

                elif count in [subj_start, subj_end]:
                    if count == subj_start:
                        subj_starts.append(idx)
                    if count == subj_end:
                        subj_ends.append(idx)

                elif count in [obj_start, obj_end]:
                    if count == obj_start:
                        obj_starts.append(idx)
                    if count == obj_end:
                        obj_ends.append(idx)
                count += 1

        return tokenized, subj_starts, subj_ends, obj_starts, obj_ends
