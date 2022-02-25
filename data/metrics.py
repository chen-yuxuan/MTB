from typing import List, Dict, Any
import json

import numpy as np


def compute_shannon_index(
    count: Dict[Any, int], with_negative: bool = True
) -> float:
    count = list(count.values())
    count.sort()
    count = np.array(count)
    if not with_negative:
        count = count[:-1]

    proportion = count / sum(count)
    entropy_per_class = -proportion * np.log(proportion)
    return float(sum(entropy_per_class))


def compute_metrics(count_file: str) -> Dict[str, Any]:
    dataset_name = "".join(count_file.split("_")[:-1]).replace("./", "")
    has_negative_class = (
        True if "tacred" in dataset_name or "semeval" in dataset_name else False
    )

    with open(count_file, "r") as f:
        count: Dict[Any, int] = json.load(f)

    num_examples = sum(count.values())
    shannon_index_with_negative = compute_shannon_index(count, True)
    report = {
        "dataset": dataset_name,
        "num_classes": len(count),
        "num_examples": num_examples,
        "shannon_index_with_negative": shannon_index_with_negative,
    }

    if has_negative_class:
        report["num_negative_examples"] = max(count.values())
        report["shannon_index_without_negative"] = compute_shannon_index(count, False)
    else:
        report["num_negative_examples"] = 0
    return report


if __name__ == "__main__":
    print(compute_metrics("./tacred_count.json"))
    print(compute_metrics("./tacred_test_count.json"))

    print(compute_metrics("./semeval_count.json"))
    print(compute_metrics("./semeval_test_count.json"))

    print(compute_metrics("./chemprot_count.json"))
    # "SUBSTRATE_PRODUCT-OF" has no examples in test split
    print(compute_metrics("./chemprot_test_count.json"))
