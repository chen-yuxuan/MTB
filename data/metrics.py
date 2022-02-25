from typing import List, Dict, Any
import json

import numpy as np


def compute_shannon_index(count: Dict[Any, int], sum_exclude_negative: bool = True) -> float:
    count = list(count.values())
    count.sort()
    count = np.array(count)

    proportion = count / sum(count)
    entropy_per_class = -proportion * np.log(proportion)
    if sum_exclude_negative:
        entropy_per_class = entropy_per_class[:-1]
    return float(sum(entropy_per_class))


def compute_metrics(count_file: str, has_negative_class: bool=True) -> Dict[str, Any]:
    dataset_name = "".join(count_file.split('_')[:-1]).replace('./', '')

    with open(count_file, 'r') as f:
        count: Dict[Any, int] = json.load(f)
    #print(dataset_name, "\n", count)

    num_examples = sum(count.values())
    if has_negative_class:
        num_negative_examples = max(count.values())
    else:
        num_negative_examples = 0

    # compute shannon_index
    shannon_index = compute_shannon_index(count)
    
    return {
        "dataset": dataset_name,
        "num_classes": len(count),
        "num_examples": num_examples,
        "num_negative_examples": num_negative_examples,
        "shannon_index": shannon_index,
    }


if __name__ == "__main__":
    print(compute_metrics("./tacred_count.json", True))
    print(compute_metrics("./tacred_test_count.json", True))

    print(compute_metrics("./semeval_count.json", True))
    print(compute_metrics("./semeval_test_count.json", True))

    print(compute_metrics("./chemprot_count.json", False))
    # "SUBSTRATE_PRODUCT-OF" has no examples in test split
    print(compute_metrics("./chemprot_test_count.json", False))



""" 
{'dataset': 'tacred', 'num_classes': 42, 'num_examples': 106264, 'num_negative_examples': 84491}
{'dataset': 'tacredtest', 'num_classes': 42, 'num_examples': 15509, 'num_negative_examples': 12184}
{'dataset': 'semeval', 'num_classes': 19, 'num_examples': 10717, 'num_negative_examples': 1864}
{'dataset': 'semevaltest', 'num_classes': 19, 'num_examples': 2717, 'num_negative_examples': 454}
{'dataset': 'chemprot', 'num_classes': 13, 'num_examples': 10065, 'num_negative_examples': 0}
{'dataset': 'chemprottest', 'num_classes': 12, 'num_examples': 3469, 'num_negative_examples': 0}
"""