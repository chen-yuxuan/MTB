from typing import List, Dict, Any
import json

import numpy as np


def entropy(support: List[float]) -> float:
    pass


def compute_metrics(count_file: str, exclude_majority: bool=True) -> Dict[str, Any]:
    dataset_name = count_file.split('_')[:-1]
    with open(count_file, 'r') as f:
        count: Dict[Any, int] = json.load(f)
    print(count_file, count)
    
