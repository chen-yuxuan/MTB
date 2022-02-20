from typing import Any, List, Dict, Tuple
import re
import json

from datasets import load_dataset


_RELATIONS = [
    "Cause-Effect(e1,e2)",
    "Cause-Effect(e2,e1)",
    "Component-Whole(e1,e2)",
    "Component-Whole(e2,e1)",
    "Content-Container(e1,e2)",
    "Content-Container(e2,e1)",
    "Entity-Destination(e1,e2)",
    "Entity-Destination(e2,e1)",
    "Entity-Origin(e1,e2)",
    "Entity-Origin(e2,e1)",
    "Instrument-Agency(e1,e2)",
    "Instrument-Agency(e2,e1)",
    "Member-Collection(e1,e2)",
    "Member-Collection(e2,e1)",
    "Message-Topic(e1,e2)",
    "Message-Topic(e2,e1)",
    "Product-Producer(e1,e2)",
    "Product-Producer(e2,e1)",
    "Other",
]
_LABEL_TO_ID = {k: v for v, k in enumerate(_RELATIONS)}
_ID_TO_LABEL = {v: k for v, k in enumerate(_RELATIONS)}


def reformat_to_tacred(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an original semeval-formatted example to tacred-format.

    Args:
        example: A row with columns "sentence" and int-type "relation", which is the
        original format of the semeval dataset.

    Returns:
        A dictionary with columns "token", "relation", "subj_start",
        "subj_end", "obj_start" and "obj_end".
    """
    # convert relation ids to relation names for the "relation column"
    example["relation"] = _ID_TO_LABEL[example["relation"]]

    # tokenize the "sentence" column and mark the start/end positions of
    # e1 (subj) and e2 (obj)
    sentence = (
        example["sentence"]
        .replace("<e1>", " subjstart ")
        .replace("</e1>", " subjend ")
        .replace("<e2>", " objstart ")
        .replace("</e2>", " objend ")
    )
    token = re.findall(r"[\w]+|[^\s\w]", sentence)
    subj_start, subj_end, obj_start, obj_end = (
        token.index("subjstart"),
        token.index("subjend"),
        token.index("objstart"),
        token.index("objend"),
    )
    for marker in ["subjstart", "subjend", "objstart", "objend"]:
        if marker in token:
            token.remove(marker)
    example["token"] = token

    # calibrate the entity spans after removal of markers
    if subj_start < obj_start:
        subj_end -= 2
        obj_start -= 2
        obj_end -= 4
    else:
        obj_end -= 2
        subj_start -= 2
        subj_end -= 4
    (
        example["subj_start"],
        example["subj_end"],
        example["obj_start"],
        example["obj_end"],
    ) = (subj_start, subj_end, obj_start, obj_end)

    return example


if __name__ == "__main__":
    dataset = load_dataset("sem_eval_2010_task_8")
    trainset, testset = dataset["train"], dataset["test"]

    trainset = trainset.map(reformat_to_tacred).remove_columns("sentence")
    testset = testset.map(reformat_to_tacred).remove_columns("sentence")

    with open("train.json", "w") as fw:
        for example in trainset:
            fw.write(json.dumps(example) + "\n")
    with open("test.json", "w") as fw:
        for example in testset:
            fw.write(json.dumps(example) + "\n")
