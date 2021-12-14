from logging import getLogger
from typing import List

import torch
from transformers import AutoModel

from .base import MTBModel


logger = getLogger(__name__)


class BERTModel(MTBModel):
    pass