from torch.utils.data import DataLoader

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from mtb.data import TACREDDataset
from mtb.processors import BERTProcessor
from mtb.utils import resolve_relative_path


logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="configs")
def main(cfg: DictConfig) -> None:
    """
    Conducts evaluation given the configuration.
    Args:
        cfg: Hydra-format configuration given in a dict.
    """
    resolve_relative_path(cfg)
    print(OmegaConf.to_yaml(cfg))

    train_dataset = TACREDDataset(cfg.train_file)
    val_dataset = TACREDDataset(cfg.val_file)

    if cfg.variant in ["d", "e", "f"]:
        entity_marker = True
    processor = BERTProcessor(
        tokenizer_name_or_path=cfg.model,
        entity_marker=entity_marker,
        max_length=cfg.max_length,
    )

    train_dataset, val_dataset = map(
        processor, (train_dataset, val_dataset)
    )


if __name__ == "__main__":
    main()
