import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from mtb.data import TACREDDataset
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
    valid_dataset = TACREDDataset(cfg.val_file)
    test_dataset = TACREDDataset(cfg.test_file)

    processor = 0


if __name__ == "__main__":
    main()
