import logging

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from mtb.data import TACREDDataset
from mtb.model import MTBModel
from mtb.processor import BatchTokenizer
from mtb.evaluator import trainer
from mtb.utils import resolve_relative_path, seed_everything


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

    seed_everything(cfg.seed)

    # prepare dataset: parse raw dataset and do some simple pre-processing such as
    # convert special tokens and insert entity markers
    entity_marker = True if cfg.variant in ["d", "e", "f"] else False
    train_dataset = TACREDDataset(cfg.train_file, entity_marker=entity_marker)
    val_dataset = TACREDDataset(cfg.val_file, entity_marker=entity_marker)

    # set dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True
    )

    # set processor that tokenizes and aligns all the tokens in a batch
    batch_processor = BatchTokenizer(
        tokenizer_name_or_path=cfg.model,
        variant=cfg.variant,
        max_length=cfg.max_length,
    )
    vocab_size = len(batch_processor.tokenizer)

    # set model
    model = MTBModel(vocab_size=vocab_size)
    device = (
        torch.device("cuda", cfg.cuda_device)
        if cfg.cuda_device > -1
        else torch.device("cpu")
    )

    trainer(
        model,
        train_dataloader,
        val_dataloader,
        batch_processor,
        num_epochs=cfg.num_epochs,
        lr=cfg.lr,
        device=device,
    )


if __name__ == "__main__":
    main()
