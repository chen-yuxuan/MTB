import logging

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from mtb.data import TACREDDataset
from mtb.model import MTBModel
from mtb.processor import BatchTokenizer, aggregate_batch
from mtb.train_eval import train, eval
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
    eval_dataset = TACREDDataset(cfg.eval_file, entity_marker=entity_marker)
    label_names = train_dataset.label_to_id.keys()

    # set dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, collate_fn=aggregate_batch
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, collate_fn=aggregate_batch
    )

    # set a processor that tokenizes and aligns all the tokens in a batch
    batch_processor = BatchTokenizer(
        tokenizer_name_or_path=cfg.model,
        variant=cfg.variant,
        max_length=cfg.max_length,
    )
    vocab_size = len(batch_processor.tokenizer)

    # set model and device
    model = MTBModel(
        encoder_name_or_path=cfg.model,
        variant=cfg.variant,
        vocab_size=vocab_size,
        num_classes=len(label_names),
        dropout=cfg.dropout,
    )
    device = (
        torch.device("cuda", cfg.cuda_device)
        if cfg.cuda_device > -1
        else torch.device("cpu")
    )

    train(
        model,
        train_loader,
        label_names,
        batch_processor,
        num_epochs=cfg.num_epochs,
        lr=cfg.lr,
        device=device,
    )
    result = eval(model, eval_loader, label_names, batch_processor, device)
    logger.info("Result F1: {:.4f}".format(result))


if __name__ == "__main__":
    main()
