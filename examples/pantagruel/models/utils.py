import logging
from typing import Union
from collections import OrderedDict

import torch
import torch.nn as nn

from fairseq import checkpoint_utils
from fairseq.file_io import PathManager

from fairseq.modules import MultiheadAttention

logger = logging.getLogger(__name__)


def _to_bf16(x, forward=True):
    if forward:
        return x.to(torch.bfloat16)
    else:
        return x.to(torch.float16)


def load_all_pretrained_modules_to_model(
    model: nn.Module,
    checkpoint: str,
    skip_modules=[],
):
    """
    load all modules in the checkpopint to the model if the module exists in the model
    """
    l2_norm = torch.sqrt(sum(torch.sum(p ** 2) for p in model.parameters()))
    logger.info(f"{model.__class__.__name__} BEFORE init: {l2_norm.item()}")

    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)

    model_state_dict_after_init = OrderedDict()
    for key, value in model.state_dict().items():
        _init_pretrained = False
        if not skip_modules or (skip_modules and skip_modules[0] != "all"):
            not_skip = all([m not in key for m in skip_modules]) if skip_modules else True
            if not_skip:
                for pk, pv in state["model"].items():
                    if key in pk:
                        logger.info(f"init {key} using pretrained weights")
                        model_state_dict_after_init[key] = pv
                        _init_pretrained = True
                        break
        if not _init_pretrained:
            logger.info(f"random/previous init for {key}")
            model_state_dict_after_init[key] = value

    model.load_state_dict(model_state_dict_after_init, strict=True)

    l2_norm = torch.sqrt(sum(torch.sum(p ** 2) for p in model.parameters()))
    logger.info(f"{model.__class__.__name__} AFTER init: {l2_norm.item()}")
    return model