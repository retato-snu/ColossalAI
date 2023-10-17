from typing import Optional

import torch.nn as nn
from transformers import GPTNeoXConfig, GPTNeoXModel

from ..base import RewardModel


class PolyglotkoRM(RewardModel):
    """
    OPT Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPTNeoXConfig): Model config.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        config: Optional[GPTNeoXConfig] = None,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
    ) -> None:
        if pretrained is not None:
            model = GPTNeoXModel.from_pretrained(pretrained)
        elif config is not None:
            model = GPTNeoXModel(config)
        else:
            model = GPTNeoXModel(GPTNeoXConfig())

        value_head = nn.Linear(model.config.hidden_size, 1)
        value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))
        super().__init__(model, value_head, lora_rank, lora_train_bias)
