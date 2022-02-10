import math

import torch
from torch import nn


def is_transformer_batch_first(transformer_model: nn.Transformer):
    return transformer_model.batch_first


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    r"""
    Generate a square mask for the sequence.
        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz, dtype=torch.bool)) == 1).transpose(
        0, 1
    )
    # mask = (
    #     mask.float()
    #     .masked_fill(mask == 0, float("-inf"))
    #     .masked_fill(mask == 1, float(0.0))
    # )
    return mask


def count_module_parameters(module: nn.Module):
    parameter_counts = sum(torch.numel(p) for p in module.parameters())
    return parameter_counts


def positional_encoding(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Return:

    """
    *_, seq_len, hidden_size = hidden_states.size()
    hidden_size = hidden_states.size(-1)

    pe = hidden_states.new_zeros(seq_len, hidden_size)

    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(dim=1)
    div_term = torch.exp(
        (
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * -(math.log(10000.0 / hidden_size))
        )
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.expand_as(hidden_states)
