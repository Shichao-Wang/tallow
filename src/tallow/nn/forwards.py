from typing import Dict, Optional

import torch
from molurus import build_fn_kwargs
from torch import nn
from torch.nn.utils import rnn as rnn_utils
from torch.nn.utils.rnn import PackedSequence

from tallow.metrics import TorchMetricProtocol


def module_forward(module: nn.Module, **kwargs):
    forward_kwargs = build_fn_kwargs(module.forward, **kwargs)
    outputs = module(**forward_kwargs)
    return outputs


@torch.no_grad()
def metric_forward(
    metric: TorchMetricProtocol, **kwargs
) -> Dict[str, torch.Tensor]:
    update_kwargs = build_fn_kwargs(metric.update, **kwargs)
    return metric(**update_kwargs)


def rnn_forward(
    rnn: nn.RNNBase,
    embedding: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    preserve_length: bool = False,
):
    """
    :param rnn: nn.RNN
    :param embeddings: (*, seq, emb)
    :param mask: (*, seq)
    :param preserve_length
    :return: (*, seq, hid), (*, num_layers * hid)
    """
    *outer_sizes, seq_len, embed_size = embedding.size()
    flatten_embedding = torch.flatten(embedding, 0, -3)

    if mask is None:
        lengths = torch.sum(
            torch.sum(flatten_embedding, dim=-1) != 0,
            dtype=torch.long,
            dim=-1,
        )
    else:
        flatten_mask = torch.flatten(mask, 0, -2)
        lengths = torch.sum(flatten_mask, dim=1, dtype=torch.long)
    lengths = lengths.to("cpu")

    packed = rnn_utils.pack_padded_sequence(
        flatten_embedding,
        lengths=lengths,
        batch_first=rnn.batch_first,
        enforce_sorted=False,
    )
    packed_hidden_states: PackedSequence
    packed_hidden_states, last_hidden = rnn(packed)
    if isinstance(rnn, nn.LSTM):
        last_hidden = last_hidden[0]
    total_length = seq_len if preserve_length else None
    hidden_states, _ = rnn_utils.pad_packed_sequence(
        packed_hidden_states,
        batch_first=rnn.batch_first,
        total_length=total_length,
    )
    recovered_hidden_states = torch.reshape(
        hidden_states, (*outer_sizes, seq_len, -1)
    )
    recovered_last_state = torch.transpose(last_hidden, 0, 1).reshape(
        *outer_sizes, -1
    )
    return recovered_hidden_states, recovered_last_state


def mh_attention_forward(
    mh: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
):
    """
    Arguments:
        query: (*, q_len, q_size)
        key: (*, m_len, k_size)
        value: (*, m_len, v_size)
        key_mask: (*, m_len)
    """
    *outer_sizes, q_len, _ = query.size()
    query, key, value = [
        x.view(-1, *x.size()[-2:]) for x in [query, key, value]
    ]
    if key_mask is not None:
        key_mask = key_mask.view(-1, key_mask.size(-1))

    if not mh.batch_first:
        query, key, value = [x.transpose(0, 1) for x in [query, key, value]]

    x, weight = mh(
        query=query,
        key=key,
        value=value,
        key_padding_mask=~key_mask if key_mask is not None else None,
        attn_mask=None if attn_mask is None else ~attn_mask,
    )
    if not mh.batch_first:
        x = torch.transpose(x, 0, 1)
    x = x.view(*outer_sizes, q_len, -1)
    return {"output": x, "weight": weight}


def transformer_forward(
    transformer_model: nn.Transformer,
    src_input: torch.Tensor,
    tgt_input: torch.Tensor,
    src_mask: torch.Tensor = None,
    tgt_mask: torch.Tensor = None,
):
    if not transformer_model.batch_first:
        src_input, tgt_input = [
            x.transpose(0, 1) for x in [src_input, tgt_input]
        ]
    # todo generate mask
    transformer_output: torch.Tensor = transformer_model(
        src=src_input,
        tgt=tgt_input,
        src_key_padding_mask=~src_mask if src_mask is not None else None,
        tgt_key_padding_mask=~tgt_mask if tgt_mask is not None else None,
    )
    if not transformer_model.batch_first:
        transformer_output = transformer_output.transpose(0, 1)

    return transformer_output


def transformer_encoder_forward(
    encoder_model: nn.TransformerEncoder,
    src_input: torch.Tensor,
    *,
    seq_mask: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    """
    Arguments:
        encoder_model:
        src_input: (*, seq_len, hid_size)
        seq_mask: (seq_len, seq_len)
        mask: (*, seq_len)
    Return:
        (*, seq_len, size)
    """
    *outer_sizes, length, hidden_size = src_input.size()
    encoder_output: torch.Tensor = encoder_model(
        src=src_input.view(-1, length, hidden_size),
        mask=None if seq_mask is None else ~seq_mask,
        src_key_padding_mask=None if mask is None else ~mask,
    )
    return encoder_output.view(*outer_sizes, length, -1)


def transformer_decoder_forward(
    decoder_model: nn.TransformerDecoder,
    tgt_input: torch.Tensor,
    mem_input: torch.Tensor,
    tgt_mask: torch.Tensor = None,
    mem_mask: torch.Tensor = None,
    seq_mask: torch.Tensor = None,
):
    """
    Arguments:
        decoder_model: nn.TransformerDecoder assume it is batch_first
    """
    return decoder_model.forward(
        tgt_input,
        mem_input,
        tgt_mask=None,
        tgt_key_padding_mask=None if tgt_mask is None else ~tgt_mask,
        memory_key_padding_mask=None if mem_mask is None else ~mem_mask,
    )
