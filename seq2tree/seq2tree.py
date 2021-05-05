import copy
import math
import os
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from seq2tree_encoder import T5Stack
from seq2tree.seq2tree_decoder import TreeDecoderStack


class T5SmallConfig:

    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50000,
        d_model=512,
        d_kv=64,
        d_ff=1024,
        num_layers=8,
        num_decoder_layers=8,
        num_heads=6,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        is_decoder=False,
    ):
        self.is_decoder = is_decoder
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers



class Seq2TreeModel(nn.Module):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TreeDecoderStack(decoder_config)

        self.types_lm_head = nn.Linear(config.d_model, config.types_vocab_size, bias=False)
        self.values_lm_head = nn.Linear(config.d_model, config.types_vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_dict=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        types,values,tree_positions,rel = decoder_input_dict["types"], decoder_input_dict["values"],["tree_positions"],decoder_input_dict["rel_tokens"]
        # Decode
        sequence_output = self.decoder(
            types,
            values,
            tree_positions,
            rel,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # return (decoder_outputs[0], )
        types_lm_logits = self.types_lm_head(sequence_output)
        values_lm_logits = self.values_lm_head(sequence_output)
        return types_lm_logits, values_lm_logits


class MaskedLoss(nn.Module):
    def __init__(self, pad_idx, oov_idx, empty_idx):
        super(MaskedLoss, self).__init__()
        self.pad_idx = pad_idx
        self.oov_idx = oov_idx
        self.empty_idx = empty_idx
        self.loss_fct = CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, pred_logits, labels):
        # y_pred,  = inputs
        # assert len(y.size()) == 2
        # # we do not calculate loss on the history part of the sequence
        # # from ast splitting
        # ext_r = ext.unsqueeze(-1).repeat(1, y.size(-1))
        # ext_ids = torch.arange(y.size(-1), device=ext_r.device).view(1, -1).repeat(*(y.size()[:-1] + (1,)))
        # where = ext_ids >= ext_r  # skip the memory from the previous code snippet
        # where &= y != self.pad_idx  # calc loss only on known tokens and filter padding and empty values
        # where &= y != self.oov_idx
        # where &= y != self.empty_idx
        # where = where.view(-1)
        #
        # y_pred = y_pred.view(-1, y_pred.size(-1))
        # y = y.view(-1)
        # if where.sum() == 0:
        #     return y_pred.new_ones(1, requires_grad=True) * 1e-8  # in case the seq is empty
        # loss = super(MaskedLoss, self).forward(y_pred[where], y[where])
        # return loss
        loss = self.loss_fct(pred_logits.view(-1, pred_logits.size(-1)), labels.view(-1))
        return loss