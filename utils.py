# Copyright (c) Zixuan Zhang.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


from transformers.configuration_utils import PretrainedConfig


class BertConfig(PretrainedConfig):

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        pad_token_id=1,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        activation_function="gelu",
        decoder_ffn_dim=4096,
        decoder_layers=1,
        init_std=0.02,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.init_std = init_std
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.activation_function = activation_function
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers


class RobertaConfig(BertConfig):

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


if __name__ == "__main__":
    c = BertConfig()
    d = RobertaConfig()
