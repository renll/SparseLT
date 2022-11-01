# Copyright (c) Liliang Ren, Zixuan Zhang.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


from transformers import RobertaPreTrainedModel, RobertaModel,RobertaForMaskedLM, AutoModel, BertPreTrainedModel, BertModel, BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from transformers.activations import ACT2FN
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

from utils import RobertaConfig
from typing import List, Optional, Tuple, Union

from decoder import BartDecoder, _make_causal_mask, _expand_mask

import torch
import math
import torch.nn as nn

from gumbel_latent_typer import GumbelLatentTyper


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
        


class RobertaLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class RobertaAutoEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        
        self.glm_head = None
        self.glm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True) # for confitional generation (lm)
        nn.init.constant_(self.glm_head.bias, 0.0)

        self.decoder = BartDecoder(config, self.roberta.embeddings)
        
        
        self.sa_pm = GumbelLatentTyper(
                dim = config.hidden_size,
                num_vars = 64,
                temp =  (5, 0.5, 1-3e-5),
                var_dim = config.hidden_size,
                hard = False,
            )

        self.tie_weights()
    
    @property
    def roberta(self):
        return self.model.bert

    @property
    def mlm_head(self):
        return self.model.cls.predictions

    def tie_weights(self,):
        if self.glm_head is not None:
            self.glm_head.weight = self.roberta.embeddings.word_embeddings.weight

        self.mlm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight
        
    def forward(self, input_ids=None, attention_mask=None, mlm_input_ids=None, mlm_labels=None, decoder_input_ids=None, decoder_attention_mask=None, gen_labels=None, original_tokens=None, return_dict=None):
        

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # loss #1: masked lm loss
        masked_sequence_output = self.roberta(
            mlm_input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
        prediction_scores = self.mlm_head(masked_sequence_output[0])

        masked_lm_loss = None

        if mlm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        
        # loss #2: reconstruction loss
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
        sequence_output = outputs[0]

        # sequence_output: (batch, seq_len, dim)
        EPS = torch.finfo(sequence_output.dtype).tiny
        b,q,c = sequence_output.shape
        result = self.sa_pm(sequence_output,mask=attention_mask, deterministic=True)
        
        div_loss = (result["num_vars"] - result["prob_perplexity"])/result["num_vars"] 
        
        soft_probs =  result["soft_probs"].view(b,q,-1)[:,:,0]
        reduced_output = (sequence_output * result["x"])
        pm_loss = - torch.log((soft_probs*attention_mask).sum()/attention_mask.sum()+EPS)

        seq_logits = self.decoder(
            input_ids=decoder_input_ids, 
            attention_mask=decoder_attention_mask, 
            encoder_hidden_states=reduced_output, 
            encoder_attention_mask=attention_mask
        )[0]

        lm_logits = self.glm_head(seq_logits)
        gen_loss = None
        if gen_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            gen_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), gen_labels.view(-1)) 
        
        if torch.isnan(masked_lm_loss):
            masked_lm_loss = gen_loss.new_zeros(1)[0]

        return masked_lm_loss, gen_loss, pm_loss, div_loss
    

    def test_generate(self, input_ids=None, attention_mask=None, original_tokens=None, return_dict=None, tsne=False, return_latent = False):
        bs, seq_len = input_ids.shape
        decoder_input_ids = torch.zeros(bs, seq_len).long()
        decoder_attn_mask = torch.zeros(bs, seq_len).long()

        decoder_input_ids[:, 0:1] = input_ids[:, 0:1]
        decoder_attn_mask[:, 0:1] = torch.ones(bs, 1).long()

        output_ids = torch.zeros(bs, seq_len).long()
        output_ids[:, 0:1] = input_ids[:, 0:1]

        type_str = ""
        selected_list = []

        with torch.no_grad():
            # loss #2: reconstruction loss
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict
            )
            sequence_output = outputs[0]

            # sequence_output: (batch, seq_len, dim)
            EPS = torch.finfo(sequence_output.dtype).tiny
            b,q,c = sequence_output.shape
            result = self.sa_pm(sequence_output, deterministic=True)
            gumbel_types = torch.argmax(result["gumbel_probs"][:, 0, :], 1)
            
            if tsne:
                return sequence_output, gumbel_types

            #only support batch_size = 1 after this line
            reduced_output = (sequence_output * result["x"])
            type_ids = []
            for j in range(len(original_tokens[0])):
                token = original_tokens[0][j]
                type_idx = gumbel_types.tolist()[j]
                type_ids.append(type_idx)
                if type_idx != 0:
                    type_str += (token + '(' + str(type_idx)+'), ')
                    selected_list.append(token)

            if return_latent:
                return type_ids

            print("LATENT TYPINGS: ")
            print(type_str)
            print('\n')
            

            for i in range(seq_len - 1):
                seq_logits = self.decoder(
                    input_ids=decoder_input_ids, 
                    attention_mask=decoder_attn_mask, 
                    encoder_hidden_states=reduced_output, 
                    encoder_attention_mask=attention_mask
                )[0]
                # seq_logits: bs, seq_len, vocab_size
                lm_logits = self.glm_head(seq_logits)
                selected_logits = lm_logits[:, i, :]
                logit_idxs = torch.argmax(selected_logits, 1)
                output_ids[:, i+1:i+2] = logit_idxs.unsqueeze(-1)

                decoder_input_ids[:, i+1:i+2] = logit_idxs.unsqueeze(-1)
                decoder_attn_mask[:, i+1:i+2] = torch.ones(bs, 1)

        return output_ids



