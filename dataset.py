# Copyright (c) Zixuan Zhang, Liliang Ren.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


class PLMDataset(Dataset):

    def __init__(self, tokenizer, file_path, block_size):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        batch_encoding = tokenizer([line[:-1] for line in lines], add_special_tokens=True, truncation=True, max_length=block_size)
        examples = batch_encoding["input_ids"]
        original_tokens = [['START'] + tokenizer.tokenize(line[:-1]) + ['END'] for line in lines]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long), "original_tokens": original_tokens[i]} for i,e in enumerate(examples)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class PLMDataCollator(object):

    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        # examples: list of {"input_ids": xxx, "original_tokens": xxx}
        input_ids = [{"input_ids": example["input_ids"]} for example in examples]
        original_tokens = [{"original_tokens": example["original_tokens"]} for example in examples]

        batch_src = self.tokenizer.pad(input_ids, return_attention_mask=True, return_tensors="pt")
        batch_tgt = self.tokenizer.pad(input_ids, return_attention_mask=True, return_tensors="pt")

        # If special token mask has been preprocessed, pop it from the dict.
        tgt_labels = batch_tgt.input_ids[:, 1:].clone()
        if self.tokenizer.pad_token_id is not None:
            tgt_labels[tgt_labels == self.tokenizer.pad_token_id] = -100
        # batch_tgt.input_ids[:, 0] = self.tokenizer.eos_token_id
        masked_input_ids, masked_labels = self.mask_tokens(batch_src.input_ids)

        # batch_src 
        batch = {
            "input_ids": batch_tgt.input_ids,
            "attention_mask": batch_src.attention_mask,
            "masked_input_ids": masked_input_ids,
            "masked_labels": masked_labels,
            "decoder_input_ids": batch_tgt.input_ids[:, :-1],
            "decoder_attention_mask": batch_tgt.attention_mask[:, :-1],
            "labels": tgt_labels,
            "original_tokens": original_tokens
        }
        return batch

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100 # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class PLMTrainingArgs(TrainingArguments):

    def add_loss_weights(self, mlm, alpha, beta, gamma ):
        self.mlm = mlm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class PLMTrainer(Trainer):

    def load_tb(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            param_optimizer = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight','layer_norm.bias','layer_norm.weight',]
            slow_lr=['bert']
            optimizer_grouped_parameters = [
                            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and \
                                            not any(nd in n for nd in slow_lr) ], 'weight_decay': self.args.weight_decay,
                                                         'lr': self.args.learning_rate},
                                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and \
                                                         any(nd in n for nd in slow_lr) ], 'weight_decay': self.args.weight_decay,
                                                                      'lr': self.args.learning_rate*0.1},
                                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and \
                                                                     any(nd in n for nd in slow_lr) ], 'weight_decay': 0.0,
                                                                                  'lr': self.args.learning_rate*0.1},
                                                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and \
                                                                                not any(nd in n for nd in slow_lr) ], 'weight_decay': 0.0,
                                                                                             'lr': self.args.learning_rate},
                                                                        ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        ''' main model '''
        losses= model(
            input_ids=inputs["input_ids"].cuda(), 
            attention_mask=inputs["attention_mask"].cuda(),
            mlm_input_ids=inputs["masked_input_ids"].cuda(), 
            mlm_labels=inputs["masked_labels"].cuda(), 
            decoder_input_ids=inputs["decoder_input_ids"].cuda(), 
            decoder_attention_mask=inputs["decoder_attention_mask"].cuda(), 
            gen_labels=inputs["labels"].cuda(), 
            original_tokens=inputs["original_tokens"],
            return_dict=None
        )
        mlm_loss, gen_loss, pm_loss, div_loss = losses
        self.writer.add_scalar('sparse_loss', torch.mean(pm_loss).item(), self.global_step)
        self.writer.add_scalar('mlm_loss', torch.mean(mlm_loss).item(), self.global_step)
        self.writer.add_scalar('gen_loss', torch.mean(gen_loss).item(), self.global_step)
        self.writer.add_scalar('div_loss', torch.mean(div_loss).item(), self.global_step)

        self.model.sa_pm.set_num_updates(self.global_step)
        self.global_step += 1
        
        return self.args.mlm * mlm_loss + self.args.alpha * gen_loss + self.args.beta * pm_loss + self.args.gamma * div_loss
    


