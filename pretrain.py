# Copyright (c) Zixuan Zhang, Liliang Ren.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse

from transformers import AutoConfig, AutoTokenizer

from utils import RobertaConfig
from model import RobertaAutoEncoder
from dataset import PLMDataset, PLMDataCollator, PLMTrainer, PLMTrainingArgs
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="default_latent_typing")
parser.add_argument('--data', type=str, default="voa_corpus")
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--beta', type=float, default=0.05)
parser.add_argument('--gamma', type=float, default=0.1)
args = parser.parse_args()

if args.local_rank in [-1,0]:
    wandb.tensorboard.patch(root_logdir="./tb_logs/")
    wandb.init(project='latent-typing', sync_tensorboard=True)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
checkpoint_dir = "./checkpoints/" + args.name
dataset_dir = "./data/" + args.data + '.txt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)



config = AutoConfig.from_pretrained(model_name) 

config.decoder_layers = 1
config.activation_function = config.hidden_act
config.decoder_ffn_dim = config.intermediate_size
config.init_std = config.initializer_range 

print(config)

model = RobertaAutoEncoder(config)
print(model)


training_args = PLMTrainingArgs(
    output_dir=checkpoint_dir, 
	overwrite_output_dir=False,
	do_train=True,
	do_eval=False,
	do_predict=False,
	evaluation_strategy='no',
	prediction_loss_only=False,
	per_device_train_batch_size=32, 
	per_device_eval_batch_size=8, 
	gradient_accumulation_steps=1,
	eval_accumulation_steps=32,
	learning_rate=1e-4,
	weight_decay=0.01,
	adam_beta1=0.9,
	adam_beta2=0.999,
	adam_epsilon=1e-8,
	max_grad_norm=1.,
	num_train_epochs=10, 
	max_steps=100000, 
	lr_scheduler_type='linear',
    warmup_steps=300,  
	save_steps=10000, 
	save_total_limit=100, 
	no_cuda=False,
	seed=61820, 
	local_rank=args.local_rank,
	dataloader_drop_last=False,
)

training_args.add_loss_weights(
    mlm=1, # mlm
    alpha = args.alpha, # gen
    beta = args.beta, # pm
    gamma = args.gamma #diversity
    )

train_dataset = PLMDataset(
    tokenizer=tokenizer,
    file_path=dataset_dir,
    block_size=128
)

data_collator = PLMDataCollator(tokenizer=tokenizer, mlm_probability=0.15)


trainer = PLMTrainer(
    model=model,                         
    args=training_args,
    data_collator=data_collator,                
    train_dataset=train_dataset            
)

run_name = args.name+"_mlm_" + str(trainer.args.mlm) + "_gen_" + str(trainer.args.alpha) + "_pm_" + str(trainer.args.beta) + "_div_" + str(trainer.args.gamma)
log_dir = "./tb_logs/" + run_name

trainer.load_tb(log_dir)

trainer.train()
trainer.save_model(checkpoint_dir)
