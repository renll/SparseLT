# Copyright (c) Zixuan Zhang, Liliang Ren.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


from transformers import RobertaTokenizerFast, RobertaModel, AutoTokenizer
import torch
from model import RobertaAutoEncoder

t = AutoTokenizer.from_pretrained("bert-base-uncased")


ckpt_dirs = ["./checkpoints/YOUR_FULLMODEL_CKPT_DIR/"]
for ckpt_dir in ckpt_dirs:
    print(ckpt_dir.split("/")[-2])
    m = RobertaAutoEncoder.from_pretrained(ckpt_dir)
    
    
    input_sentences = ["She was murdered in her New York office, just days after learning that Waitress had been accepted into the Sundance Film Festival."]
   
    for input_sentence in input_sentences:
        print("INPUT SENTENCE: ")
        print(input_sentence + '\n')
        input_batch = t(input_sentence, return_tensors="pt")
        
        input_ids = input_batch["input_ids"]
        attn_mask = input_batch["attention_mask"]
        # print(input_ids.shape)
        input_tokens = [["CLS"] + t.tokenize(input_sentence) + ["SEP"]]
        
        output_ids = m.test_generate(input_ids=input_ids, attention_mask=attn_mask, original_tokens=input_tokens)
        sentence_output = t.decode(output_ids[0], skip_special_tokens=False)
        print("OUTPUT SENTENCE: ")
        print(sentence_output)
        print('\n')
