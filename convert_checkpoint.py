# Copyright (c) Liliang Ren.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.


from model import RobertaAutoEncoder
import os

in_dir = "./checkpoints/YOUR_MODEL_CKPT_DIR"
out_dir = "./checkpoints/for_container" # Assuming it is used for few-shot eval.

os.makedirs(out_dir, exist_ok=True)

m = RobertaAutoEncoder.from_pretrained(in_dir)
m.model.save_pretrained(out_dir)
