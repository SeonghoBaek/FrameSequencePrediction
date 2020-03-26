#!/bin/bash

python FSPNet.py --mode finetune --base_model_path ./model/m.ckpt --task_model_path ./task_model/m.ckpt --train_data input
