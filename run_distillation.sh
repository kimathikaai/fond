#!/bin/bash

# local variables
gpu=1

source ../venv/bin/activate
CUDA_VISIBLE_DEVICES=${gpu} python -m main --wandb_entity critical-ml-dg --wandb_project fond_distillation_initial