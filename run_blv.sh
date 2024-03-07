#!/bin/bash

# local variables
gpu=0

source ../venv/bin/activate
CUDA_VISIBLE_DEVICES=${gpu} python -m main --wandb_entity critical-ml-dg --wandb_project fond_blv --sweep_id s7s6eqy8