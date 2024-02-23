import argparse
import logging
import os

import wandb
import yaml
from dotenv import load_dotenv
from lightning.pytorch.callbacks import checkpoint

from datasets import DATASETS
from networks import ALGORITHMS
from src.train.fit import fit
from src.utils.hparams import random_hparams, seed_hash
from src.utils.misc import config_logging


def get_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    # fmt:off
    # directories
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--log_dir", type=str)

    # training
    parser.add_argument("--algorithm", type=str, choices=list(ALGORITHMS.keys()))
    parser.add_argument("--checkpoint_freq", type=int, default=300)
    parser.add_argument("--n_steps", type=int, default=5001)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--overall_seed", type=int, default=0)
    parser.add_argument("--trial_id", type=int)

    # data
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()))
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--num_domain_linked_classes", type=int)
    parser.add_argument("--overlap_type", type=str)
    parser.add_argument("--test_domain_id", type=int)

    # checkpoints
    parser.add_argument("--checkpoint_maximize", type=str, choices=['true', 'false'])
    parser.add_argument("--checkpoint_metric", type=str)

    # hparams
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--hparams_seed", type=int)
    
    args = parser.parse_args()
    # fmt:on

    # Validate
    args.data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    assert os.path.isdir(args.data_dir), f"Directory {args.data_dir} not found"
    args.log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    assert os.path.isdir(args.log_dir), f"Directory {args.log_dir} not found"

    return args


if __name__ == "__main__":
    # load enviroment variables
    load_dotenv()
    # load cmd line arguments
    args = get_args()
    # format logger
    config_logging()
    # initialize run
    run = wandb.init()

    id = "{}-{}-{}".format(run.project, run.sweep_id, run.id)
    logging.info(f"Starting training for {id}")

    # run training
    fit(
        id=id,
        seed=args.overall_seed,
        trial_seed=args.trial_seed,
        hparams_seed=args.hparams_seed,
        algorithm_name=args.algorithm,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        num_workers=args.num_workers,
        test_envs=[args.test_domain_id],
        overlap_type=args.overlap_type,
        holdout_fraction=args.holdout_fraction,
        n_steps=args.n_steps,
        checkpoint_freq=args.checkpoint_freq,
        model_checkpoint={
            "maximize": True if args.checkpoint_maximize == "true" else False,
            "metric": args.checkpoint_metric,
        },
        teacher_paths=None,
        num_domain_linked_classes=args.num_domain_linked_classes,
        num_classes=args.num_classes,
    )

    wandb.finish()
