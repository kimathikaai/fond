import argparse
import logging

import wandb
import yaml
from dotenv import load_dotenv

from src.train.fit import fit
from src.utils.hparams import random_hparams, seed_hash
from src.utils.misc import config_logging

NUM_TEST_SETS = 4


def main():
    run = wandb.init()

    # Access experiment parameters
    overall_seed = wandb.config.overall_seed
    data_dir = wandb.config.data_dir
    log_dir = wandb.config.log_dir
    num_workers = wandb.config.num_workers
    holdout_fraction = wandb.config.holdout_fraction
    n_steps = wandb.config.n_steps
    checkpoint_freq = wandb.config.checkpoint_freq
    model_checkpoint = wandb.config.model_checkpoint
    teacher_paths = (
        wandb.config.teacher_paths if "teacher_paths" in wandb.config else None
    )

    # Access sweep configuration
    kd_algo = wandb.config.kd_algo
    dataset = wandb.config.dataset
    test_set_id = wandb.config.test_set_id
    hparam_id = wandb.config.hparam_id
    trial_id = wandb.config.trial_id
    overlap = wandb.config.overlap

    fit(
        id="{}-{}-{}".format(run.project, run.sweep_id, run.id),
        seed=overall_seed,
        trial_seed=trial_id,
        hparams_seed=hparam_id,
        algorithm_name=kd_algo,
        dataset_name=dataset,
        data_dir=data_dir,
        log_dir=log_dir,
        num_workers=num_workers,
        test_envs=[test_set_id],
        overlap_type=overlap,
        holdout_fraction=holdout_fraction,
        n_steps=n_steps,
        checkpoint_freq=checkpoint_freq,
        model_checkpoint=model_checkpoint,
        teacher_paths=teacher_paths,
        num_domain_linked_classes=wandb.config.num_domain_linked_classes,
        num_classes=wandb.config.num_classes
    )

    wandb.finish()


if __name__ == "__main__":
    # load enviroment variables
    load_dotenv()
    # load cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sweep_config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # format logger
    config_logging()

    # Format parameters to match sweep parameter formatting
    sweep_parameters = {
        "kd_algo": {"values": config["sweep_parameters"]["kd_algos"]},
        "dataset": {"values": config["sweep_parameters"]["datasets"]},
        "overlap": {"values": config["sweep_parameters"]["overlaps"]},
        "num_classes": {"values": config["sweep_parameters"]["num_classes"]},
        "test_set_id": {"values": [i for i in range(NUM_TEST_SETS)]},
        "hparam_id": {
            "values": [i for i in range(config["sweep_parameters"]["n_hparams"])]
        },
        "trial_id": {
            "values": [i for i in range(config["sweep_parameters"]["n_trials"])]
        },
    }

    experiment_parameters = config["experiment_parameters"]
    for parameter in experiment_parameters:
        if parameter == "teacher_paths":
            for dataset in config["sweep_parameters"]["datasets"]:
                assert dataset in experiment_parameters[parameter].keys()
            for dataset in experiment_parameters[parameter].keys():
                assert len(experiment_parameters[parameter][dataset].keys()) == 4
        sweep_parameters[parameter] = {"value": experiment_parameters[parameter]}

    sweep_config = {**config["sweep_config"], "parameters": {**sweep_parameters}}

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
    )

    wandb.agent(sweep_id, function=main)
