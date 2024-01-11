import wandb
import argparse
import yaml

from src.utils.hparams import seed_hash, random_hparams
from src.train.fit import fit

NUM_TEST_SETS = 4

def main():
    run = wandb.init()

    # Access experiment parameters
    overall_seed = wandb.config.overall_seed
    data_dir = wandb.config.data_dir
    num_workers = wandb.config.num_workers
    holdout_fraction = wandb.config.holdout_fraction
    n_steps = wandb.config.n_steps
    checkpoint_freq = wandb.config.checkpoint_freq

    # Access sweep configuration
    kd_algo = wandb.config.kd_algo
    dataset = wandb.config.dataset
    test_set_id = wandb.config.test_set_id
    hparam_id = wandb.config.hparam_id
    trial_id = wandb.config.trial_id
    overlap = wandb.config.overlap

    fit(
        seed=overall_seed,
        trial_seed=trial_id,
        hparams_seed=hparam_id,
        algorithm_name=kd_algo,
        dataset_name=dataset,
        data_dir=data_dir,
        num_workers=num_workers,
        test_envs=[test_set_id],
        overlap_type=overlap,
        holdout_fraction=holdout_fraction,
        n_steps=n_steps,
        checkpoint_freq=checkpoint_freq
    )

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Format parameters to match sweep parameter formatting
    sweep_parameters = {
        "kd_algo": {
            "values": config["sweep_parameters"]["kd_algos"]
        },
        "dataset": {
            "values": config["sweep_parameters"]["datasets"]
        },
        "overlap": {
            "values": config["sweep_parameters"]["overlaps"]
        },
        "test_set_id": {
            "values": [i for i in range(NUM_TEST_SETS)]
        },
        "hparam_id": {
            "values": [i for i in range(config["sweep_parameters"]["n_hparams"])]
        },
        "trial_id": {
            "values": [i for i in range(config["sweep_parameters"]["n_trials"])]
        },
    }

    experiment_parameters = config["experiment_parameters"]
    for parameter in experiment_parameters:
        sweep_parameters[parameter] = {
            "value": experiment_parameters[parameter]
        }

    sweep_config = {
        **config["sweep_config"],
        "parameters": {
            **sweep_parameters
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"]
    )

    wandb.agent(sweep_id, function=main)
  
