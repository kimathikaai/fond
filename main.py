import wandb
import argparse
import yaml
from src.utils.hparams import seed_hash, random_hparams

NUM_TEST_SETS = 4

def main():
    run = wandb.init()

    # Access sweep configuration
    kd_algo = wandb.config.kd_algo
    dataset = wandb.config.dataset
    test_set_id = wandb.config.test_set_id
    hparam_id = wandb.config.hparam_id
    trial_id = wandb.config.trial_id

    # Get hyperparameters
    hparam_seed = seed_hash(trial_id, hparam_id)
    hparams = random_hparams(kd_algo, dataset, hparam_seed)

    # TODO: function to run experiment with algorithm

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
  