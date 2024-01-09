import wandb
import argparse
import yaml
from src.utils.hparams import seed_hash, random_hparams

last_test_set = {
    "pacs": [],
    "vlcs": [],
    "office_home": [],
}

def main():
    run = wandb.init()

    # Get test domain index based on last one run
    # TODO: Find a better way to get the test set from sweep
    if not run.config.datasets in last_test_set:
        raise KeyError("Invalid dataset name.")
    else:
        test_domain = len(last_test_set[run.config.datasets])
        last_test_set[run.config.datasets].append(test_domain)

    # Get hparams
    hparam_seed = seed_hash(run.config.trials, run.config.hparams)
    hparams = random_hparams(run.config.kd_algos, run.config.datasets, hparam_seed)

    # TODO: function to run experiment with algorithm

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_parameters = {
        "kd_algos": {
            "values": config["sweep_parameters"]["kd_algos"]
        },
        "datasets": {
            "values": config["sweep_parameters"]["datasets"]
        },
        "hparams": {
            "values": [i for i in range(config["sweep_parameters"]["n_hparams"])]
        },
        "trials": {
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
        project=config["wandb"]["project"]
    )

    wandb.agent(sweep_id, function=main, count=4)
  